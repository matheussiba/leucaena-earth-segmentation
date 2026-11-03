#!/usr/bin/env Rscript
# =============================================================================
# make_lidar_products.R
# -----------------------------------------------------------------------------
# Passo 1 do orquestrador (abordagem C): gera MDT, MDS e CHM por arquivo .laz
# usando lidR (ground CSF/classe 2 + lspikefree).
#
#   - MDT: pontos de solo (classe 2) -> TIN
#   - CHM: extract bilinear do MDT (buracos NA ignorados, SEM IDW), ruido, lspikefree()
#   - MDS: opcional (--mds); nao e usado no treino late-fusion
#
# Paralelismo: --workers N processa N LAZ ao mesmo tempo (PSOCK). Threads do
# lidR por worker = floor(--threads / --workers) para nao estourar a CPU.
#
# Uso:
#   Rscript make_lidar_products.R --laz-dir <raw> --out-dir <products> \
#       [--names a,b,c] [--res 0.5] [--chm-max 120] [--workers 4] [--threads 12] [--force]
# =============================================================================

suppressPackageStartupMessages({
  library(lidR)
  library(terra)
})

parse_args <- function(args) {
  out <- list(
    laz_dir = NULL, out_dir = NULL, res = 0.5, chm_max = 120,
    threads = 12, workers = 4, force = FALSE, ground_class_only = TRUE,
    names = NULL, mds = FALSE
  )
  i <- 1
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--laz-dir")      { out$laz_dir <- args[[i + 1]]; i <- i + 2 }
    else if (a == "--out-dir") { out$out_dir <- args[[i + 1]]; i <- i + 2 }
    else if (a == "--res") {
      out$res <- as.numeric(sub(",", ".", args[[i + 1]], fixed = TRUE)); i <- i + 2
    }
    else if (a == "--chm-max") { out$chm_max <- as.numeric(args[[i + 1]]); i <- i + 2 }
    else if (a == "--threads") { out$threads <- as.integer(args[[i + 1]]); i <- i + 2 }
    else if (a == "--workers") { out$workers <- as.integer(args[[i + 1]]); i <- i + 2 }
    else if (a == "--names")   { out$names <- args[[i + 1]]; i <- i + 2 }
    else if (a == "--force")   { out$force <- TRUE; i <- i + 1 }
    else if (a == "--csf")     { out$ground_class_only <- FALSE; i <- i + 1 }
    else if (a == "--mds")     { out$mds <- TRUE; i <- i + 1 }
    else { message("Ignorando argumento desconhecido: ", a); i <- i + 1 }
  }
  out
}

# ---- um LAZ -----------------------------------------------------------------
process_one_laz <- function(caminho_laz, out_dir, res, chm_max, force, use_csf,
                            threads_per_worker, do_mds = FALSE) {
  set_lidr_threads(threads_per_worker)
  options(lidR.check.nested = FALSE)

  nome_base <- gsub("\\.laz$", "", basename(caminho_laz), ignore.case = TRUE)
  nome_mdt <- file.path(out_dir, paste0(nome_base, "_MDT.tif"))
  nome_mds <- file.path(out_dir, paste0(nome_base, "_MDS.tif"))
  nome_chm <- file.path(out_dir, paste0(nome_base, "_CHM.tif"))

  # Treino so precisa de CHM (+ MDT intermediario). MDS e opcional.
  if (!force && file.exists(nome_mdt) && file.exists(nome_chm) &&
      (!do_mds || file.exists(nome_mds))) {
    return(sprintf("SKIP %s (ja processado)", basename(caminho_laz)))
  }

  t0 <- Sys.time()
  message(sprintf("\n[WORKER] PROCESSANDO: %s (lidR threads=%d)",
                  basename(caminho_laz), threads_per_worker))

  las <- readLAS(caminho_laz, select = "xyzcr", filter = "-drop_z_below 0")
  if (is.empty(las)) {
    return(sprintf("WARN %s vazio", basename(caminho_laz)))
  }

  if (use_csf) {
    message("  -> Classificando solo (CSF)...")
    las <- classify_ground(las, algorithm = csf())
  }

  message("  -> MDT...")
  las_ground <- filter_poi(las, Classification == 2L)
  dtm <- NULL
  if (npoints(las_ground) > 0) {
    dtm <- rasterize_terrain(las_ground, res = res, algorithm = tin())
    terra::writeRaster(dtm, nome_mdt, overwrite = TRUE)
  } else {
    warning("  -> Sem pontos de solo (classe 2); MDT nao gerado.")
  }

  if (do_mds) {
    message("  -> MDS (lspikefree)...")
    las_first <- filter_poi(las, ReturnNumber == 1L)
    dsm <- rasterize_canopy(las_first, res = res, algorithm = lspikefree())
    terra::writeRaster(dsm, nome_mds, overwrite = TRUE)
    rm(las_first, dsm)
  }

  if (!is.null(dtm)) {
    message("  -> Normalizando nuvem pelo MDT (sem IDW; ignora buracos)...")
    z0 <- as.numeric(terra::extract(dtm, cbind(las$X, las$Y), method = "bilinear")[, 1])
    n_tot <- length(z0)
    n_na <- sum(is.na(z0) | !is.finite(z0))
    message(sprintf("     MDT valido: %s / %s pontos (buraco ignorado: %.1f%%)",
                    format(n_tot - n_na, big.mark = ","),
                    format(n_tot, big.mark = ","),
                    100 * n_na / max(n_tot, 1)))
    if (n_na < n_tot) {
      las <- add_lasattribute(las, z0, name = "Zground", desc = "MDT extract")
      las_norm <- filter_poi(las, !is.na(Zground) & is.finite(Zground))
      las_norm$Z <- las_norm$Z - las_norm$Zground
      message("  -> Removendo ruido (ivf) e limites de altura...")
      las_norm <- classify_noise(las_norm, ivf(res = 5, n = 6))
      las_norm <- filter_poi(las_norm, Classification != LASNOISE & Z >= 0 & Z <= chm_max)
      if (!is.empty(las_norm) && npoints(las_norm) > 0) {
        message("  -> CHM (lspikefree)...")
        chm <- rasterize_canopy(las_norm, res = res, algorithm = lspikefree())
        chm[is.na(chm) | chm < 0] <- 0
        terra::writeRaster(chm, nome_chm, overwrite = TRUE)
        rm(chm)
      } else {
        warning("  -> Sem pontos apos filtro; CHM nao gerado.")
      }
      rm(las_norm)
    } else {
      warning("  -> Nenhum ponto com MDT valido; CHM nao gerado.")
    }
  }

  rm(las, las_ground, dtm)
  gc()
  dt <- round(difftime(Sys.time(), t0, units = "mins"), 1)
  sprintf("OK %s (%.1f min)", basename(caminho_laz), as.numeric(dt))
}

# ---- main -------------------------------------------------------------------
a <- parse_args(commandArgs(trailingOnly = TRUE))
if (is.null(a$laz_dir)) stop("--laz-dir obrigatorio")
if (is.null(a$out_dir)) a$out_dir <- file.path(a$laz_dir, "saida_produtos")
if (!dir.exists(a$laz_dir)) stop("Diretorio nao encontrado: ", a$laz_dir)
if (!dir.exists(a$out_dir)) dir.create(a$out_dir, recursive = TRUE)

use_csf <- !a$ground_class_only
if (use_csf) suppressPackageStartupMessages(library(RCSF))

lista_laz <- list.files(a$laz_dir, pattern = "\\.laz$", full.names = TRUE,
                        recursive = FALSE, ignore.case = TRUE)
if (length(lista_laz) == 0) stop("Nenhum .laz em: ", a$laz_dir)

if (!is.null(a$names) && nzchar(a$names)) {
  want <- trimws(unlist(strsplit(a$names, ",", fixed = TRUE)))
  want <- want[nzchar(want)]
  stems <- gsub("\\.laz$", "", basename(lista_laz), ignore.case = TRUE)
  keep <- stems %in% want
  missing <- setdiff(want, stems)
  if (length(missing) > 0) {
    message("AVISO: stems pedidas sem .laz no pool: ", paste(missing, collapse = ", "))
  }
  lista_laz <- lista_laz[keep]
  if (length(lista_laz) == 0) stop("Nenhum .laz restou apos --names")
}

n_laz <- length(lista_laz)
workers <- max(1L, min(as.integer(a$workers), n_laz))
threads_per <- max(1L, as.integer(floor(a$threads / workers)))

message(">>> ", n_laz, " arquivo(s) .laz | res=", a$res,
        " m | chm_max=", a$chm_max, " m | ground=",
        if (use_csf) "CSF" else "classe2",
        " | mds=", a$mds,
        " | workers=", workers, " | lidR_threads/worker=", threads_per,
        " | out=", a$out_dir)

if (workers <= 1L) {
  set_lidr_threads(threads_per)
  options(lidR.check.nested = FALSE)
  results <- lapply(lista_laz, process_one_laz,
                    out_dir = a$out_dir, res = a$res, chm_max = a$chm_max,
                    force = a$force, use_csf = use_csf,
                    threads_per_worker = threads_per, do_mds = a$mds)
} else {
  # Windows: PSOCK. Cada worker carrega lidR/terra e processa 1+ LAZ.
  cl <- parallel::makeCluster(workers, outfile = "")
  on.exit(try(parallel::stopCluster(cl), silent = TRUE), add = TRUE)
  parallel::clusterEvalQ(cl, {
    suppressPackageStartupMessages({
      library(lidR)
      library(terra)
    })
    NULL
  })
  if (use_csf) {
    parallel::clusterEvalQ(cl, {
      suppressPackageStartupMessages(library(RCSF))
      NULL
    })
  }
  parallel::clusterExport(cl, varlist = c("process_one_laz"), envir = environment())
  results <- parallel::parLapply(cl, lista_laz, process_one_laz,
                                 out_dir = a$out_dir, res = a$res, chm_max = a$chm_max,
                                 force = a$force, use_csf = use_csf,
                                 threads_per_worker = threads_per, do_mds = a$mds)
  parallel::stopCluster(cl)
  on.exit(NULL)
}

message("\n--- resumo ---")
for (r in results) message("  ", r)
message("\n>>> Produtos LiDAR prontos em: ", a$out_dir)
