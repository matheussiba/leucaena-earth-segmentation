#!/usr/bin/env Rscript
# =============================================================================
# make_lidar_products_geotrees.R
# -----------------------------------------------------------------------------
# Engine "geotrees": protocolo GeoTrees / GCA (Fischer et al. 2024,
# Methods Ecol Evol; https://doi.org/10.1111/2041-210X.14416) — o mesmo
# workflow testado nos dados IGC (ALS_processing.R nos Downloads).
#
# Dois modos (--protocol):
#   perfile  (default)  CHM por .laz no estilo open-source do helper:
#                       DTM (TIN classe 2) + DSM (1º retorno + lspikefree) →
#                       CHM = clip(DSM − DTM, 0, height_lim).
#                       Saída: <stem>_CHM.tif  (compatível com align Engine=r).
#   catalog             Chama process.dataset() do helper completo (pasta
#                       geotrees/). Requer lidR, terra, sf, data.table, etc.
#                       e opcionalmente LAStools (LASTOOLS_PATH / --path-lastools).
#                       Depois copia chm_lspikefree.tif → cada <stem>_CHM.tif
#                       (mesmo mosaico; o align recorta à grade óptica).
#
# Uso:
#   Rscript make_lidar_products_geotrees.R --laz-dir <raw> --out-dir <products> \
#       [--names a,b,c] [--res 1] [--chm-max 125] [--workers 4] [--threads 12] \
#       [--protocol perfile|catalog] [--path-lastools ""] [--force]
# =============================================================================

suppressPackageStartupMessages({
  library(lidR)
  library(terra)
})

parse_args <- function(args) {
  out <- list(
    laz_dir = NULL, out_dir = NULL, res = 1.0, chm_max = 125,
    max_edge = 50, threads = 12, workers = 4, force = FALSE,
    names = NULL, protocol = "perfile", path_lastools = "",
    size_tile = 100, size_buffer = 50, tmpdir = NULL
  )
  i <- 1
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--laz-dir")      { out$laz_dir <- args[[i + 1]]; i <- i + 2 }
    else if (a == "--out-dir") { out$out_dir <- args[[i + 1]]; i <- i + 2 }
    else if (a == "--tmpdir")  { out$tmpdir <- args[[i + 1]]; i <- i + 2 }
    else if (a == "--res") {
      out$res <- as.numeric(sub(",", ".", args[[i + 1]], fixed = TRUE)); i <- i + 2
    }
    else if (a == "--max-edge") {
      out$max_edge <- as.numeric(sub(",", ".", args[[i + 1]], fixed = TRUE)); i <- i + 2
    }
    else if (a == "--chm-max") { out$chm_max <- as.numeric(args[[i + 1]]); i <- i + 2 }
    else if (a == "--threads") { out$threads <- as.integer(args[[i + 1]]); i <- i + 2 }
    else if (a == "--workers") { out$workers <- as.integer(args[[i + 1]]); i <- i + 2 }
    else if (a == "--names")   { out$names <- args[[i + 1]]; i <- i + 2 }
    else if (a == "--protocol"){ out$protocol <- tolower(args[[i + 1]]); i <- i + 2 }
    else if (a == "--path-lastools") { out$path_lastools <- args[[i + 1]]; i <- i + 2 }
    else if (a == "--size-tile")   { out$size_tile <- as.numeric(args[[i + 1]]); i <- i + 2 }
    else if (a == "--size-buffer") { out$size_buffer <- as.numeric(args[[i + 1]]); i <- i + 2 }
    else if (a == "--force")   { out$force <- TRUE; i <- i + 1 }
    else { message("Ignorando argumento desconhecido: ", a); i <- i + 1 }
  }
  out
}

# ---- GeoTrees per-file (open-source branch do helper) -----------------------
# Espelha make.dsm_locallyadaptive (OS) + DTM + CHM = DSM - DTM.
process_one_laz_geotrees <- function(caminho_laz, out_dir, res, chm_max, force,
                                     threads_per_worker, max_edge = 50) {
  set_lidr_threads(threads_per_worker)
  options(lidR.check.nested = FALSE)

  nome_base <- gsub("\\.laz$", "", basename(caminho_laz), ignore.case = TRUE)
  nome_mdt <- file.path(out_dir, paste0(nome_base, "_MDT.tif"))
  nome_dsm <- file.path(out_dir, paste0(nome_base, "_DSM_lspikefree.tif"))
  nome_chm <- file.path(out_dir, paste0(nome_base, "_CHM.tif"))

  if (!force && file.exists(nome_chm) && file.exists(nome_mdt)) {
    return(sprintf("SKIP %s (ja processado)", basename(caminho_laz)))
  }

  t0 <- Sys.time()
  message(sprintf(
    "\n[GEOTREES/perfile] %s | res=%g m | height_lim=%g | lspikefree(height_buffer=0.5)",
    basename(caminho_laz), res, chm_max
  ))

  las <- readLAS(caminho_laz, select = "xyzcr", filter = "-drop_z_below 0")
  if (is.empty(las)) {
    return(sprintf("WARN %s vazio", basename(caminho_laz)))
  }

  # DTM: solo classe 2 + TIN (max_edge evita preencher buracoes) — alinhado ao Engine=r
  message("  -> MDT (TIN classe 2, max_edge=", max_edge, " m)...")
  las_ground <- filter_poi(las, Classification == 2L)
  if (npoints(las_ground) == 0) {
    return(sprintf("WARN %s sem pontos classe 2 (solo)", basename(caminho_laz)))
  }
  tin_algo <- if (is.finite(max_edge) && max_edge > 0) tin(max_edge = max_edge) else tin()
  dtm <- rasterize_terrain(las_ground, res = res, algorithm = tin_algo)
  terra::writeRaster(dtm, nome_mdt, overwrite = TRUE, NAflag = -9999)

  # DSM GeoTrees OS: só 1º retorno + lspikefree(height_buffer=0.5)
  # (params adaptive multi/slope/offset são do ramo LAStools; OS usa height_buffer)
  message("  -> DSM (1st return + lspikefree height_buffer=0.5)...")
  las_first <- filter_poi(las, ReturnNumber == 1L)
  if (is.empty(las_first) || npoints(las_first) == 0) {
    return(sprintf("WARN %s sem 1os retornos", basename(caminho_laz)))
  }
  dsm <- rasterize_canopy(las_first, res = res,
                          algorithm = lspikefree(height_buffer = 0.5))
  terra::writeRaster(dsm, nome_dsm, overwrite = TRUE, NAflag = -9999)

  # CHM = DSM - DTM (protocolo GeoTrees); gaps → 0
  message("  -> CHM = DSM - DTM (clip 0..", chm_max, ")...")
  # alinha grades se necessario
  if (!terra::compareGeom(dsm, dtm, stopOnError = FALSE, ext = TRUE, rowcol = TRUE)) {
    dtm <- terra::resample(dtm, dsm, method = "near")
  }
  chm <- dsm - dtm
  chm[!is.finite(chm)] <- 0
  chm[chm < 0] <- 0
  chm[chm > chm_max] <- chm_max
  terra::writeRaster(chm, nome_chm, overwrite = TRUE)

  rm(las, las_ground, las_first, dtm, dsm, chm)
  gc()
  dt <- round(difftime(Sys.time(), t0, units = "mins"), 1)
  sprintf("OK %s (%.1f min)", basename(caminho_laz), as.numeric(dt))
}

# ---- catalog: process.dataset do helper vendorizado -------------------------
.script_dir <- function() {
  ca <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", ca[grep("^--file=", ca)])
  if (length(f) == 1 && nzchar(f)) return(dirname(normalizePath(f)))
  getwd()
}

run_catalog_geotrees <- function(a, lista_laz) {
  helper <- file.path(.script_dir(), "geotrees", "ALS_processing_helperfunctions.R")
  if (!file.exists(helper)) {
    stop("Helper GeoTrees nao encontrado: ", helper)
  }
  message("Sourcing helper: ", helper)
  suppressPackageStartupMessages({
    library(data.table)
  })
  source(helper, local = FALSE)

  tmp_root <- if (!is.null(a$tmpdir) && nzchar(a$tmpdir)) a$tmpdir else
    file.path(a$out_dir, "_geotrees_tmp")
  dir_in  <- file.path(tmp_root, "input", "aoi")
  dir_out <- file.path(tmp_root, "processed")
  dir_tmp <- file.path(tmp_root, "workdir")
  unlink(tmp_root, recursive = TRUE, force = TRUE)
  dir.create(dir_in, recursive = TRUE, showWarnings = FALSE)
  dir.create(dir_out, recursive = TRUE, showWarnings = FALSE)
  dir.create(dir_tmp, recursive = TRUE, showWarnings = FALSE)

  for (f in lista_laz) {
    file.copy(f, file.path(dir_in, basename(f)), overwrite = TRUE)
  }

  path_lt <- a$path_lastools
  if (!nzchar(path_lt)) {
    path_lt <- Sys.getenv("LASTOOLS_PATH", unset = "")
  }

  message(">>> GeoTrees catalog | res=", a$res, " | tile=", a$size_tile,
          " | buffer=", a$size_buffer, " | cores=", a$workers,
          " | lastools=", ifelse(nzchar(path_lt), path_lt, "(open-source lidR)"))

  results <- process.dataset(
    name_job = "phd_geotrees",
    type_file = "laz",
    dir_dataset = dirname(dir_in),
    dir_processed = dir_out,
    tmpdir_processing = dir_tmp,
    path_lastools = path_lt,
    resolution = a$res,
    n_cores = a$workers,
    size_tile = a$size_tile,
    size_buffer = a$size_buffer,
    force.utm = FALSE,
    force.recompute = a$force,
    remove.vlr = FALSE,
    remove.evlr = FALSE,
    use.blast2dem = nzchar(path_lt),
    type_architecture = "64",
    cleanup = TRUE,
    retile = FALSE,
    height_lim = a$chm_max,
    types_dsm = c("tin", "lspikefree"),
    params_dsmadaptive = data.table(
      multi = 3.1, slope = 1.75, offset = 2.1
    )
  )

  chms <- list.files(dir_out, pattern = "chm_lspikefree\\.tif$",
                     recursive = TRUE, full.names = TRUE)
  if (length(chms) == 0) {
    stop("GeoTrees catalog nao gerou chm_lspikefree.tif em ", dir_out)
  }
  chm_src <- chms[[1]]
  message("CHM GeoTrees: ", chm_src)

  for (f in lista_laz) {
    stem <- gsub("\\.laz$", "", basename(f), ignore.case = TRUE)
    dest <- file.path(a$out_dir, paste0(stem, "_CHM.tif"))
    file.copy(chm_src, dest, overwrite = TRUE)
    message("  -> ", dest)
  }
  invisible(results)
  invisible(chm_src)
}

# ---- main -------------------------------------------------------------------
a <- parse_args(commandArgs(trailingOnly = TRUE))
if (is.null(a$laz_dir)) stop("--laz-dir obrigatorio")
if (is.null(a$out_dir)) a$out_dir <- file.path(a$laz_dir, "saida_produtos_geotrees")
if (!dir.exists(a$laz_dir)) stop("Diretorio nao encontrado: ", a$laz_dir)
if (!dir.exists(a$out_dir)) dir.create(a$out_dir, recursive = TRUE)
if (!a$protocol %in% c("perfile", "catalog")) {
  stop("--protocol deve ser perfile ou catalog, veio: ", a$protocol)
}

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
message(">>> GeoTrees | protocol=", a$protocol, " | ", n_laz, " .laz | res=", a$res,
        " m | chm_max=", a$chm_max, " | out=", a$out_dir)

if (a$protocol == "catalog") {
  run_catalog_geotrees(a, lista_laz)
  message("\n>>> GeoTrees catalog concluido. Produtos em: ", a$out_dir)
  quit(status = 0)
}

workers <- max(1L, min(as.integer(a$workers), n_laz))
threads_per <- max(1L, as.integer(floor(a$threads / workers)))
message("workers=", workers, " | lidR_threads/worker=", threads_per)

if (workers <= 1L) {
  set_lidr_threads(threads_per)
  options(lidR.check.nested = FALSE)
  results <- lapply(lista_laz, process_one_laz_geotrees,
                    out_dir = a$out_dir, res = a$res, chm_max = a$chm_max,
                    force = a$force, threads_per_worker = threads_per,
                    max_edge = a$max_edge)
} else {
  cl <- parallel::makeCluster(workers, outfile = "")
  on.exit(try(parallel::stopCluster(cl), silent = TRUE), add = TRUE)
  parallel::clusterEvalQ(cl, {
    suppressPackageStartupMessages({
      library(lidR)
      library(terra)
    })
    NULL
  })
  parallel::clusterExport(cl, varlist = c("process_one_laz_geotrees"),
                          envir = environment())
  results <- parallel::parLapply(cl, lista_laz, process_one_laz_geotrees,
                                 out_dir = a$out_dir, res = a$res,
                                 chm_max = a$chm_max, force = a$force,
                                 threads_per_worker = threads_per,
                                 max_edge = a$max_edge)
  parallel::stopCluster(cl)
  on.exit(NULL)
}

message("\n--- resumo ---")
for (r in results) message("  ", r)
message("\n>>> Produtos GeoTrees prontos em: ", a$out_dir)
