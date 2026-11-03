#!/usr/bin/env Rscript
# =============================================================================
# make_lidar_products.R
# -----------------------------------------------------------------------------
# Passo 1 do orquestrador (abordagem C): gera MDT, MDS e CHM por arquivo .laz
# usando lidR (ground CSF/classe 2 + lspikefree), como no fluxo do laboratorio.
#
# Diferente do atalho PDAL (DTM = min global), aqui:
#   - MDT: pontos de solo (classe 2) -> TIN
#   - MDS: 1o retorno -> lspikefree()
#   - CHM: normaliza a nuvem pelo MDT, remove ruido, lspikefree()
#
# Reentrante: pula .laz cujos 3 TIFs ja existem (a menos de --force).
#
# Uso:
#   Rscript make_lidar_products.R --laz-dir "<pasta com .laz>" \
#       --out-dir "<pasta saida>" [--res 0.5] [--chm-max 120] [--force]
# =============================================================================

suppressPackageStartupMessages({
  library(lidR)
  library(terra)
})

# ---- parse de argumentos simples (--chave valor / --flag) -------------------
parse_args <- function(args) {
  out <- list(
    laz_dir = NULL, out_dir = NULL, res = 0.5, chm_max = 120,
    threads = 12, force = FALSE, ground_class_only = TRUE,
    names = NULL  # opcional: stems separados por virgula (filtra o pool)
  )
  i <- 1
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--laz-dir")      { out$laz_dir <- args[[i + 1]]; i <- i + 2 }
    else if (a == "--out-dir") { out$out_dir <- args[[i + 1]]; i <- i + 2 }
    else if (a == "--res")     { out$res <- as.numeric(args[[i + 1]]); i <- i + 2 }
    else if (a == "--chm-max") { out$chm_max <- as.numeric(args[[i + 1]]); i <- i + 2 }
    else if (a == "--threads") { out$threads <- as.integer(args[[i + 1]]); i <- i + 2 }
    else if (a == "--names")   { out$names <- args[[i + 1]]; i <- i + 2 }
    else if (a == "--force")   { out$force <- TRUE; i <- i + 1 }
    else if (a == "--csf")     { out$ground_class_only <- FALSE; i <- i + 1 }
    else { message("Ignorando argumento desconhecido: ", a); i <- i + 1 }
  }
  out
}

a <- parse_args(commandArgs(trailingOnly = TRUE))
if (is.null(a$laz_dir)) stop("--laz-dir obrigatorio")
if (is.null(a$out_dir)) a$out_dir <- file.path(a$laz_dir, "saida_produtos")

set_lidr_threads(a$threads)
options(lidR.check.nested = FALSE)

if (!dir.exists(a$laz_dir)) stop("Diretorio nao encontrado: ", a$laz_dir)
if (!dir.exists(a$out_dir)) dir.create(a$out_dir, recursive = TRUE)

# CSF so eh usado se --csf (senao usa classe 2 ja existente)
use_csf <- !a$ground_class_only
if (use_csf) suppressPackageStartupMessages(library(RCSF))

lista_laz <- list.files(a$laz_dir, pattern = "\\.laz$", full.names = TRUE,
                        recursive = FALSE, ignore.case = TRUE)
if (length(lista_laz) == 0) stop("Nenhum .laz em: ", a$laz_dir)

# Filtro opcional: so processa stems listados (--names a,b,c). Sem pasta tmp.
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

message(">>> ", length(lista_laz), " arquivo(s) .laz | res=", a$res,
        " m | chm_max=", a$chm_max, " m | ground=",
        if (use_csf) "CSF" else "classe2", " | out=", a$out_dir)

for (i in seq_along(lista_laz)) {
  caminho_laz <- lista_laz[[i]]
  nome_base <- gsub("\\.laz$", "", basename(caminho_laz), ignore.case = TRUE)

  nome_mdt <- file.path(a$out_dir, paste0(nome_base, "_MDT.tif"))
  nome_mds <- file.path(a$out_dir, paste0(nome_base, "_MDS.tif"))
  nome_chm <- file.path(a$out_dir, paste0(nome_base, "_CHM.tif"))

  if (!a$force && file.exists(nome_mdt) && file.exists(nome_mds) && file.exists(nome_chm)) {
    message(sprintf("[%d/%d] JA PROCESSADO (pulando): %s",
                    i, length(lista_laz), basename(caminho_laz)))
    next
  }

  message(sprintf("\n[%d/%d] PROCESSANDO: %s", i, length(lista_laz), basename(caminho_laz)))
  las <- readLAS(caminho_laz, select = "xyzcr", filter = "-drop_z_below 0")
  if (is.empty(las)) { warning("Vazio/sem pontos: ", basename(caminho_laz)); next }

  # (opcional) classificar solo por CSF quando nao confiamos na classe 2
  if (use_csf) {
    message("  -> Classificando solo (CSF)...")
    las <- classify_ground(las, algorithm = csf())
  }

  # A) MDT (classe 2 - solo)
  message("  -> MDT...")
  las_ground <- filter_poi(las, Classification == 2L)
  dtm <- NULL
  if (npoints(las_ground) > 0) {
    dtm <- rasterize_terrain(las_ground, res = a$res, algorithm = tin())
    terra::writeRaster(dtm, nome_mdt, overwrite = TRUE)
  } else {
    warning("  -> Sem pontos de solo (classe 2); MDT nao gerado.")
  }

  # B) MDS (1o retorno, lspikefree)
  message("  -> MDS (lspikefree)...")
  las_first <- filter_poi(las, ReturnNumber == 1L)
  dsm <- rasterize_canopy(las_first, res = a$res, algorithm = lspikefree())
  terra::writeRaster(dsm, nome_mds, overwrite = TRUE)

  # C) CHM via normalizacao 3D + lspikefree
  if (!is.null(dtm)) {
    message("  -> Normalizando nuvem pelo MDT...")
    las_norm <- normalize_height(las, dtm)
    message("  -> Removendo ruido (ivf) e limites de altura...")
    las_norm <- classify_noise(las_norm, ivf(res = 5, n = 6))
    las_norm <- filter_poi(las_norm, Classification != LASNOISE & Z >= 0 & Z <= a$chm_max)
    message("  -> CHM (lspikefree)...")
    chm <- rasterize_canopy(las_norm, res = a$res, algorithm = lspikefree())
    chm[is.na(chm) | chm < 0] <- 0
    terra::writeRaster(chm, nome_chm, overwrite = TRUE)
    rm(las_norm, chm)
  }

  rm(las, las_ground, las_first, dtm, dsm); gc()
  message("  -> OK")
}

message("\n>>> Produtos LiDAR prontos em: ", a$out_dir)
