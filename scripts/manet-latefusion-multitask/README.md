# Scripts: MANet + Late Fusion + Multitask

## Versoes (nao misturar pastas)

| Versao | Pasta na AOI | Notas |
|--------|----------------|-------|
| **v1** | `experiments/manet-latefusion-multitask/` | Treino antigo (patch 256). **Nao apagar.** |
| **v2** | `experiments/manet-latefusion-multitask-v2/` | Patch **512**, filtro CHM>2 **so com LiDAR** (INT>0); gaps mantem poligono |

No `.env` da raiz:

```env
DIR_DATASETS_AI_PATH=E:/dados-msb/00_DATASETS_AI
LEUCAENA_DATASET_ID=261002-riodaspedras-aoi
TRAIN_PROFILE=lab
MANET_EXP_NAME=manet-latefusion-multitask-v2
```

Para voltar a predizer a v1: `MANET_EXP_NAME=manet-latefusion-multitask`.

## Perfis

| `TRAIN_PROFILE` | patch | batch (v2) | workers |
|-----------------|-------|------------|---------|
| `home` | 512 | 2 | 2 |
| `lab` | 512 | 4 | 8 |

## Rodar v2

```powershell
git pull
cd scripts\manet-latefusion-multitask
.\run_01_build_labels.ps1
.\run_02_train_docker.ps1
.\run_03_predict_docker.ps1
.\run_04_postprocess.ps1
```
