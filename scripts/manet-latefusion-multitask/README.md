# Scripts: MANet + Late Fusion + Multitask

Codigo de teste. Pesos/logs/predicoes ficam na AOI:

`<DIR_DATASETS_AI_PATH>/<LEUCAENA_DATASET_ID>/experiments/manet-latefusion-multitask/`

## Multi-PC (casa vs laboratorio)

No `.env` da **raiz do repo** (gitignored; cada maquina tem o seu):

```env
DIR_DATASETS_AI_PATH=E:/dados-msb/00_DATASETS_AI
LEUCAENA_DATASET_ID=261002-riodaspedras-aoi
TRAIN_PROFILE=lab
```

| `TRAIN_PROFILE` | batch | workers | shm |
|-----------------|-------|---------|-----|
| `home` | 8 | 2 | 8gb |
| `lab` | 14 | 8 | 16gb |

Casa tipica: `DIR_DATASETS_AI_PATH=C:/00_DATASETS_AI` + `TRAIN_PROFILE=home`.  
O git sincroniza os scripts; o `.env` local nao vai no git.

Veja `.env.example` na raiz.

## Rodar

```powershell
cd scripts\manet-latefusion-multitask
.\run_01_build_labels.ps1
.\run_02_train_docker.ps1
.\run_03_predict_docker.ps1
.\run_04_postprocess.ps1
```

Build da imagem (raiz, so se ainda nao existir): `docker compose build`
