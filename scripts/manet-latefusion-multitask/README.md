# Scripts: MANet + Late Fusion + Multitask

Codigo de teste (nao pesos). Pesos/logs/predicoes ficam na AOI:

`C:\00_DATASETS_AI\261002-riodaspedras-aoi\experiments\manet-latefusion-multitask\`

No repo, `experiments/exp_*` continua so para artefatos (.pt, logs) do pipeline antigo.

## Rodar

```powershell
cd scripts\manet-latefusion-multitask
.\run_01_build_labels.ps1   # ja rodou uma vez
.\run_02_train_docker.ps1   # precisa imagem leucaena-segmentation:cuda
.\run_03_predict_docker.ps1
.\run_04_postprocess.ps1
```

Build da imagem (raiz do repo, so se ainda nao existir): `docker compose build`
