# GUIDE.md - Rodando o pipeline por partes

Este guia e para rodar o pipeline **uma etapa por vez**, para descobrir
exatamente onde esta o erro quando o `run_pipeline.py` completo nao funciona.

Use este arquivo sempre que quiser testar do zero, retomar uma etapa, ou
entender onde ficam os arquivos gerados.

---

## 0. Antes de comecar

Abra o **PowerShell normal do Windows**.

Entre na pasta do repositorio:

```powershell
cd "C:\Users\mathe\OneDrive\Documents\0-GITHUB\leucaena-earth-segmentation"
```

Defina estas variaveis para nao precisar repetir caminhos enormes:

```powershell
$AOI = "G:\My Drive\PHD\02-Tese\02-data\adote-uma-leucena\v1-LEUCENA MAPPING\gdb-leucena_v2.gpkg"
$LAYER = "articulacao_laser_voo22_AOI_treino"
$SOURCE = "D:\"
$DEST = "C:\00_DATASETS_AI\260515-piracicaba-aoi"
```

Estrutura esperada:

```text
D:\
+-- rgb\
+-- ir\
+-- laz\

C:\00_DATASETS_AI\260515-piracicaba-aoi\
+-- annotations\
|   +-- polygons.geojson
+-- opt\
|   +-- raw\
|   |   +-- rgb\
|   |   +-- ir\
|   +-- rgbnir\
+-- lidar\
|   +-- raw\
|   +-- chm\
+-- patches\
+-- models\
    +-- logs\
```

---

## 1. Checagens basicas

### 1.1 Ver se as pastas existem

```powershell
Test-Path $AOI
Test-Path "D:\rgb"
Test-Path "D:\ir"
Test-Path "D:\laz"
Test-Path "$DEST\annotations\polygons.geojson"
```

Tudo deve retornar `True`.

### 1.2 Contar arquivos de entrada

```powershell
(Get-ChildItem "D:\rgb" -File).Count
(Get-ChildItem "D:\ir" -File).Count
(Get-ChildItem "D:\laz" -File).Count
```

Se algum resultado for `0`, a etapa correspondente nao vai funcionar.

### 1.3 Ver se Python tem as bibliotecas das etapas 1 e 2

```powershell
python -c "import geopandas, rasterio; print('geopandas OK'); print('rasterio OK')"
```

Se der erro, instale:

```powershell
pip install -r requirements.txt
```

### 1.4 Ver se Docker esta pronto para etapas 3, 4 e 5

Abra o **Docker Desktop** e espere ficar rodando.

Depois rode:

```powershell
docker image inspect leucaena-segmentation:cuda
```

Se der erro, construa a imagem:

```powershell
docker compose build
```

---

## 2. Onde ficam os logs

Sempre que usar `run_pipeline.py`, o log completo fica em:

```text
C:\00_DATASETS_AI\260515-piracicaba-aoi\models\logs\pipeline_latest.log
```

Para acompanhar o log em tempo real em outro PowerShell:

```powershell
Get-Content "$DEST\models\logs\pipeline_latest.log" -Wait
```

Quando der erro, procure por:

```text
[ERROR]
Traceback
FAILED
Process exited with code
```

---

## 3. Modo recomendado: rodar uma etapa por vez com run_pipeline.py

Este modo ainda usa o orquestrador, mas **somente uma etapa por comando**.
E o melhor jeito para testar sem misturar varios problemas.

### Etapa 1 - copiar RGB, IR e LAZ da AOI

Primeiro faca dry-run:

```powershell
python run_pipeline.py `
  --aoi $AOI `
  --layer $LAYER `
  --source $SOURCE `
  --dest $DEST `
  --steps 1 `
  --dry-run --verbose
```

Se passar, rode de verdade:

```powershell
python run_pipeline.py `
  --aoi $AOI `
  --layer $LAYER `
  --source $SOURCE `
  --dest $DEST `
  --steps 1 `
  --verbose
```

Verifique se copiou:

```powershell
(Get-ChildItem "$DEST\opt\raw\rgb" -Filter "*.tif" -File).Count
(Get-ChildItem "$DEST\opt\raw\ir" -Filter "*.tif" -File).Count
(Get-ChildItem "$DEST\lidar\raw" -Filter "*.laz" -File).Count
```

Resultado esperado: todos maiores que `0`.

Se RGB ou IR der `0`, nao continue para etapa 2.

---

### Etapa 2 - gerar RGBNIR 4 bandas

Rode:

```powershell
python run_pipeline.py `
  --aoi $AOI `
  --layer $LAYER `
  --source $SOURCE `
  --dest $DEST `
  --steps 2 `
  --workers 4 `
  --verbose
```

Verifique:

```powershell
(Get-ChildItem "$DEST\opt\rgbnir" -Filter "*.tif" -File).Count
```

Resultado esperado: maior que `0`.

Arquivos gerados ficam aqui:

```text
C:\00_DATASETS_AI\260515-piracicaba-aoi\opt\rgbnir
```

Cada `.tif` deve ter 4 bandas:

```text
1 = Red
2 = Green
3 = Blue
4 = NIR
```

---

### Etapa 2b - gerar overviews para QGIS (opcional)

Use se o QGIS estiver lento abrindo os `.tif`.

```powershell
python run_pipeline.py `
  --aoi $AOI `
  --layer $LAYER `
  --source $SOURCE `
  --dest $DEST `
  --steps 2 `
  --build-ovr `
  --verbose
```

Observacao: este comando tambem chama a etapa 2. Se os RGBNIR ja existem, a
etapa 2 deve pular os arquivos existentes e depois gerar os `.ovr`.

Verifique:

```powershell
(Get-ChildItem "$DEST\opt\rgbnir" -Filter "*.ovr" -File).Count
```

---

### Etapa 3 - transformar LAZ em CHM

Esta etapa usa Docker/PDAL.

Rode:

```powershell
python run_pipeline.py `
  --aoi $AOI `
  --layer $LAYER `
  --source $SOURCE `
  --dest $DEST `
  --steps 3 `
  --verbose
```

Verifique:

```powershell
(Get-ChildItem "$DEST\lidar\chm" -Filter "*.tif" -File).Count
```

Resultado esperado: maior que `0`.

Se der erro aqui, quase sempre e uma destas coisas:

- Docker Desktop nao esta rodando.
- Imagem `leucaena-segmentation:cuda` nao existe.
- O LAZ nao tem RGBNIR correspondente.
- PDAL falhou em algum tile especifico.

O log completo fica em:

```text
C:\00_DATASETS_AI\260515-piracicaba-aoi\models\logs\pipeline_latest.log
```

---

### Etapa 4 - gerar patches de treinamento

Esta etapa usa:

```text
opt\rgbnir\
lidar\chm\
annotations\polygons.geojson
```

Rode:

```powershell
python run_pipeline.py `
  --aoi $AOI `
  --layer $LAYER `
  --source $SOURCE `
  --dest $DEST `
  --steps 4 `
  --annotations "$DEST\annotations\polygons.geojson" `
  --verbose
```

Verifique:

```powershell
Test-Path "$DEST\patches\manifest.csv"
(Get-ChildItem "$DEST\patches\opt" -Filter "*.npy" -File).Count
(Get-ChildItem "$DEST\patches\lbl" -Filter "*.npy" -File).Count
```

Resultado esperado:

- `manifest.csv` existe.
- `patches\opt` tem `.npy`.
- `patches\lbl` tem `.npy`.

---

### Etapa 5 - treinamento

So rode depois que a etapa 4 funcionar.

```powershell
python run_pipeline.py `
  --aoi $AOI `
  --layer $LAYER `
  --source $SOURCE `
  --dest $DEST `
  --steps 5 `
  --train `
  --experiment 1 `
  --verbose
```

Saidas do treino ficam em:

```text
experiments\exp_1\
```

---

## 4. Comandos diretos dos scripts (para isolar ainda mais)

Use esta secao se voce quiser testar um script especifico sem o
`run_pipeline.py`.

### 4.1 Testar somente a copia de tiles

Dry-run:

```powershell
python prep-copy-tiles-from-aoi.py `
  --aoi $AOI `
  --layer $LAYER `
  --source-laz "D:\laz" `
  --source-rgb "D:\rgb" `
  --source-ir "D:\ir" `
  --dest-laz "$DEST\lidar\raw" `
  --dest-rgb "$DEST\opt\raw\rgb" `
  --dest-ir "$DEST\opt\raw\ir" `
  --dry-run
```

Rodar real:

```powershell
python prep-copy-tiles-from-aoi.py `
  --aoi $AOI `
  --layer $LAYER `
  --source-laz "D:\laz" `
  --source-rgb "D:\rgb" `
  --source-ir "D:\ir" `
  --dest-laz "$DEST\lidar\raw" `
  --dest-rgb "$DEST\opt\raw\rgb" `
  --dest-ir "$DEST\opt\raw\ir"
```

### 4.2 Testar somente RGB + IR -> RGBNIR

```powershell
python prep-rgbnir-from-rgb-ir.py `
  --aoi $AOI `
  --layer $LAYER `
  --source-rgb "$DEST\opt\raw\rgb" `
  --source-ir "$DEST\opt\raw\ir" `
  --out-dir "$DEST\opt\rgbnir" `
  --workers 4
```

### 4.3 Inspecionar LAZ antes de gerar CHM

Esta etapa precisa do Docker. O jeito mais facil e usar o orquestrador so
para etapa 3:

```powershell
python run_pipeline.py `
  --aoi $AOI `
  --layer $LAYER `
  --source $SOURCE `
  --dest $DEST `
  --steps 3 `
  --verbose
```

Se quiser processar poucos LAZ primeiro, rode direto dentro do Docker exige
montagem de volumes manual. Para evitar erro de caminho, prefira o comando
acima.

---

## 5. Como retomar sem refazer tudo

Se uma etapa ja funcionou, voce nao precisa rodar de novo.

Exemplos:

```powershell
# Ja copiou os arquivos, agora so gerar RGBNIR
python run_pipeline.py --aoi $AOI --layer $LAYER --source $SOURCE --dest $DEST --steps 2 --verbose

# Ja tem RGBNIR, agora so gerar CHM
python run_pipeline.py --aoi $AOI --layer $LAYER --source $SOURCE --dest $DEST --steps 3 --verbose

# Ja tem CHM, agora so patches
python run_pipeline.py --aoi $AOI --layer $LAYER --source $SOURCE --dest $DEST --steps 4 --annotations "$DEST\annotations\polygons.geojson" --verbose
```

---

## 6. Quando usar --overwrite

Por padrao, o pipeline tenta **pular arquivos existentes**.

Use `--overwrite` se voce quer recriar os arquivos de uma etapa:

```powershell
python run_pipeline.py `
  --aoi $AOI `
  --layer $LAYER `
  --source $SOURCE `
  --dest $DEST `
  --steps 2 `
  --workers 4 `
  --overwrite `
  --verbose
```

Cuidado: `--overwrite` pode fazer uma etapa demorada rodar de novo.

---

## 7. Limpar uma etapa manualmente

Se uma etapa ficou pela metade, voce pode renomear a pasta antes de rodar de
novo. E mais seguro do que deletar direto.

Exemplo para RGBNIR:

```powershell
Rename-Item "$DEST\opt\rgbnir" "$DEST\opt\rgbnir_bkp"
New-Item -ItemType Directory -Force "$DEST\opt\rgbnir"
```

Exemplo para patches:

```powershell
Rename-Item "$DEST\patches" "$DEST\patches_bkp"
New-Item -ItemType Directory -Force "$DEST\patches"
```

---

## 8. Checklist de sucesso

Use este checklist para saber se pode passar para a proxima etapa.

```powershell
# Etapa 1
(Get-ChildItem "$DEST\opt\raw\rgb" -Filter "*.tif" -File).Count
(Get-ChildItem "$DEST\opt\raw\ir" -Filter "*.tif" -File).Count
(Get-ChildItem "$DEST\lidar\raw" -Filter "*.laz" -File).Count

# Etapa 2
(Get-ChildItem "$DEST\opt\rgbnir" -Filter "*.tif" -File).Count

# Etapa 3
(Get-ChildItem "$DEST\lidar\chm" -Filter "*.tif" -File).Count

# Etapa 4
Test-Path "$DEST\patches\manifest.csv"
(Get-ChildItem "$DEST\patches\opt" -Filter "*.npy" -File).Count
(Get-ChildItem "$DEST\patches\lbl" -Filter "*.npy" -File).Count
```

Se qualquer contagem esperada der `0`, pare e olhe o log antes de seguir.

---

## 9. Erros comuns

### Erro: UnicodeEncodeError / charmap codec

Normalmente acontece quando o PowerShell tenta imprimir caracteres especiais.
O logger ja foi ajustado para UTF-8, mas se aparecer de novo rode:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

Depois rode o comando novamente.

### Erro: Docker image not found

```powershell
docker compose build
```

### Erro: Docker Desktop not running

Abra o Docker Desktop e espere iniciar.

### Erro: Input RGB / IR = 0

A etapa 1 nao copiou RGB/IR. Verifique:

```powershell
(Get-ChildItem "$DEST\opt\raw\rgb" -Filter "*.tif" -File).Count
(Get-ChildItem "$DEST\opt\raw\ir" -Filter "*.tif" -File).Count
```

Se ainda for `0`, rode a etapa 1 com `--verbose` e veja se aparecem `MISS`.

### Erro: annotations file not found

Confirme:

```powershell
Test-Path "$DEST\annotations\polygons.geojson"
```

Se o nome for outro, passe o caminho manualmente:

```powershell
--annotations "CAMINHO_DO_SEU_GEOJSON"
```

---

## 10. Ordem segura para testar hoje

Rode exatamente nesta ordem:

```powershell
# 1) Copia
python run_pipeline.py --aoi $AOI --layer $LAYER --source $SOURCE --dest $DEST --steps 1 --verbose

# 2) RGBNIR
python run_pipeline.py --aoi $AOI --layer $LAYER --source $SOURCE --dest $DEST --steps 2 --workers 4 --verbose

# 3) CHM
python run_pipeline.py --aoi $AOI --layer $LAYER --source $SOURCE --dest $DEST --steps 3 --verbose

# 4) Patches
python run_pipeline.py --aoi $AOI --layer $LAYER --source $SOURCE --dest $DEST --steps 4 --annotations "$DEST\annotations\polygons.geojson" --verbose

# 5) Treino
python run_pipeline.py --aoi $AOI --layer $LAYER --source $SOURCE --dest $DEST --steps 5 --train --experiment 1 --verbose
```

Se uma etapa falhar, nao continue. Abra:

```text
C:\00_DATASETS_AI\260515-piracicaba-aoi\models\logs\pipeline_latest.log
```

e procure por `[ERROR]`, `Traceback` ou `FAILED`.
