# Apostila — Segmentação semântica de leucaena (estudo pessoal)

> Esta é uma apostila informal feita para acompanhar os planos `01`, `03` e
> `04` do seu repositório. Foco em **conceitos**, não em código. Os
> trechos de código são só ilustrativos. Sinta-se à vontade para anotar,
> riscar e reorganizar.
>
> Pasta `studies/` está no `.gitignore` — este arquivo não vai para o GitHub.

## Sumário

1. [O problema: segmentação semântica binária](#1-o-problema-segmenta%C3%A7%C3%A3o-sem%C3%A2ntica-bin%C3%A1ria)
2. [Dados geoespaciais 101](#2-dados-geoespaciais-101)
3. [GeoJSON, polígonos e rasterização](#3-geojson-pol%C3%ADgonos-e-rasteriza%C3%A7%C3%A3o)
4. [Tiles, mosaicos e VRT](#4-tiles-mosaicos-e-vrt)
5. [Patches e janela deslizante](#5-patches-e-janela-deslizante)
6. [Normalização, dtypes e outliers](#6-normaliza%C3%A7%C3%A3o-dtypes-e-outliers)
7. [Splits: train / val / test](#7-splits-train--val--test)
8. [PyTorch: Dataset, DataLoader e tensores](#8-pytorch-dataset-dataloader-e-tensores)
9. [Augmentations](#9-augmentations)
10. [Arquitetura ResUNet](#10-arquitetura-resunet)
11. [Fusão optical + LiDAR (experimentos 1, 2, 3)](#11-fus%C3%A3o-optical--lidar-experimentos-1-2-3)
12. [Loss function, classes e pesos](#12-loss-function-classes-e-pesos)
13. [Treinamento: optimizer, LR scheduler, early stopping](#13-treinamento-optimizer-lr-scheduler-early-stopping)
14. [Métricas: F1, precision, recall, IGNORE_INDEX](#14-m%C3%A9tricas-f1-precision-recall-ignore_index)
15. [Inferência: sliding window, overlap, softmax averaging](#15-infer%C3%AAncia-sliding-window-overlap-softmax-averaging)
16. [Armazenamento: .npy vs HDF5 vs Zarr](#16-armazenamento-npy-vs-hdf5-vs-zarr)
17. [Reprojection e SetSpatialFilter](#17-reprojection-e-setspatialfilter)
18. [Glossário rápido](#18-gloss%C3%A1rio-r%C3%A1pido)
19. [Leituras sugeridas](#19-leituras-sugeridas)

---

## 1. O problema: segmentação semântica binária

Há três tarefas clássicas de visão computacional. Saber a diferença ajuda
muito porque artigos usam termos parecidos:

| Tarefa                | Pergunta que responde                                     | Saída |
|-----------------------|-----------------------------------------------------------|-------|
| **Classificação**     | “Que objeto há nesta imagem?”                              | 1 label por imagem |
| **Detecção**          | “Onde está cada objeto? Que tipo é cada um?”               | Caixas (bounding boxes) + labels |
| **Segmentação semântica** | “Para cada pixel, a que classe ele pertence?”          | Máscara do mesmo tamanho da imagem |

A sua tarefa é a **segmentação semântica binária**: cada pixel é `0`
(fundo) ou `1` (leucaena). Não importa se há um arbusto ou cem; importa
**quais pixels** são leucaena.

```
Imagem 256x256
                          Modelo                       Máscara 256x256
[ R G B NIR pixels ]  --->  ResUNet  --->  [ 0 0 0 0 1 1 1 0 ... ]
```

**Por que “semântica”?** Porque o modelo não distingue dois indivíduos da
mesma espécie. Distinguir “esta árvore é a #5 e aquela é a #6” seria
**segmentação de instâncias** (Mask R-CNN, por exemplo). Para mapear a
**cobertura** de leucaena, semântica basta.

**Binária vs multi-classe.** O `tree_fusion` original tinha ~8 classes de
árvores. Você reduziu para 2. Isso simplifica:

- Saída do modelo: `(N, 2, H, W)` em vez de `(N, 10, H, W)`.
- Loss: pesos `[0.3, 0.7]` (só dois números).
- Métricas: F1 da classe 1 (leucaena) já resume bem o resultado.

---

## 2. Dados geoespaciais 101

Tudo aqui gira em torno de **raster** (imagem em grade de pixels) e
**vetor** (linhas e polígonos com coordenadas).

### Raster (GeoTIFF)

Um **GeoTIFF** é uma imagem com metadados georreferenciados. Os 3 pedaços
que importam:

| Metadado | Para que serve | Exemplo |
|----------|----------------|---------|
| **CRS** (Coordinate Reference System) | Em que “mapa” esses pixels estão | EPSG:31983 (SIRGAS 2000 / UTM 23S, em metros) |
| **GeoTransform** | Cantos e tamanho de pixel | `(x_min, dx, 0, y_max, 0, -dy)` |
| **Shape** (H, W) | Quantos pixels | 5000 × 5000 |

Combinando os três, dá para responder “este pixel (1500, 800) está em
qual longitude/latitude?”.

### Bandas

GeoTIFF pode ter várias **bandas** (camadas) por pixel:

- **RGB**: 3 bandas (vermelho, verde, azul) — foto que humano enxerga.
- **NIR**: infravermelho próximo — vegetação reflete muito; aparece bem
  brilhante. Junto com RGB vira **RGBN** (4 bandas).
- **LiDAR**: laser do avião mede a altura/intensidade de retorno. Vira
  imagens como **CHM** (Canopy Height Model, altura da copa) e
  **intensidade** (quanto da luz voltou).

Ordem das bandas é convenção:

| Convenção  | Banda 1 | Banda 2 | Banda 3 | Banda 4 |
|-----------|---------|---------|---------|---------|
| RGBN      | Red     | Green   | Blue    | NIR     |
| BGRN      | Blue    | Green   | Red     | NIR     |

Seu repositório usa **BGRN** internamente (`conf/general.py`), então o
script de patches reordena RGBN → BGRN antes de salvar. Não é “certo ou
errado”, é só convenção; o modelo aprende com qualquer ordem desde que
**seja sempre a mesma** no treino e na inferência.

### Vetor (GeoJSON, Shapefile)

Polígonos, linhas e pontos com coordenadas. No `leucaena.earth` cada
desenho fica como **polígono** com CRS (em geral WGS84, EPSG:4326).

Coexistem com rasters: para a IA usar, o polígono precisa virar uma
**máscara raster** alinhada à imagem (próxima seção).

---

## 3. GeoJSON, polígonos e rasterização

### O que é rasterização

Transformar polígonos em pixels:

```
Polígono "leucaena"  →  rasterizar  →  matriz HxW
       \                                / \   / \
        \                              0 0 1 1 1 0 0
         \                             0 1 1 1 1 1 0
          \                            0 1 1 1 1 0 0
```

Tudo que estiver **dentro** do polígono recebe valor `1`; o resto fica
`0`. Em outras tarefas se queima outro número (ex.: id da espécie).

### Como sua função faz isso (`utils/ops.py`)

`rasterize_geojson(geojson, raster_de_referencia)`:

1. Abre o raster de referência para saber **grid** (CRS, transform, H, W).
2. Cria uma matriz em memória do mesmo tamanho, preenchida com `0`.
3. Abre o GeoJSON; se o CRS dele for diferente, **reprojeta** cada
   polígono para o CRS do raster.
4. Chama `gdal.RasterizeLayer(...)` que “pinta” os polígonos com `1`.
5. Devolve a matriz.

Resultado: máscara `H × W` perfeitamente alinhada à imagem.

### Por que isso é melhor que arquivos `imgTrain_X.tif` prontos

O `tree_fusion` original esperava você **pré-rasterizar no QGIS**, salvar
TIFF, e versionar. Problemas:

- Mudou o polígono → tem que refazer o TIFF.
- Pesado para versionar.
- Não escala para muita gente desenhando (no `leucaena.earth`).

Com GeoJSON tudo isso some: a rasterização é **on-the-fly** dentro do
`prep-data.py` ou `prep-patches-from-tiles.py`. O GeoJSON é leve
(KB/MB), versionável, editável.

### Detalhe importante: `IGNORE_INDEX`

Imagine um voo cobrindo 100 km² mas você só desenhou polígonos em 30 km².
Os outros 70 km² **não são “sem leucaena” com certeza** — pode ter
leucaena que ninguém anotou.

Se você marcar tudo isso como `0` (background) e treinar, o modelo aprende
errado: “naquela área grande nunca tem leucaena”. Falso negativo certo.

Solução:

- Pixels dentro de polígono → `1`.
- Pixels fora, **mas dentro de área anotada** → `0`.
- Pixels fora de área anotada → **`255` (IGNORE_INDEX)**.

No treino, a loss e as métricas **ignoram** `255`. É como dizer “não sei,
não conte”.

No `conf/general.py`:

```python
IGNORE_INDEX = 255
DISCARDED_CLASS = IGNORE_INDEX
```

No `prep-data.py`, na hora de montar o `test_label`, ele começa **todo
255** e só coloca `0/1` nos pixels que estão dentro das janelas de teste.
Treino e teste não se misturam.

---

## 4. Tiles, mosaicos e VRT

### Tile

Um **tile** é uma imagem que cobre uma carta/quadra (ex.: `SF-23-Y-A-IV-...`).
A IGC entrega o voo dividido assim por convenção cartográfica. Cada tile
tem sua própria extensão e arquivo `.tif`.

### Mosaico

Juntar todos os tiles num único TIFF gigante. Vantagem: ferramentas tratam
como uma cena. Desvantagem: arquivo **enorme** (terabytes para Brasil
todo); duplica o disco; demora para gerar; e nenhuma ferramenta consegue
abrir tudo de uma vez.

### VRT (Virtual Raster)

Um arquivo **XML** do GDAL que **finge** ser um único raster, mas por trás
aponta para os tiles originais. Não copia nada:

```xml
<VRTDataset rasterXSize="500000" rasterYSize="500000">
  <SimpleSource>
    <SourceFilename>D:\leucaena\rgbir\SF-23-Y-A-IV-4-NE-F.tif</SourceFilename>
    ...
  </SimpleSource>
  <SimpleSource>
    <SourceFilename>D:\leucaena\rgbir\SF-23-Y-A-IV-4-NE-E.tif</SourceFilename>
    ...
  </SimpleSource>
  ...
</VRTDataset>
```

Para ferramentas GDAL (incluindo seu `prep-data.py`), abre como se fosse
um TIFF único. Gerar com:

```bash
gdalbuildvrt mosaico.vrt /data/rgbir/*.tif
```

Ótimo para **estudo pequeno** (10–100 tiles, RAM aguenta).

### Quando VRT não basta

Para Brasil inteiro, mesmo o VRT vira um “raster” de centenas de gigapixels.
Não cabe na RAM em nenhum momento — então em vez de fingir que é uma cena
só, você passa a tratar **cada tile como uma unidade independente**.

É exatamente isso que o plano 03 faz: o script `prep-patches-from-tiles.py`
**itera tile a tile**, gera patches por tile e nunca carrega tudo.

---

## 5. Patches e janela deslizante

### Por que patches

Redes neurais convolucionais (CNN) trabalham com **tensores de tamanho
fixo**. Você não vai jogar uma imagem 50000×50000 numa GPU de 12 GB.
Solução: cortar em **patches** menores (256×256), treinar nesses
pedaços, e na hora de predizer cortar de novo e remontar.

### Patch size

O `PATCH_SIZE = 256` é um compromisso:

- **Menor (64, 128)** → mais patches por imagem, treino mais rápido por
  iteração; mas cada patch enxerga menos contexto. Pode confundir
  copa de leucaena com outras espécies sem ver vizinhança.
- **Maior (512, 1024)** → mais contexto; mas menos patches, GPU pode
  estourar a memória, treino mais lento.

`256` é típico para sensoriamento remoto óptico de alta resolução.

### Overlap

`PATCH_OVERLAP = 0.5` (50%). Significa que o passo entre patches é
metade do tamanho:

```
patch 1: cols   0..255
patch 2: cols 128..383   (50% sobrepostos)
patch 3: cols 256..511
```

Por que overlap?

- **No treino**: gera mais exemplos a partir da mesma imagem (data
  augmentation barato), e os “alvos” caem em posições diferentes do
  patch (não fica todo recurso na mesma borda).
- **Na inferência**: pixel da borda fica longe do centro do patch, onde
  a CNN tem menos contexto. Com overlap, o mesmo pixel é predito várias
  vezes em posições diferentes do patch, e a gente **faz média** das
  predições (mais robusto).

### `view_as_windows` (`skimage`)

Função que cria, de graça, uma “visão” da imagem em janelas:

```python
windows = view_as_windows(image, (256, 256), step=128)
# shape: (n_rows, n_cols, 256, 256)
```

Não copia os dados. Cada janela é “uma fatia” da matriz original. Você
itera, calcula a fração de classe positiva, descarta as ruins.

### Filtragem por fração de classe

```python
fraction = (window == 1).mean()  # quantos pixels são leucaena?
if fraction >= 0.01:             # MIN_TRAIN_CLASS
    keep(window)
```

Sem isso, você teria 99% de patches só com background (porque leucaena é
rara). O modelo aprenderia a sempre dizer `0` e teria F1 alto na média
**mas zero recall** em leucaena. Filtrar mantém o problema de
desbalanceamento dentro do razoável.

### Reflect padding

Na **inferência** (não no treino), os patches precisam cobrir as bordas.
A última coluna pode não bater com múltiplo de 256. Solução: espelhar as
bordas para fora:

```
Imagem com borda à direita "abc"   →   após reflect padding "abccba..."
```

Assim a CNN nunca recebe pixel “vazio”; o que parece extensão da imagem
é só o reflexo dela.

---

## 6. Normalização, dtypes e outliers

### Dtypes (tipos de dados numéricos)

Quanto cada pixel ocupa:

| Dtype     | Bytes | Range            | Quando usar |
|-----------|-------|------------------|-------------|
| `uint8`   | 1     | 0..255           | Imagem normal (foto), máscara binária |
| `uint16`  | 2     | 0..65535         | Imagens de satélite com 12–16 bits (Sentinel-2) |
| `float32` | 4     | ±3.4e38          | Após normalização; durante o cálculo da rede |
| `float16` | 2     | ±65504           | Mixed-precision training |

A regra prática:

- **Disco**: salvar como o menor possível (uint8). Patches uint8 são 4×
  menores que float32.
- **GPU**: a rede calcula em float32 (ou float16). Conversão é barata.

No seu pipeline tile-based:

- Tile original: uint8 (DJI Phantom, IGC, etc.).
- Patch salvo: uint8.
- Dataloader divide por 255 → float32 em [0, 1] na hora de treinar.

### Normalização

Redes treinam melhor quando entradas estão em uma escala pequena e
consistente. Várias maneiras:

| Método | O que faz | Quando |
|--------|-----------|--------|
| `/255` | Mapeia uint8 [0..255] para [0, 1] | Quando todos os pixels já estão na mesma escala (uint8) |
| Min-max por banda | `(x - bmin) / (bmax - bmin)` por banda da cena | Sensor com calibração variável; cobre o range real |
| Standardização (z-score) | `(x - mean) / std` por banda | Comum em RGB “natural” (ImageNet) |
| Por dataset | Mean/std calculados em todo o dataset | Mais estável em coleções grandes |

O `prep-data.py` (cena única) faz **min-max por banda da cena**.

O `prep-patches-from-tiles.py` (tile-based) **não normaliza no disco**
(fica uint8), e divide por 255 no dataloader. Por quê?

- Para Brasil todo, **min-max global** exigiria varrer todos os tiles
  duas vezes (uma para descobrir min/max, outra para aplicar). Caro.
- Como todos os tiles vêm do mesmo sensor com mesma calibração relativa,
  `/255` já dá uma escala razoável.
- Se isso virar problema (treino não converge bem em tiles novos), a
  solução é normalização por estatística fixa (mean/std do dataset
  inteiro, calculado uma vez e gravado).

### Outlier clipping (`filter_outliers`)

Sensores às vezes têm pixels malucos (saturados a 65535, NaN, etc.).
O `filter_outliers` calcula histograma cumulativo por banda e corta os
extremos:

```python
# usa o 0.1% mais baixo como "min" e o 99.9% mais alto como "max"
bth, uth = 0.001, 0.999
```

Tudo abaixo do quantil 0.001 vira esse valor; tudo acima do 0.999 vira
o teto. Sem isso, um pixel saturado puxaria o min-max e achataria a
imagem inteira para perto de zero ou um.

Na versão atual, é usado só no `prep-data.py` (cena única). Para o
pipeline tile-based, valeria adicionar como pós-processo, mas só
quando virar necessário.

---

## 7. Splits: train / val / test

### Por que três conjuntos

- **Train**: dados que o modelo enxerga durante o treino e ajusta pesos.
- **Val**: dados que o modelo **não treina**, usados para decidir
  hiperparâmetros, early stopping, etc.
- **Test**: dados intocados até o fim, usados **uma vez** para reportar
  o resultado final no paper/tese.

Misturar val com test enviesa o resultado (você acaba “escolhendo” o
modelo que parece bom no test).

### Patch-level vs tile-level

Detalhe **importantíssimo** em sensoriamento remoto:

**Patch-level split (o que seu pipeline faz hoje):**

```
17 tiles → 5000 patches → embaralha → 60% train / 20% val / 20% test
```

Risco: patches do **mesmo tile** podem cair em train e test. Eles têm
iluminação, sensor, condições idênticas. O modelo aprende a textura
daquela imagem específica e parece muito bom no test, sem realmente
generalizar.

**Tile-level split:**

```
17 tiles → embaralha tiles → 10 train / 3 val / 4 test → patches do tile herdam o split
```

Modelo precisa generalizar para um tile **que nunca viu** — mais honesto.

Por que ainda não fizemos tile-level: você tem **17 tiles**. Tirar 4
para teste cobre uma área grande e pode não ter polígonos suficientes
para uma estatística confiável. Quando vocês passar para centenas de
tiles, tile-level vira o caminho certo. Está registrado no plano 04.

### Seed

`np.random.default_rng(42)` ou `np.random.seed(42)`. Garante que “quando
você rodar de novo, vai cair tudo no mesmo lugar”. **Reprodutibilidade
científica** = mesma seed + mesmo código → mesmo resultado.

---

## 8. PyTorch: Dataset, DataLoader e tensores

### Dataset

Classe que diz “tenho X exemplos; me peça o exemplo `i` que eu te devolvo
o tensor”. Implementa dois métodos:

```python
class MyDataset(Dataset):
    def __len__(self):
        return n_exemplos
    def __getitem__(self, i):
        return tensor_x, tensor_y
```

No seu repo:

- `TreeTrainDataSet` (caminho antigo): carrega `opt_img.npy` inteiro em
  RAM, indexa patch por índice linear.
- `PatchFileDataset` (novo): lê `.npy` do disco quando o índice é pedido.
  Vantagem: não precisa caber tudo em RAM.

### DataLoader

Empilha vários exemplos em batch e cuida do paralelismo:

```python
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
```

| Argumento     | O que faz |
|---------------|-----------|
| `batch_size`  | Quantos exemplos por iteração da rede |
| `shuffle`     | Embaralha índices a cada epoch |
| `num_workers` | Processos em paralelo lendo `.npy` (acelera I/O) |
| `pin_memory`  | Transferência mais rápida CPU → GPU |

### Tensores: shape e ordem dos eixos

Numpy típico de imagem: `(H, W, C)` — Height × Width × Channels.

PyTorch quer: `(C, H, W)` — canal primeiro.

`torchvision.transforms.ToTensor()`:
- Recebe HWC,
- devolve CHW,
- divide por 255 se vier uint8.

Batch: a primeira dimensão é o batch. Então o que chega no modelo é
`(B, C, H, W)` — 4D.

---

## 9. Augmentations

Truques para **simular mais dados** rotacionando e espelhando patches.
A rede aprende invariâncias (uma copa girada continua sendo uma copa).

No seu repo (`TreeTrainDataSet` e `PatchFileDataset`):

```python
k = random.randint(0, 3)
opt_tensor = torch.rot90(opt_tensor, k, (1, 2))   # 0, 90, 180, 270 graus

if random.getrandbits(1):
    opt_tensor = hflip(opt_tensor)                # espelhamento horizontal

if random.getrandbits(1):
    opt_tensor = vflip(opt_tensor)                # espelhamento vertical
```

Detalhe crítico: aplique a **mesma** transformação na imagem **E** na
máscara. Se você gira só a imagem, a anotação fica desalinhada e o
modelo aprende lixo.

Outras augmentations comuns (não usadas aqui ainda):

- Color jitter (brilho/contraste) — útil para fotos aéreas em horários
  diferentes.
- Gaussian noise — robustez.
- CutMix / MixUp — embaralhar pedaços; pouco usado em remote sensing.

---

## 10. Arquitetura ResUNet

### U-Net (a base)

Imagine uma rede em formato de “U”:

```
encoder ↓                 decoder ↑
     |                          |
     |                          |
     +---  skip connections  ---+
     |                          |
     |                          |
gargalo (representação mais compacta)
```

- **Encoder** (descida): convoluções + max-pool. Cada nível tem mais
  canais e menos resolução espacial.
- **Decoder** (subida): convolução + upsample. Reconstrói a resolução
  espacial.
- **Skip connections**: ligações horizontais que copiam a saída de cada
  nível do encoder para o nível correspondente do decoder. Sem isso o
  decoder não recupera detalhes finos.

### Resíduo (ResNet)

Em vez de cada bloco aprender `y = f(x)`, ele aprende `y = f(x) + x`.
A camada **adiciona** o resíduo ao input. Por quê:

- Em redes muito profundas, gradientes somem (vanishing gradient).
- O atalho `+ x` deixa gradiente fluir mesmo se `f(x)` for pequeno.
- Treinos ficam mais estáveis.

### ResUNet

U-Net com blocos residuais no encoder. É o que `models/resunet.py` faz.

`get_model()` retorna `(model, lidar_bands)`:

- `lidar_bands = None` → modelo 1 (só óptico).
- `lidar_bands = [...]` → seleciona quais bandas do tensor LiDAR usar.

Profundidades do encoder no seu repo: `[32, 64, 128, 256]`. Cada número
é o número de filtros em um nível. Quanto maior, mais capacidade
(parâmetros) — mas também mais memória GPU.

---

## 11. Fusão optical + LiDAR (experimentos 1, 2, 3)

Quando há duas fontes (foto colorida + altura LiDAR), você pode combinar
de três jeitos:

### Experimento 1 — Sem fusão (só óptico)

LiDAR é ignorado. Modelo recebe 4 canais (BGRN).

Quando usar: você não tem LiDAR alinhado, ou quer um baseline simples.

### Experimento 2 — Early fusion (fusão na entrada)

Concatena as bandas óptico + LiDAR antes do primeiro `Conv`:

```
input: (B, 4+L, H, W)   ← 4 bandas óticas + L bandas LiDAR juntas
       |
   encoder único processa tudo
```

Simples. A rede aprende sozinha como combinar. Funciona bem quando as
duas modalidades têm informação **complementar** mas correlacionada.

### Experimento 3 — Late fusion (fusão no final)

Dois encoders separados:

```
óptico ──> encoder_o ──┐
                       ├──> concat ──> decoder ──> mask
LiDAR  ──> encoder_l ──┘
```

Cada encoder aprende representações **independentes** da sua modalidade
e só no final elas se encontram. Mais parâmetros, mais complexo, às
vezes generaliza melhor quando as modalidades são muito diferentes
(óptico = textura; LiDAR = geometria 3D).

Resumo:

| Modalidade   | Variação esperada de F1 (paper Ferrari) |
|--------------|------------------------------------------|
| Só óptico    | Bom (referência) |
| Early fusion | Costuma melhorar em copas pequenas |
| Late fusion  | Costuma melhorar em separar espécies parecidas |

Para o seu PhD, o **experimento 1 é o caminho rápido** (você ainda não
tem LiDAR alinhado a todos os tiles).

---

## 12. Loss function, classes e pesos

### Cross-entropy

A loss padrão para classificação:

```
L = - Σ y_true * log(y_pred)
```

Para um pixel cujo rótulo é `1` e a probabilidade prevista para classe 1
é `p`, a contribuição é `-log(p)`. Modelo é punido quando `p` é baixo
(o que ele “acredita” errado).

`nn.CrossEntropyLoss` no PyTorch:

- Recebe **logits** (saída sem softmax) e o **índice da classe** (não
  vetor one-hot).
- Calcula softmax + log + negative log likelihood internamente.

### `ignore_index`

```python
nn.CrossEntropyLoss(ignore_index=255, weight=torch.tensor([0.3, 0.7]))
```

Diz: “sempre que o rótulo for `255`, ignore esse pixel; não conte na
loss”. É o `IGNORE_INDEX` da seção 3.

### Class weights

O dataset é **muito desbalanceado**: ~99% background, ~1% leucaena.
Sem pesos, o gradiente vem 99% do tempo do background, e o modelo
aprende a sempre dizer `0`.

Pesos: `[0.3, 0.7]`

- Classe 0 (fundo) → peso 0.3
- Classe 1 (leucaena) → peso 0.7

Cada pixel de leucaena conta **mais que o dobro** que um de fundo. O
sinal de leucaena não some.

Detalhe: o paper original do Ferrari usava `[B_W=0.01, T_W=0.13, ..., 0]`
porque eram 10 classes. Para binário, basta um par.

### Outras losses (não usadas aqui mas comuns)

- **Dice loss**: 1 - dice coefficient. Boa para classes minoritárias.
- **Focal loss**: penaliza mais erros “fáceis” (o modelo errar uma
  amostra que “devia ser fácil” pesa mais).
- **Combo (CE + Dice)**: estável e popular em segmentação médica e
  remota.

---

## 13. Treinamento: optimizer, LR scheduler, early stopping

### Optimizer

Quem decide “quanto e para onde” atualizar os pesos a cada batch.
**Adam** é o padrão moderno:

```python
torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
```

Mais robusto que SGD puro, raramente precisa de tuning. `lr=1e-4` é
um ponto de partida saudável.

### Learning rate

A “taxa de aprendizado”. Multiplicador do gradiente.

- LR alto demais (1e-2): treino explode, perda vai pra NaN.
- LR baixo demais (1e-6): treino fica eternamente sem convergir.

Para fine-tuning (continuar a partir de um modelo pré-treinado), LR
baixo (~1e-5).

### LR scheduler

Reduzir LR ao longo do treino ajuda a “refinar” a convergência:

- **`ExponentialLR(gamma=0.995)`**: a cada epoch, LR vira `lr * 0.995`.
  Decaimento suave e contínuo.
- **`MultiStepLR(milestones=[5, 20], gamma=0.5)`**: corta LR pela
  metade nos epochs 5 e 20.
- **`CosineAnnealingLR`**: LR varia como cosseno, popular hoje.
- **`ReduceLROnPlateau`**: reduz LR só quando val_loss para de melhorar.

Seu repo usa `ExponentialLR`. Está no `train.py`.

### Early stopping

Para o treino **antes** do `MAX_EPOCHS` se a métrica de validação
parar de melhorar:

```python
EARLY_STOP_PATIENCE = 15        # epochs sem melhora
EARLY_STOP_MIN_DELTA = 5e-5     # melhora mínima para "contar"
EARLY_STOP_MIN_EPOCHS = 20      # antes disso nem tenta parar
```

Logica em `utils/trainer.EarlyStop`. Salva o melhor modelo no caminho
`models/model.pt`. Se passou 15 epochs sem melhorar, encerra.

Por que importa: evita **overfitting** (modelo decora o train, perde no
test).

---

## 14. Métricas: F1, precision, recall, IGNORE_INDEX

### Matriz de confusão (binária)

|             | Predito 0 | Predito 1 |
|-------------|-----------|-----------|
| Real **0**  | TN        | FP        |
| Real **1**  | FN        | TP        |

- **TP** (True Positive): leucaena real → predito leucaena. Acerto.
- **FP** (False Positive): fundo real → predito leucaena. Erro,
  superestimou.
- **FN** (False Negative): leucaena real → predito fundo. Erro,
  perdeu árvore.
- **TN** (True Negative): fundo real → predito fundo. Acerto.

### Métricas

```
Precision = TP / (TP + FP)   "dos que disse leucaena, quantos eram?"
Recall    = TP / (TP + FN)   "das leucaenas reais, quantas peguei?"
F1        = 2 * P * R / (P + R)   "média harmônica"
```

- **Accuracy** sozinho engana: 99% de fundo → modelo que sempre diz `0`
  tem 99% accuracy mas F1 = 0. Inútil.
- **F1** é o resumo padrão para classes desbalanceadas.

### IGNORE_INDEX nas métricas

Como na loss, as métricas usam:

```python
torchmetrics.F1Score(task='binary', ignore_index=255)
```

Pixels `255` não contam. Mesmo princípio: você só mede onde sabe a
resposta certa.

### IoU (Intersection over Union)

Outra métrica popular em segmentação:

```
IoU = |A ∩ B| / |A ∪ B|
```

Onde A é a máscara prevista e B a real. F1 e IoU andam juntos:
`IoU = TP / (TP + FP + FN)`.

---

## 15. Inferência: sliding window, overlap, softmax averaging

### Por que não dá para predizer a imagem inteira

GPU não aguenta uma matriz `(1, 4, 50000, 50000)`. Solução: cortar em
patches, predizer cada um, e remontar a máscara final.

### Sliding window de inferência

Igual ao do treino, mas **sem filtrar por classe** (você precisa cobrir
**todos** os pixels). Com overlap:

```
Patches:    0 1 2 3 4 5 ...
            ─┴─┴─┴─┴─┴─┴─
                ↑
            pixels nesta coluna são preditos em vários patches
```

### Softmax averaging

Para cada patch a rede dá **probabilidades**:

```
patch_softmax shape: (1, 2, 256, 256)
  - patch_softmax[0, 0, :, :] = P(fundo)
  - patch_softmax[0, 1, :, :] = P(leucaena)
```

Para pixels que aparecem em mais de um patch, **soma** as probabilidades
em uma matriz acumuladora e **conta** quantas vezes esse pixel foi
predito. No final divide acumulador / contador → probabilidade média.

Vantagens:

- Pixel da borda do patch (onde a CNN tem menos contexto) é compensado
  pela predição do mesmo pixel no centro de outro patch.
- Reduz o efeito “costura” entre patches.

No `prediction.py`, é o que `PREDICTION_OVERLAPS = [0, 0.25, 0.5]` faz:
três passadas com overlaps diferentes, todas somadas.

### Da probabilidade ao mapa final

```python
prob_leucaena = soma_softmax[:, 1] / contador
mapa_binario = (prob_leucaena > 0.5).astype(np.uint8)
```

Salva `pred_prob.tif` (float32) e `pred.tif` (uint8 binário).

### Tile boundary artifacts

Em pipelines tile-based de inferência (plano 04), cada tile é predito
independentemente. Pode aparecer “costura” no limite entre tiles porque
o pixel da borda só vê metade da copa.

Mitigações:

- Patches com overlap alto (0.5).
- Predizer cada tile com **uma margem extra** das bordas dos vizinhos
  (overlap **entre tiles**, não só entre patches).
- Median filter na máscara final.

---

## 16. Armazenamento: .npy vs HDF5 vs Zarr

### Cenário

Você gera centenas de milhares de patches. Cada um pequeno.

### Pasta com muitos `.npy`

O que o pipeline tile-based faz hoje. Vantagens:

- Simples. Dá para abrir qualquer um no Python.
- Cada arquivo é independente; falha em um não afeta os outros.

Desvantagens quando o volume cresce:

- Filesystem fica lento listando milhões de arquivos.
- Backup/transferência horrível (cada arquivo tem overhead de metadado).
- I/O random fica caro.

Regra prática: até ~100k arquivos pequenos, tudo bem.

### HDF5

Um único arquivo binário com vários **datasets** dentro (como uma pasta
virtual):

```
train.h5
  /opt    shape=(50000, 256, 256, 4) dtype=uint8 chunks=(64,256,256,4)
  /lbl    shape=(50000, 256, 256)    dtype=uint8 chunks=(64,256,256)
```

Vantagens:

- Um arquivo único = fácil de copiar/backup.
- **Chunking**: dados são fatiados em blocos do tamanho típico de
  acesso (ex.: batch de 64 patches). Leitura sequencial é rápida.
- **Compressão**: opcional (LZF rápido; gzip lento; Blosc rápido e bom).
- Suporte excelente em Python (`h5py`).

Desvantagens:

- Concorrência: escrever em paralelo é chato. Leitura em paralelo
  funciona, mas exige cuidado.
- “Caixa preta”: difícil inspecionar um patch específico sem ferramenta.

### Zarr

Filosoficamente parecido com HDF5, mas em **vários arquivinhos** numa
pasta — cada chunk é um arquivo. Bom para:

- Cloud storage (S3 / GCS): cada chunk vira um objeto.
- Escrita paralela (cada worker grava chunks diferentes).
- Estrutura inspecionável (`ls` na pasta funciona).

Vai ser a escolha provável para o plano 04, especialmente se a inferência
for em cluster ou nuvem no futuro.

### Comparação prática

| Critério                | `.npy` pasta | HDF5 | Zarr |
|-------------------------|--------------|------|------|
| Setup                   | Trivial      | Simples | Simples |
| Backup/transferência    | Ruim em massa| Excelente | Médio |
| Concorrência leitura    | Ótima        | Boa  | Excelente |
| Concorrência escrita    | Ótima        | Difícil | Boa |
| Cloud (S3, GCS)         | Não nativo   | Não nativo | Nativo |
| Inspeção manual         | Excelente    | Precisa h5py / HDFView | Boa |
| Comunidade ML           | Tudo         | h5py + PyTorch | Crescendo |

### Chunk size

Tanto HDF5 quanto Zarr precisam de **chunk** definido. Regra:

- Chunk = múltiplo do batch que você lê. Se vai treinar com batch 8,
  chunk de 8 ou 16 patches faz sentido.
- Não deixe chunk minúsculo (overhead) nem gigante (lê demais para
  pegar 1 patch).

---

## 17. Reprojection e SetSpatialFilter

### Reprojection (CRS transform)

GeoJSON do `leucaena.earth` em geral está em **EPSG:4326** (WGS84,
graus). Seus tiles IGC em geral estão em **EPSG:31983** (UTM 23S, metros).

Para rasterizar polígono sobre tile, ambos precisam estar no mesmo CRS.
GDAL/OGR faz isso com `osr.CoordinateTransformation`:

```python
src_srs.IsSame(ref_srs)        # mesmo CRS?
ct = osr.CoordinateTransformation(src_srs, ref_srs)
geom.Transform(ct)             # reprojeta in-place
```

A função `rasterize_geojson` cuida disso: se os CRSs forem iguais, pula
a reprojeção (rápido); se forem diferentes, reprojeta cada polígono
para o CRS do raster.

### SetSpatialFilter (truque do plano 03)

GeoJSON do Brasil todo pode ter **milhares** de polígonos. Rasterizar
tudo contra um tile pequeno é desperdício enorme: a maioria dos
polígonos está em estados distantes.

`OGR.SetSpatialFilter(geom)` diz à camada: “de agora em diante, só me
mostre as features que **intersectam essa geometria**”. O índice
espacial interno (R-tree) faz isso rápido.

No `rasterize_geojson_for_tile`:

1. Pega a **bbox do tile** (em coordenadas do raster).
2. Reprojeta a bbox para o **CRS do GeoJSON**.
3. `layer.SetSpatialFilter(bbox_no_crs_do_geojson)`.
4. `layer.GetFeatureCount()` → quantos polígonos sobraram.
5. Itera só esses, reprojeta-os para o CRS do raster e rasteriza.

Resultado: para o tile médio, só ~dezenas de polígonos são realmente
processados, em vez de milhares.

### Ordem das coordenadas (pega-pegas comum)

GDAL antes da v3 usava **(x, y)** ou **(lon, lat)** em todo lugar. Da
v3 em diante, alguns CRS (como EPSG:4326) “oficialmente” esperam
**(lat, lon)**. Para evitar surpresas:

```python
srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
```

Se um polígono aparecer **rotacionado 90 graus**, esse é o sintoma.
Hoje seu código está estável com a versão GDAL do conda-forge usada no
Docker. Se um dia rasterizar “esquisito”, lembre desse switch.

---

## 18. Glossário rápido

| Sigla / termo            | Significado |
|--------------------------|-------------|
| **AOI** (Area of Interest) | Região onde você anotou e/ou quer rodar o modelo |
| **CRS** | Coordinate Reference System (EPSG:4326 etc.) |
| **CHM** | Canopy Height Model (raster de altura da copa) |
| **CNN** | Convolutional Neural Network |
| **CE** | CrossEntropy (loss) |
| **CUDA** | API NVIDIA para usar GPU |
| **CUDA cores** | Núcleos de processamento da GPU |
| **EPSG** | European Petroleum Survey Group — códigos de CRS |
| **F1** | Métrica de classificação (média de precision e recall) |
| **GDAL / OGR** | Bibliotecas C++ para raster (GDAL) e vetor (OGR) |
| **GeoJSON** | Formato JSON para vetor geoespacial |
| **GeoTIFF** | TIFF com metadados de georreferenciamento |
| **HDF5** | Hierarchical Data Format v5 |
| **IGC** | Instituto Geográfico e Cartográfico (SP) — autor dos voos |
| **IoU** | Intersection over Union |
| **LiDAR** | Light Detection and Ranging (laser) |
| **NIR** | Near Infrared |
| **OGR** | Lado vetorial da GDAL |
| **One-hot** | Codificação de classe como vetor (ex.: `[0, 1]` para classe 1) |
| **Patch** | Recorte fixo da imagem (256×256 etc.) |
| **Rasterize** | Converter vetor (polígono) em raster (matriz de pixels) |
| **ResUNet** | U-Net com blocos residuais |
| **Sliding window** | Janela móvel que percorre a imagem |
| **Softmax** | Função que transforma logits em probabilidades [0, 1] que somam 1 |
| **TIFF** | Tagged Image File Format |
| **U-Net** | Arquitetura encoder-decoder com skip connections |
| **uint8 / float32** | Tipos numéricos (1 byte / 4 bytes por valor) |
| **VRT** | Virtual Raster (XML do GDAL) |
| **WSL** | Windows Subsystem for Linux |
| **Zarr** | Storage chunked para arrays, amigável a nuvem |

---

## 19. Leituras sugeridas

### Conceitos básicos

- Stanford CS231n (vídeos no YouTube) — visão computacional moderna.
- *Dive into Deep Learning* (livro gratuito online) — bom para Python
  + PyTorch.

### Segmentação semântica

- Long, Shelhamer, Darrell (2015), *Fully Convolutional Networks for
  Semantic Segmentation* — papel histórico.
- Ronneberger, Fischer, Brox (2015), *U-Net: Convolutional Networks
  for Biomedical Image Segmentation* — define a arquitetura U-Net.
- He et al. (2016), *Deep Residual Learning for Image Recognition* —
  o paper da ResNet.

### Remote sensing aplicado

- Ferrari et al. (2024–2026), `tree_fusion` paper se já estiver
  publicado — base direta do seu repo.
- Audebert et al. (2017), *Semantic Segmentation of Earth Observation
  Data Using Multimodal Fusion*.
- Maggiori et al. (2017), *Convolutional Neural Networks for
  Large-Scale Remote-Sensing Image Classification* (Inria Aerial dataset).

### Engenharia de dados geo

- Documentação GDAL/OGR ([gdal.org](https://gdal.org/)).
- Documentação Rasterio (`rasterio.readthedocs.io`) — wrapper Python
  mais amigável que `osgeo.gdal` direto.
- Tutorial Zarr ([zarr.readthedocs.io](https://zarr.readthedocs.io/)).

### Métricas e avaliação

- `torchmetrics` docs — todas as métricas com `ignore_index`,
  `multiclass`, etc.
- `scikit-learn.metrics` — `classification_report`,
  `confusion_matrix`.

---

## Apêndice — Dúvidas frequentes (que provavelmente vão aparecer)

**“Posso treinar sem GPU?”**
Pode, mas em CPU vai demorar dezenas de horas para uma epoch. Use a
RTX 4080. Confirme `torch.cuda.is_available() == True` no container.

**“Por que o modelo prevê tudo zero no começo?”**
Normal. As primeiras epochs, com pesos aleatórios, o modelo cospe
qualquer coisa; só após algumas iterações começa a aprender.

**“Diferença entre epoch, batch e iteration?”**
- **Epoch**: uma passada por todos os patches do train.
- **Batch**: subconjunto pequeno (8 patches) processado de uma vez.
- **Iteration**: uma chamada de `optimizer.step()` (= um batch).

**“O que é overfitting na prática?”**
Train F1 sobe, val F1 estaciona ou cai. O modelo decorou o train.
Early stopping resolve.

**“Posso aumentar o batch para 32?”**
Se a GPU aguentar. RTX 4080 com 12 GB deve aguentar 32 patches
256×256 em uint8 normalizado. Se der OOM, volte para 16.

**“Quantos patches são suficientes para treinar?”**
Não há número mágico. Para ResUNet num problema bem definido,
~1000 patches positivos costuma dar resultado razoável. Mais
sempre ajuda, especialmente com tiles diferentes.

**“Por que a inferência é mais lenta que o treino por epoch?”**
Inferência cobre **toda** a área (sem filtrar patches), com 3
overlaps. Pode demorar várias vezes mais que uma epoch.

---

Fim. Bom estudo. Quando aparecer dúvida específica em algum capítulo,
basta abrir um chat novo e cair direto na seção.
