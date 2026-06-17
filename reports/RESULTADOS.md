# Detecção de *Leucaena leucocephala* por Segmentação Semântica — Resultados

> **Projeto de Doutorado** — Mapeamento e estimativa de biomassa de leucaena com Deep Learning
> **Instituição:** ESALQ/USP
> **Autor:** Matheus Siba · **Orientador:** [PREENCHER]
> **Área de estudo:** AOI Piracicaba (SP) — imagens IGC-SP + LiDAR
> **Data da apresentação:** [PREENCHER]

---

> ### 💡 Como usar este documento
> Este é um **template**. Onde estiver `[PREENCHER]` você troca pelo valor real;
> os blocos `💡` te dizem de onde vem cada número/figura (e em qual comando ele é
> gerado — veja [`COMO-GERAR-FIGURAS.md`](COMO-GERAR-FIGURAS.md)).
> Quando estiver preenchido, dá pra exportar pra PDF/Word/slides (peça que eu converto).
> Apague os blocos `💡` na versão final que vai pro professor.

---

## 1. Objetivo e hipótese

**Objetivo.** Treinar um modelo de *segmentação semântica* que detecte, pixel a pixel,
copas de leucaena em imagens aéreas, distinguindo-as do fundo (pasto, solo, outras
árvores, construções).

**Problema observado.** O modelo baseline (só imagem óptica) gerava **muitos falsos
positivos** — chamava de leucaena coisas que não são (pasto alto, clareiras, telhados).

**Hipótese central.** Usar informação de **altura (CHM, do LiDAR)** e de **vigor de
vegetação (NDVI)** — tanto para *refinar os rótulos* de treino quanto como *entradas*
do modelo — reduz essa confusão, aumentando a **precisão** da classe leucaena sem
perder muita revocação.

---

## 2. Dados

| Item | Valor |
|---|---|
| Sensor óptico | IGC-SP, 4 bandas (B, G, R, NIR), ~25 cm/pixel |
| LiDAR | CHM (altura de copa) + intensidade, rasterizado a 1 m e reamostrado p/ a grade óptica |
| Anotações | Polígonos de leucaena coletados na plataforma [leucaena.earth](https://leucaena.earth) |
| Tiles usados | [PREENCHER — ex.: 17 tiles SF-23-Y-A-IV-2-\*] |
| Tamanho do patch | 256 × 256 px |
| Overlap entre patches | 60% |
| Total de patches | **[PREENCHER]** (ex.: 9.341) |
| Split train / val / test | **[PREENCHER]** / [PREENCHER] / [PREENCHER] |

> 💡 Os números de patches e splits estão em `/data/patches/preparation.txt` e no
> rodapé do `manifest.csv`. O log de preparação também mostra, por tile, quantos
> pixels viraram leucaena(1), background(0) e IGNORE(255).

**Refino de rótulo (regra do orientador).** Dentro dos polígonos anotados, um pixel só
é rotulado como leucaena(1) quando **CHM ≥ 4,5 m E NDVI ≥ 0,3**; senão vira
background(0). Tudo **fora** dos polígonos é `IGNORE (255)` — não entra no treino nem
nas métricas. Detalhes em [`../studies/labeling-refinado.md`](../studies/labeling-refinado.md).

---

## 3. Método

### 3.1 Pipeline
1. **Preparo** — recorta tiles + LiDAR em patches, rasteriza polígonos, aplica o refino de rótulo, separa train/val/test.
2. **Treino** — ResUNet (U-Net com blocos residuais), até 300 épocas, *early stopping*.
3. **Predição** — janela deslizante sobre a cena inteira → mapa GeoTIFF.
4. **Avaliação** — métricas por classe + matriz de confusão.

### 3.2 O modelo apresentado (exp_4 — *early fusion* + NDVI)

O modelo desta apresentação é o **exp_4**, que combina, na entrada (7 canais):

| Fonte | Canais |
|---|---|
| Óptico | B, G, R, NIR |
| Índice de vegetação | **NDVI** (calculado de NIR e R) |
| LiDAR | **CHM** (altura de copa), **INTENSIDADE** |

Arquitetura: **ResUNet** com *early fusion* (todas as fontes empilhadas antes do
encoder). Assim o modelo enxerga, em cada pixel, a cor, o vigor de vegetação (NDVI)
e a altura (CHM) ao mesmo tempo.

> 💡 **Comparação com baselines** (óptico puro / fusão sem NDVI) está nos *próximos
> passos* (Seção 8) — esta reunião foca em apresentar o exp_4 e a viabilidade da abordagem.

### 3.3 Métricas — e uma ressalva importante

- **Precision (leucaena)** = dos pixels que o modelo chamou de leucaena, quantos eram mesmo. *Precision baixa = muito falso positivo.*
- **Recall (leucaena)** = de toda a leucaena real, quanto o modelo achou. *Recall baixo = leucaena perdida.*
- **F1** = média harmônica de precision e recall (equilíbrio dos dois).
- **IoU** = interseção sobre união (sobreposição entre predito e real).

> ⚠️ **Ressalva metodológica (dizer ao professor):** todas as métricas são calculadas
> **apenas dentro dos polígonos anotados** (fora = IGNORE). Logo, elas **não medem**
> os falsos positivos na paisagem aberta (pasto, telhado fora de polígono). Para isso
> usamos a **inspeção visual do mapa de predição** (Seção 5.2) e/ou uma tile totalmente
> anotada (próximos passos).

---

## 4. Resultados quantitativos

### 4.1 Métricas do exp_4 (conjunto de **teste**)

| Classe | Precision | Recall | F1 | IoU | Suporte (px) |
|--------|----------:|-------:|---:|----:|-------------:|
| background | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] |
| **leucaena** | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] |
| *macro F1* | | | [PREENCHER] | | |

> 💡 Tudo sai de `python eval-patches.py -e 4 --split test`
> (relatório salvo em `experiments/exp_4/logs/eval_patches_test.txt`).
> **Foque na linha leucaena, coluna Precision** — é a métrica ligada ao seu problema de falso positivo.

### 4.2 Matriz de confusão (melhor modelo — exp_4)

|  | Predito: background | Predito: leucaena |
|---|---|---|
| **Real: background** | TN = [PREENCHER] | FP = [PREENCHER] |
| **Real: leucaena** | FN = [PREENCHER] | TP = [PREENCHER] |

> 💡 Sai no mesmo relatório do `eval-patches.py`. FP = falso positivo, FN = leucaena perdida.

### 4.3 Curvas de treino (exp_4)

![Curvas de treino exp_4](figuras/exp4_training_curves.png)

> 💡 Gerada por `python -m utils.plot_training -e 4`. Mostra loss e F1 (treino vs.
> validação) por época. **O que comentar:** val loss caindo e estabilizando =
> aprendizado saudável; val próximo do treino = sem overfitting grave.

---

## 5. Resultados qualitativos

> **Total: 5 figuras (+ curvas = 6 no relatório).** Cada painel mostra as camadas
> que o modelo enxerga (RGB, CIR, NDVI, CHM, rótulo) ou a predição sobre o rótulo real.

### 5.1 Patch com muita leucaena (área densa)

![Patch densa](figuras/patch_densa.png)

> 💡 `python reports/fill-resultados.py` gera isso automaticamente.
> Mostra RGB · CIR · NDVI · CHM · rótulo refinado do patch com maior fração de leucaena.

### 5.2 Patch com pouca leucaena (área difícil)

![Patch difícil](figuras/patch_dificil.png)

> 💡 Gerado automaticamente. Patch em que o modelo tem mais dificuldade — baixa fração
> de leucaena, misturado com fundo. Útil para mostrar a complexidade do problema.

### 5.3 Predição — exemplo bom (acerto)

![Predição boa](figuras/pred_boa.png)

> 💡 `inspect_validation_errors.py -e 4 --split val --top-k 1 --rank-by f1`
> Sobreposição: rótulo real vs. predição. Verde = verdadeiro positivo.

### 5.4 Predição — exemplo difícil (erros)

![Predição difícil](figuras/pred_dificil.png)

> 💡 `inspect_validation_errors.py -e 4 --split val --top-k 1 --rank-by fp`
> Vermelho = falso positivo (modelo chamou leucaena mas era fundo).
> Mostrar o erro é honestidade científica — e justifica os próximos passos.

### 5.5 Mapa de predição na cena completa (QGIS)

![Mapa de predição](figuras/mapa_predicao.png)

> 💡 Gerado por `prediction.py -e 4` → GeoTIFF. No QGIS, sobreponha sobre a imagem real.
> Mostre: (a) uma área com muita leucaena bem detectada; (b) uma área com falso positivo.
> Veja [`COMO-GERAR-FIGURAS.md`](COMO-GERAR-FIGURAS.md).

---

## 6. Limitações

- Métricas medidas só **dentro dos polígonos** (não capturam FP na paisagem aberta).
- Treino só com tiles que têm LiDAR (sem CHM, a regra de refino não se aplica).
- Polígonos de *crowdmapping* podem ter ruído / leucaena não anotada.
- [PREENCHER — outras que você notar]

## 7. Próximos passos

1. **Comparação com baselines** — treinar exp_1 (óptico puro) e exp_2 (fusão sem NDVI)
   com o mesmo split, para isolar e quantificar o ganho do CHM e do NDVI.
2. **Negativos confiáveis (hard negatives)** fora dos polígonos (água, telhado, solo) para o modelo aprender o que *não* é leucaena na paisagem aberta.
3. **Tile totalmente anotada** para medir falso positivo de verdade (não só dentro de polígono).
4. **Funções de perda** focadas em desbalanceamento (Dice/Tversky/Focal).
5. **Escala** — rodar o pipeline tile-by-tile em área maior.
6. [PREENCHER — o que o professor sugerir na reunião]

---

## Apêndice — Configuração (reprodutibilidade)

| Hiperparâmetro | Valor |
|---|---|
| Arquitetura | ResUNet (early fusion no exp_2/exp_4) |
| Patch / overlap | 256 px / 60% |
| Batch size | 20 |
| Learning rate | 1e-4 (Adam), decay 0.995/época |
| Pesos de classe | [bg 0.3, leucaena 0.7] |
| Early stopping | paciência 15, mín. 5 épocas (1º save na época 6) |
| Augmentation | rotação 0–360°, translação ±10%, flips (on-the-fly) |
| Limiares de refino | CHM ≥ 4,5 m · NDVI ≥ 0,3 |

> 💡 Tudo em [`../conf/general.py`](../conf/general.py). Cite a versão do código
> (hash do commit) pra reprodutibilidade: `git rev-parse --short HEAD`.
