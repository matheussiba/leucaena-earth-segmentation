# Como gerar cada figura/tabela do relatório

Guia passo a passo para produzir **todo** o material de [`RESULTADOS.md`](RESULTADOS.md).
Os comandos Python rodam **dentro do container Docker**
(`docker compose run --rm segmentation bash`). Os de QGIS, no Windows.

Saída final organizada em `reports/figuras/`.

---

## Ordem recomendada

```
1. Treinar (já está rodando)          -> experiments/exp_4/...
2. Rodar fill-resultados.py           -> preenche RESULTADOS.md automaticamente
3. Painéis de patches (manual)        -> figura 5.1
4. Mapa de predição (QGIS)            -> figura 5.2
5. Exportar para PDF / slides
```

### Atalho pós-treino (1 comando faz tudo)

```bash
# dentro do container:
python reports/fill-resultados.py
# -> roda eval-patches.py, gera curvas, preenche RESULTADOS.md
```

---

## 1. Curvas de treino (Seção 4.3)

```bash
# dentro do container
python -m utils.plot_training -e 4
```
- **Gera:** `experiments/exp_4/logs/training_curves.png`
- **Copie para:** `reports/figuras/exp4_training_curves.png`
- Tabela numérica das épocas: `python -m utils.plot_training -e 4 --table`

---

## 2. Métricas por classe + matriz de confusão (Tabelas 4.1 e 4.2)

```bash
python eval-patches.py -e 4 --split test
```
- **Gera:** `experiments/exp_4/logs/eval_patches_test.txt` (Precision/Recall/F1/IoU por classe + TP/FP/FN/TN)
- **Use:** copie os números para a tabela 4.1 (background + leucaena) e a matriz de confusão 4.2.
- 💡 Quando for fazer a comparação (próximos passos), rode também `-e 1` e `-e 2` e volte a tabela 4.1 para o formato comparativo.

---

## 3. Painéis de patches (Seção 5.1)

**Exemplos de camadas (o que o modelo enxerga):**
```bash
python viz-patches.py \
  --patches-dir /data/patches \
  --select most-leucaena --top-k 6 \
  --out-dir reports/figuras/patches_camadas
```
- **Gera:** PNGs com RGB, infravermelho (CIR), NDVI, CHM e rótulo.

**Acertos e erros (precisa do modelo treinado):**
```bash
python inspect_validation_errors.py -e 4 --split val --top-k 6 --rank-by f1
```
- **Gera:** `experiments/exp_4/diagnostics/val/panels/*.png` (TP verde, FP vermelho, FN azul).
- 💡 Para o relatório, monte um painel com 3 patches bons + 1–2 difíceis. Mostrar erro = honestidade científica.

---

## 4. Mapa de predição na cena — QGIS (Seção 5.2)

### 4a. Gerar a predição
```bash
# cena única (RAM-bound, ok para 1 tile):
python prediction.py -e 4 -i /data/rgbir/<NOME_DA_TILE>.tif

# OU várias tiles (escala):
python predict-tiles.py -e 4
```
- **Gera:** `experiments/exp_4/predicted/pred_4.tif` (mapa de classe) e `pred_probs_4.tif` (probabilidade).

### 4b. Montar a figura no QGIS (Windows)
1. Abra a **imagem real** (a tile RGBN) no QGIS.
2. Adicione o **`pred_4.tif`** por cima.
3. Estilo do `pred_4.tif`: *Paletted/Unique values* → classe 1 = vermelho/laranja sólido, classe 0 = transparente. Opacidade ~50%.
4. Enquadre **duas áreas**: (a) onde acerta bem; (b) onde tem falso positivo.
5. **Project → Import/Export → Export Map to Image** → salve em `reports/figuras/mapa_predicao.png`.
6. (Opcional) Adicione escala e norte (Decorations) pra ficar profissional.

---

## 5. Checklist final

- [ ] `exp4_training_curves.png` em `reports/figuras/`
- [ ] `eval_patches_test.txt` de cada exp treinado → tabelas 4.1 / 4.2 preenchidas
- [ ] Painel de patches (`patches_exemplos.png`) montado
- [ ] `mapa_predicao.png` exportado do QGIS
- [ ] Todos os `[PREENCHER]` do `RESULTADOS.md` substituídos
- [ ] Blocos `💡` removidos da versão final
- [ ] Hash do commit anotado (`git rev-parse --short HEAD`)

---

## 6. Exportar pro professor

Quando o `RESULTADOS.md` estiver preenchido, peça para eu converter em:
- **PDF / Word** (documento formal), ou
- **PowerPoint** (deck de reunião, ~8–10 slides).

Eu monto a partir do Markdown preenchido — só avisar o formato.
