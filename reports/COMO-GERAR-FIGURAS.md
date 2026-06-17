# Como gerar cada figura/tabela do relatório

Guia passo a passo para produzir **todo** o material de [`RESULTADOS.md`](RESULTADOS.md).
Os comandos Python rodam **dentro do container Docker**
(`docker compose run --rm segmentation bash`). Os de QGIS, no Windows.

Saída final em `reports/figuras/` — **máximo 6 figuras** no relatório.

---

## Visão geral das 6 figuras

| # | Arquivo | Seção | Como gera |
|---|---------|-------|-----------|
| 1 | `exp4_training_curves.png` | 4.3 | automático via `fill-resultados.py` |
| 2 | `patch_densa.png` | 5.1 | automático via `fill-resultados.py` |
| 3 | `patch_dificil.png` | 5.2 | automático via `fill-resultados.py` |
| 4 | `pred_boa.png` | 5.3 | manual (ver passo 3a abaixo) |
| 5 | `pred_dificil.png` | 5.4 | manual (ver passo 3b abaixo) |
| 6 | `mapa_predicao.png` | 5.5 | manual no QGIS (ver passo 4) |

---

## Atalho pós-treino — figuras 1, 2 e 3 em 1 comando

```bash
# dentro do container:
python reports/fill-resultados.py
```

Isso faz tudo de uma vez:
- Roda `eval-patches.py -e 4 --split test` → preenche tabelas 4.1 e 4.2
- Gera `training_curves.png` → copia para `reports/figuras/exp4_training_curves.png`
- Seleciona patch com **mais leucaena** → `reports/figuras/patch_densa.png`
- Seleciona patch **difícil** (pouca leucaena) → `reports/figuras/patch_dificil.png`
- Preenche todos os `[PREENCHER]` do `RESULTADOS.md`

---

## Passo 3 — Figuras de predição (4 e 5, manual)

### 3. Predição boa + difícil (Figs 5.3 e 5.4)
```bash
# Gera os 2 piores patches por F1 (rank 1 = mais erros, rank 2 = segundo pior)
python inspect_validation_errors.py -e 4 --split val --top-k 2 --rank-by f1

# Copiar para reports/figuras/
cp experiments/exp_4/diagnostics/val/panels/001_*.png reports/figuras/pred_dificil.png
cp experiments/exp_4/diagnostics/val/panels/002_*.png reports/figuras/pred_boa.png
```
- `--rank-by` aceita: `loss`, `iou`, `f1` (ascendente para iou/f1 = pior primeiro)
- O rank 1 tem mais erros (FP/FN); o rank 2 é um pouco melhor — use como "exemplo bom"

---

## Passo 4 — Mapa QGIS (Fig 5.5, manual)

### 4a. Gerar o GeoTIFF de predição
```bash
python prediction.py -e 4 -i /data/rgbir/<NOME_DA_TILE>.tif
```
- Gera: `experiments/exp_4/predicted/pred_4.tif`

### 4b. Montar a figura no QGIS (Windows)
1. Abra a tile RGBN no QGIS.
2. Adicione `pred_4.tif` por cima.
3. Estilo: *Paletted/Unique values* → classe 1 = laranja sólido, classe 0 = transparente. Opacidade ~50%.
4. Mostre **duas áreas**: (a) muita leucaena bem detectada; (b) área com falso positivo.
5. **Project → Import/Export → Export Map to Image** → salve em `reports/figuras/mapa_predicao.png`.

---

## Checklist final

- [ ] `exp4_training_curves.png` — gerado pelo fill-resultados.py
- [ ] `patch_densa.png` — gerado pelo fill-resultados.py
- [ ] `patch_dificil.png` — gerado pelo fill-resultados.py
- [ ] `pred_boa.png` — copiado de `experiments/exp_4/diagnostics/`
- [ ] `pred_dificil.png` — copiado de `experiments/exp_4/diagnostics/`
- [ ] `mapa_predicao.png` — exportado do QGIS
- [ ] `[PREENCHER]` de orientador e data preenchidos manualmente
- [ ] Blocos `💡` removidos da versão final
- [ ] Hash do commit: `git rev-parse --short HEAD`

---

## Exportar pro professor

Quando o `RESULTADOS.md` estiver preenchido, peça para eu converter em:
- **PDF** (documento formal), ou
- **PowerPoint** (~8–10 slides).
