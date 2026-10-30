import os
for _v in ('PROJ_LIB', 'PROJ_DATA', 'GDAL_DATA'):
    os.environ.pop(_v, None)

import pandas as pd
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import regionprops

# osgeo e opcional no host Windows (Docker tem GDAL completo).
try:
    from osgeo import gdal
    gdal.UseExceptions()
    gdal.PushErrorHandler('CPLQuietErrorHandler')
except ImportError:
    pass

# Remove-Item Env:PROJ_LIB -ErrorAction SilentlyContinue
# Remove-Item Env:PROJ_DATA -ErrorAction SilentlyContinue
# Remove-Item Env:GDAL_DATA -ErrorAction SilentlyContinue

def _load_chm_on_grid(chm_path, height, width, transform, crs, band=1):
    """Le a banda CHM alinhada a grade de referencia (reprojeta se preciso)."""
    from rasterio.warp import reproject, Resampling
    with rasterio.open(chm_path) as src:
        chm = np.zeros((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, band),
            destination=chm,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
        )
        nodata = src.nodatavals[band - 1] if src.nodatavals else None
    if nodata is not None and np.isfinite(nodata):
        chm[chm == nodata] = np.nan
    chm[~np.isfinite(chm)] = 0.0
    return chm


def process_itc_rasterization(img_path, shp_path, out_base_dir, train_ratio=0.8,
                              apply_gaussian=True, gaussian_sigma=5, calc_distance=True, seed=42,
                              classes_dist_map=None, dist_map_as_cluster=False,
                              cluster_edge_m=10.0,
                              chm_path=None, chm_band=1, chm_min_m=3.0,
                              chm_filter_class=1):
    """
    Converte um shapefile em raster de treino e teste baseados em uma imagem de referência.
    Gera imagens rotuladas, e opcionalmente mapas de distância suavizados (Gaussiano)
    e normalizados individualmente por ITC ((val-min)/(max-min)).

    classes_dist_map: lista das classes (valor do raster de rótulos, DN_mapped)
    cujos polígonos entram no MAPA DE DISTÂNCIA. None = todas. Os polígonos das
    demais classes continuam no raster de rótulos e no vetor, mas ficam com
    distância 0 (como o fundo). O fundo (0) não tem polígonos e é ignorado.

    dist_map_as_cluster: muda o SIGNIFICADO do mapa de distância.
      False (copas): cada polígono é UMA árvore. Distância à borda normalizada
        por polígono (min-max) → cone com pico 1 no centro da copa e 0 na borda.
        Serve para localizar árvores individuais (pico = centro).
      True (maciços): cada polígono é um MACIÇO com várias árvores, embutido
        em floresta da MESMA altura. Não há borda física: um pixel a 5 m da
        linha é tão maciço quanto o centro. O cone por polígono, ou uma rampa
        que decai até a borda, descreveriam um objeto isolado no espaço — que
        não é o caso. O alvo passa a ser a PERTINÊNCIA ao maciço: 1 dentro, 0
        fora. A única transição suave é a INCERTEZA DA LINHA digitizada
        (limite difuso de composição): uma rampa SIMÉTRICA centrada na borda,
        de largura total `cluster_edge_m` (metade para dentro, metade para
        fora), valendo 0,5 exatamente sobre a linha — a probabilidade de o
        pixel pertencer ao maciço dado um erro de ±cluster_edge_m/2 no
        traçado. cluster_edge_m = 0 → máscara dura. Igual para maciços
        grandes e pequenos, independente da resolução; polígonos encostados
        formam um só maciço (sem vale entre eles). Polígonos de classes fora
        de classes_dist_map ficam exatamente em 0.
    cluster_edge_m: largura TOTAL (m) da faixa de incerteza da borda (só no
      modo maciço). ~10 m (±5 m) para limites traçados sobre imagem de 2 m.
    """
    
    print("1. Lendo a imagem de referência...")
    with rasterio.open(img_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs

    print("2. Lendo e preparando o Shapefile...")
    gdf = gpd.read_file(shp_path)
    
    if gdf.crs != crs:
        print(f"   Reprojetando shapefile para {crs}...")
        gdf = gdf.to_crs(crs)
    
    print("   Calculando remapeamento de classes...")
    unique_dns = sorted(gdf['DN'].dropna().unique())
    mapping = {int(val): i + 1 for i, val in enumerate(unique_dns)}
    
    # Classes que entram no mapa de distância (None = todas)
    if classes_dist_map is not None:
        classes_dist_map = sorted({int(c) for c in classes_dist_map})
        ausentes = [c for c in classes_dist_map if c not in mapping.values()]
        if ausentes:
            print(f"   ⚠️  classes_dist_map: {ausentes} sem polígonos (0 = fundo) — ignoradas.")
        if not set(classes_dist_map) & set(mapping.values()):
            raise ValueError(f"classes_dist_map={classes_dist_map} não contém nenhuma das "
                             f"classes existentes {sorted(mapping.values())}.")

    # Salva o arquivo de resumo TXT das classes
    out_txt_path = os.path.join(out_base_dir, "ITCs_classes_report.txt")
    os.makedirs(out_base_dir, exist_ok=True)
    with open(out_txt_path, 'w', encoding='utf-8') as f:
        f.write("Original_DN -> Mapped_Class\n")
        for orig, mapped in mapping.items():
            f.write(f"{orig} -> {mapped}\n")
        if calc_distance:
            f.write(f"Classes no mapa de distancia: "
                    f"{classes_dist_map if classes_dist_map is not None else 'todas'}\n")
            f.write("Modo do mapa de distancia: "
                    + (f"macico (pertinencia 1 dentro / 0 fora, incerteza da borda "
                       f"+-{cluster_edge_m / 2:g} m)"
                       if dist_map_as_cluster else "copa (cone normalizado por poligono)")
                    + "\n")

    print(f"👉 Dicionário de conversão salvo em: {out_txt_path}")

    # Aplica o mapeamento de classes
    gdf['DN_mapped'] = gdf['DN'].map(mapping)
    
    print(f"\n3. Dividindo os polígonos em {train_ratio*100}% Treino e {(1-train_ratio)*100}% Teste...")
    
    # Seleciona uma fração de polígonos mantendo a proporção de cada classe.
    # Evita groupby().apply() (pandas >=2.2 pode dropar a coluna de grupo).
    train_parts = []
    for _, grp in gdf.groupby('DN_mapped', group_keys=False):
        n = max(1, int(round(len(grp) * train_ratio))) if len(grp) > 1 else len(grp)
        n = min(n, len(grp))
        train_parts.append(grp.sample(n=n, random_state=seed))
    train_gdf = gpd.GeoDataFrame(
        pd.concat(train_parts, axis=0), crs=gdf.crs
    ) if train_parts else gdf.iloc[0:0].copy()
    test_gdf = gdf.drop(index=train_gdf.index)
    
    print(f"   📈 Total de polígonos no Shapefile: {len(gdf)}")
    print(f"   🔹 Polígonos selecionados para TREINO: {len(train_gdf)}")
    print(f"   🔸 Polígonos selecionados para TESTE: {len(test_gdf)}")
    
    splits = {
        'train': {
            'gdf': train_gdf,
            'label_path': os.path.join(out_base_dir, "ITCs_labels_train.tif"),
            'dist_path': os.path.join(out_base_dir, "ITCs_distance_train.tif"),
            'vector_path': os.path.join(out_base_dir, "ITCs_crowns_train.gpkg")
        },
        'test': {
            'gdf': test_gdf,
            'label_path': os.path.join(out_base_dir, "ITCs_labels_test.tif"),
            'dist_path': os.path.join(out_base_dir, "ITCs_distance_test.tif"),
            'vector_path': os.path.join(out_base_dir, "ITCs_crowns_test.gpkg")
        }
    }
    
    chm = None
    if chm_path:
        print(f"\n2b. Lendo CHM para filtrar classe {chm_filter_class} "
              f"(manter so onde CHM > {chm_min_m} m)...")
        chm = _load_chm_on_grid(chm_path, height, width, transform, crs, band=chm_band)
        print(f"   CHM: min={float(np.nanmin(chm)):.2f} max={float(np.nanmax(chm)):.2f} m")

    # --- LOOP DE PROCESSAMENTO PARA TREINO E TESTE ---
    for split_name, split_data in splits.items():
        print(f"\n 🚀 Processando conjunto de {split_name.upper()}...")
        # Copia o GeoDataFrame atual para evitar SettingWithCopyWarning
        current_gdf = split_data['gdf'].copy()
        
        # 4. Rasterização dos Rótulos (Classes)
        shapes_classes = ((geom, value) for geom, value in zip(current_gdf.geometry, current_gdf['DN_mapped']))
        labeled_array = rasterize(
            shapes=shapes_classes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=rasterio.int32
        )

        # Buracos de leucaena com CHM baixo -> classe 0 (fundo)
        if chm is not None:
            low = (labeled_array == int(chm_filter_class)) & (chm <= float(chm_min_m))
            n_low = int(low.sum())
            labeled_array[low] = 0
            print(f"   CHM <= {chm_min_m} m na classe {chm_filter_class}: "
                  f"{n_low:,} px -> classe 0")
        
        meta_label = meta.copy()
        meta_label.update({'count': 1, 'dtype': rasterio.uint8, 'nodata': 0})
        with rasterio.open(split_data['label_path'], 'w', **meta_label) as dst:
            dst.write(labeled_array.astype(np.uint8), 1)
        print(f"   ✅ Imagem rotulada salva: {split_data['label_path']}")

        # 4b. Exporta o VETOR do split (apenas as copas deste conjunto).
        # CRÍTICO p/ a extração de patches com "copas inteiras": o extrator deve
        # usar o vetor que corresponde EXATAMENTE ao raster LABEL deste split,
        # senão copas do outro split apareceriam como obstáculos de borda.
        # O 'instance_id' é gravado para que os IDs do extrator batam com os da
        # máscara de distância (mesma indexação 1..N sobre ESTE subconjunto).
        current_gdf['instance_id'] = range(1, len(current_gdf) + 1)
        gpkg_cols = ['geometry', 'DN', 'DN_mapped', 'instance_id']
        gpkg_cols = [c for c in gpkg_cols if c in current_gdf.columns or c == 'geometry']
        current_gdf[gpkg_cols].to_file(split_data['vector_path'], driver='GPKG')
        print(f"   ✅ Vetor do split salvo: {split_data['vector_path']} "
              f"({len(current_gdf)} copas)")
        
        # 5. Transformação de Distância (Isolando por ITC e Suavizando)
        if calc_distance:
            # 'instance_id' já foi criado acima (passo 4b) — reaproveita os
            # mesmos IDs, garantindo coerência entre distância, vetor e extração.
            # Só os polígonos das classes de classes_dist_map entram (os demais
            # ficam com distância 0, como o fundo).
            dist_gdf = current_gdf
            if classes_dist_map is not None:
                dist_gdf = current_gdf[current_gdf['DN_mapped'].isin(classes_dist_map)]
                print(f"   Mapa de distância só com as classes {classes_dist_map}: "
                      f"{len(dist_gdf)} de {len(current_gdf)} polígonos.")
            if len(dist_gdf):
                shapes_instances = ((geom, value) for geom, value in zip(dist_gdf.geometry, dist_gdf['instance_id']))
                instance_array = rasterize(
                    shapes=shapes_instances,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=rasterio.int32
                )
            else:   # split sem polígonos dessas classes → mapa todo 0
                instance_array = np.zeros((height, width), dtype=np.int32)

            # Distancia so onde leucaena sobreviveu ao filtro CHM (label ainda = classe alvo)
            if chm is not None and int(chm_filter_class) in (
                    set(classes_dist_map) if classes_dist_map is not None
                    else set(np.unique(labeled_array))):
                drop = (instance_array > 0) & (
                    (labeled_array != int(chm_filter_class)) | (chm <= float(chm_min_m))
                )
                n_drop = int(drop.sum())
                instance_array[drop] = 0
                print(f"   Distancia: removidos {n_drop:,} px de instancia com "
                      f"CHM <= {chm_min_m} m (ou label != {chm_filter_class})")

            inside = instance_array > 0

            if dist_map_as_cluster:
                # ── MODO MACIÇO: pertinência (1 dentro, 0 fora) com transição
                # SIMÉTRICA na borda = incerteza do traçado, não borda física
                # (o maciço tem a mesma altura da floresta em volta). Em METROS
                # (sampling = tamanho do pixel), p/ não depender da resolução.
                px_x, px_y = abs(transform.a), abs(transform.e)
                if crs is not None and crs.is_geographic:
                    # grade em graus → converte p/ metros na latitude central
                    lat_c = np.deg2rad(transform.f + transform.e * height / 2)
                    px_x, px_y = px_x * 111320 * np.cos(lat_c), px_y * 111320
                    print(f"   ⚠️  CRS geográfico: pixel ≈ {px_x:.2f} x {px_y:.2f} m (aproximado).")
                if cluster_edge_m > 0:
                    print(f"   Distância assinada à linha (m) → rampa de ±{cluster_edge_m / 2:g} m "
                          f"centrada na borda (0,5 sobre a linha)...")
                    # distância assinada: + dentro, − fora
                    s = distance_transform_edt(inside, sampling=(px_y, px_x)).astype(np.float32)
                    d_out = distance_transform_edt(~inside, sampling=(px_y, px_x))
                    np.negative(d_out, out=d_out)
                    np.copyto(s, d_out, where=~inside, casting='unsafe')
                    del d_out
                    dist_array = np.clip(0.5 + s / cluster_edge_m, 0.0, 1.0).astype(np.float32)
                    del s
                else:
                    print("   cluster_edge_m = 0 → máscara dura (1 dentro, 0 fora).")
                    dist_array = inside.astype(np.float32)
                if apply_gaussian:
                    # suavização extra, também simétrica (não se re-mascara ao interior)
                    print(f"   Aplicando filtro Gaussiano (sigma={gaussian_sigma})...")
                    dist_array = gaussian_filter(dist_array, sigma=gaussian_sigma)
                # polígonos das classes FORA de classes_dist_map: exatamente 0
                if classes_dist_map is not None:
                    dist_array[(labeled_array > 0) & ~inside] = 0.0
                n_in = int(inside.sum())
                if n_in:
                    fora = dist_array[~inside]
                    print(f"   Dentro dos maciços: {100 * (dist_array[inside] >= 0.999).mean():.1f}% em 1 "
                          f"| {100 * (dist_array[inside] < 0.5).mean():.1f}% abaixo de 0,5 (faixa interna) "
                          f"| fora: {int((fora > 0).sum()):,} px > 0 (faixa externa, máx {fora.max():.2f}).")
            else:
                # ── MODO COPA: cone normalizado (min-max) por polígono ──
                # Aplica EDT global em todo o raster binário
                print("   Calculando Distância Euclidiana...")
                global_dist = distance_transform_edt(inside)

                if apply_gaussian:
                    print(f"   Aplicando filtro Gaussiano (sigma={gaussian_sigma})...")
                    global_dist = gaussian_filter(global_dist, sigma=gaussian_sigma)

                dist_array = np.zeros_like(global_dist, dtype=np.float32)

                # Itera por cada ITC usando os IDs únicos para normalizar de 0 a 1 localmente
                props = regionprops(instance_array)

                for prop in props:
                    mask = prop.image
                    min_row, min_col, max_row, max_col = prop.bbox

                    # Extrai os valores do mapa global pertencentes APENAS a este polígono
                    obj_dist = global_dist[min_row:max_row, min_col:max_col].copy()
                    obj_values = obj_dist[mask]

                    min_val = obj_values.min()
                    max_val = obj_values.max()

                    if max_val > min_val:
                        # Normaliza: (Valor - Mínimo) / (Máximo - Mínimo)
                        obj_dist[mask] = (obj_values - min_val) / (max_val - min_val)
                    else:
                        obj_dist[mask] = 0.0

                    # Atualiza o array final apenas nas áreas verdadeiras da máscara
                    dist_array[min_row:max_row, min_col:max_col][mask] = obj_dist[mask]
                
            meta_dist = meta.copy()
            meta_dist.update({'count': 1, 'dtype': rasterio.float32, 'nodata': 0})
            with rasterio.open(split_data['dist_path'], 'w', **meta_dist) as dst:
                dst.write(dist_array, 1)
            print(f"   ✅ Imagem de distância salva: {split_data['dist_path']}")
        else:
            print("   ⏭️ Cálculo do mapa de distância ignorado (calc_distance=False).")
            
    print("\n🎉 Todo o pipeline de divisão e rasterização foi concluído com sucesso!")

def _as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'sim', 's')


def _lista_classes(valores):
    """Aceita '--classes_dist_map 0 1 2' ou '--classes_dist_map [0,1,2]'."""
    import re
    nums = re.findall(r'-?\d+', ' '.join(valores))
    if not nums:
        raise SystemExit(f"--classes_dist_map: nenhuma classe reconhecida em {valores}")
    return [int(n) for n in nums]


# --- Definição dos Caminhos e Execução ---
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Rasteriza os polígonos rotulados em rótulos "
                                            "e mapas de distância (treino/teste).")
    # Defaults adaptados para Rio das Pedras (leucaena). Scripts originais do
    # professor apontavam para CBERS/Marajó; aqui a grade vem do RGBNIR 0.25 m.
    # Defaults vazios: use run_01_build_labels.ps1 (le DIR_DATASETS_AI_PATH do .env).
    p.add_argument('--img_path', default="",
                   help='Imagem de referência (grade de saída).')
    p.add_argument('--shp_path', default="",
                   help='Polígonos rotulados (campo DN = classe).')
    p.add_argument('--out_base_dir', default="")
    p.add_argument('--train_ratio', type=float, default=0.8,
                   help='Fração dos polígonos (por classe) no treino; o resto vai p/ teste.')
    p.add_argument('--apply_gaussian', type=_as_bool, default=True,
                   help='Suaviza o mapa de distância com filtro Gaussiano.')
    p.add_argument('--gaussian_sigma', type=float, default=2,
                   help='Sigma do filtro Gaussiano (px).')
    p.add_argument('--calc_distance', type=_as_bool, default=True,
                   help='Gera os mapas de distância.')
    p.add_argument('--seed', type=int, default=27)
    # DN original 1=leucaena → após remapeamento ordenado vira classe 1 no raster.
    # Só a leucaena entra no mapa de distância (análogo ao maciço de açaí).
    p.add_argument('--classes_dist_map', nargs='+', default=['1'],
                   help='Classes (valor no raster de rótulos) cujos polígonos entram no '
                        'mapa de distância; as demais ficam com distância 0. Ex.: '
                        '--classes_dist_map 1 2  ou  --classes_dist_map [1,2]. '
                        'Padrão neste experimento: 1 (leucaena).')
    p.add_argument('--dist_map_as_cluster', type=_as_bool, default=True,
                   help='False: cada polígono é UMA copa → cone normalizado por polígono '
                        '(pico 1 no centro, 0 na borda). True (padrão): cada polígono é um '
                        'MACIÇO de várias árvores embutido em floresta da mesma altura → '
                        'mapa de PERTINÊNCIA: 1 dentro, 0 fora, com rampa simétrica de '
                        'largura --cluster_edge_m centrada na linha (0,5 sobre ela) que '
                        'representa a incerteza do traçado, não uma borda física. Igual '
                        'para maciços grandes e pequenos e independente da resolução.')
    # ~10 m (±5 m) — docstring do professor para limites digitados em imagem fina.
    p.add_argument('--cluster_edge_m', type=float, default=10.0,
                   help='[só com --dist_map_as_cluster True] Largura TOTAL (m) da faixa de '
                        'incerteza da borda, metade para cada lado da linha. 0 = máscara '
                        'dura; 1 m em pixel de 2 m já equivale a máscara dura (resta só a '
                        'suavização gaussiana). ~10 m (±5 m) se o traçado for incerto.')
    p.add_argument('--chm_path', default="",
                   help='Raster CHM (mesma AOI). Se fornecido, pixels da classe '
                        '--chm_filter_class com CHM <= --chm_min_m viram classe 0 '
                        'e saem do mapa de distancia.')
    p.add_argument('--chm_band', type=int, default=1,
                   help='Banda do CHM no raster (1 = primeira).')
    p.add_argument('--chm_min_m', type=float, default=2.0,
                   help='Altura minima (m) para manter leucaena / distancia.')
    p.add_argument('--chm_filter_class', type=int, default=1,
                   help='Classe no raster de rotulos a filtrar pelo CHM (1=leucaena).')
    a = p.parse_args()

    # Execute a função com suporte ao filtro Gaussiano e Normalização per-ITC
    process_itc_rasterization(
        img_path=a.img_path,
        shp_path=a.shp_path,
        out_base_dir=a.out_base_dir,
        train_ratio=a.train_ratio,
        apply_gaussian=a.apply_gaussian,
        gaussian_sigma=a.gaussian_sigma,
        calc_distance=a.calc_distance,
        seed=a.seed,
        classes_dist_map=(_lista_classes(a.classes_dist_map)
                          if a.classes_dist_map is not None else None),
        dist_map_as_cluster=a.dist_map_as_cluster,
        cluster_edge_m=a.cluster_edge_m,
        chm_path=(a.chm_path or None),
        chm_band=a.chm_band,
        chm_min_m=a.chm_min_m,
        chm_filter_class=a.chm_filter_class,
    )