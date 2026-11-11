# GeoTrees / GCA (vendored)

Protocolo de CHM do **Global Canopy Atlas / GeoTrees** (Fischer, Jackson, Vincent & Jucker 2024, *Methods in Ecology and Evolution*).

Arquivos trazidos dos testes IGC (`Downloads/`):

- `ALS_processing_helperfunctions.R` — pipeline completo (LAStools opcional; ramo open-source via lidR)
- `ALS_processing.R` — exemplo de chamada usado nos testes SP/IGC

No fluxo PhD use o wrapper:

```powershell
.\run_lidar.cmd -aoi <aoi> -Engine geotrees -Force
```

Isso chama `../make_lidar_products_geotrees.R` (`--protocol perfile` por default) e depois o align Docker (INTENSITY via PDAL).

Citação: https://doi.org/10.1111/2041-210X.14416  
Github GCA: https://github.com/fischer-fjd/GCA
