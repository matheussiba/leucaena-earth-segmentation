# Plans

Numbered plans that drive this repository. Each file describes one piece of
work (scope, decisions, files touched, risks). Keep the numbering stable;
append new plans by incrementing the prefix.

## Status legend

| Status            | Meaning |
|-------------------|---------|
| `not started`     | Plan written, nothing implemented yet |
| `partial`         | Some todos done, others pending; see the plan file |
| `built`           | All todos implemented and merged to `main` |
| `built + deployed`| `built` plus running successfully on the target machine |
| `abandoned`       | Plan was superseded or rejected (kept for history) |

## Index

| #  | Plan | Status |
|----|------|--------|
| 01 | [Migrate tree_fusion to leucaena binary + GeoJSON masks](01-migrate-tree-fusion-to-leucaena-geojson.md) | built |
| 02 | [Docker WSL CUDA](02-docker-wsl-cuda.md) | built + deployed |
| 03 | [Tile-based patches pipeline](03-tile-based-patches-pipeline.md) | built |
| 04 | [Tile-based pipeline part 2 (predict-tiles, HDF5/Zarr, LiDAR)](04-tile-based-part2-predict-scale-lidar.md) | partial (3C done) |

## Conventions

- One Markdown file per plan, prefixed `NN-` for ordering.
- Top of the file: `# Title`, then a small block with `Status`, `Owner`,
  `Last update` (YYYY-MM-DD).
- Sections that are useful: `Why`, `Decisions`, `Files touched`,
  `How to run`, `Risks`, `Out of scope` (becomes the next plan), `Todos`.
- When a plan is fully implemented, update its `Status` to `built` and
  add a short `Outcome` paragraph at the bottom.
- Do not delete plans when they are done; future-you uses them as a log.
