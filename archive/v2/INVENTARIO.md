# Inventario WISE-RONIN V2

- Repositorio: https://github.com/orbitalronin94/WISE-RONIN
- Rama inspeccionada: `main`
- Commit inspeccionado: `a0a0c55fec9f924e8e47d5ddf517d2aa24ce28f1`
- Archivos rastreados: 54
- Inventario obtenido del árbol Git remoto. Los tamaños y SHA son exactos; el recuento de líneas de archivos grandes no se ha calculado en esta pasada.
- Dependencias observadas en el trainer: `torch`, `tokenizers`; los scripts auxiliares requieren inspección individual para cerrar el inventario de imports.
- No se encontraron `requirements.txt` ni `pyproject.toml` en la raíz V2.

| Ruta original | Bytes | Blob SHA | Función |
|---|---:|---|---|
| `.github/workflows/extract-best-holdout.yml` | 5,618 | `0ad7246686f65f2249aa7bd418575113bcba9e0b` | Workflow GitHub Actions V2 |
| `.github/workflows/validate-expanded-corpus.yml` | 2,802 | `4c64cf0ac1ab2603ccfdb349eb8f7a1841ffabdb` | Workflow GitHub Actions V2 |
| `.github/workflows/validate-wise-ronin-capability.yml` | 2,045 | `4b16ad4875301a61238ce735487fe7f7f1a9fb58` | Workflow GitHub Actions V2 |
| `.github/workflows/wise-ronin-balanced-80k-validation.yml` | 1,646 | `f6208b2a8fb991b8e37f21f2f51c4f43479d273f` | Workflow GitHub Actions V2 |
| `.github/workflows/wise-ronin-balanced-80k.yml` | 4,625 | `46d6b9f45efc6ee3ac7603e4b6fed5e87d56b623` | Workflow GitHub Actions V2 |
| `.github/workflows/wise-ronin-capability-30k.yml` | 8,195 | `8859f71084fc916a9fc7af952a0002162e877fcc` | Workflow GitHub Actions V2 |
| `.github/workflows/wise-ronin-correction-sft.yml` | 6,558 | `aa148c8e4240d0d73634275c9dbb79c08e85632b` | Workflow GitHub Actions V2 |
| `.github/workflows/wise-ronin-correction-validation.yml` | 1,329 | `113875485e9502027fdd1b68c3f6d58e8cc04cd1` | Workflow GitHub Actions V2 |
| `.github/workflows/wise-ronin-dataset-audit.yml` | 4,789 | `d85604b69b111b1292cc6320c30d4c9eca7934d6` | Workflow GitHub Actions V2 |
| `.github/workflows/wise-ronin-expand-dataset.yml` | 3,662 | `ab5d8e888a34d58223952dc9d17955fa06ac9390` | Workflow GitHub Actions V2 |
| `.github/workflows/wise-ronin-expanded-50k.yml` | 5,852 | `262c6088654763c905e31a6aa1af6df0dd42f058` | Workflow GitHub Actions V2 |
| `.github/workflows/wise-ronin-first-10k.yml` | 5,185 | `9be6b6969b926cdf0ea6286e0880778ce843641f` | Workflow GitHub Actions V2 |
| `.github/workflows/wise-ronin-frozen-30k-evaluation.yml` | 3,597 | `8c700f055d700bb08590c6165dd465c7e98a856a` | Workflow GitHub Actions V2 |
| `.github/workflows/wise-ronin-holdout-70k.yml` | 3,414 | `d71a14854c7a8469e0c480e7745994ea2214780c` | Workflow GitHub Actions V2 |
| `.gitignore` | 55 | `0c0d5837ef73dc18ab1be87bfee62a9c326bc1d0` | Reglas de exclusión Git V2 |
| `README.md` | 7,398 | `a4105ca5589615a327212996a0dc2db4ce1efe85` | Documentación principal V2 |
| `Train-Wise-Ronin (1).py` | 264 | `7ea830428d9b2a46be056cf718653059f07a792c` | Script Python V2; revisar imports para dependencias exactas |
| `audit_wise_ronin_dataset.py` | 7,027 | `e8a0c9b9ca893f753e1b2a99c73343e7fe3ded08` | Script Python V2; revisar imports para dependencias exactas |
| `build_balanced_80k.py` | 8,341 | `80a9d451d7d80e9378aa79441c5165df92617d70` | Script Python V2; revisar imports para dependencias exactas |
| `build_wise_ronin_correction.py` | 9,665 | `c8903a98856ce1eb4bb9effac8adcf732706f2ff` | Script Python V2; revisar imports para dependencias exactas |
| `build_wise_ronin_dataset.py` | 9,340 | `c6adf88934a3c5e1940d8e20da92ad16fff6e9f9` | Script Python V2; revisar imports para dependencias exactas |
| `data/capability/README.md` | 1,409 | `7c03a57d4ae87c84ce1c1ff8bb439b8bed39d2a1` | Documentación V2 |
| `data/capability/wise_ronin_capability_all.jsonl` | 4,204,177 | `1820aba0fc9fd735c7446de7ca638ddac3b57704` | Dataset / holdout JSONL V2 |
| `data/capability/wise_ronin_capability_holdout.jsonl` | 206,766 | `e2f1264928a1cebbcc2fc6dbe971fb458701b513` | Dataset / holdout JSONL V2 |
| `data/capability/wise_ronin_capability_report.json` | 1,734 | `19e94c2253d4515b0fa86de9cc8f9007b6493236` | Manifiesto o informe de datos V2 |
| `data/capability/wise_ronin_capability_train.jsonl` | 3,786,858 | `d1af2973b199813c7d9c337a492b09a7b4cffbcd` | Dataset / holdout JSONL V2 |
| `data/capability/wise_ronin_capability_validation.jsonl` | 210,553 | `c660012ee58422b542f9becc108b63d525970fbc` | Dataset / holdout JSONL V2 |
| `data/capability/wise_ronin_identity_manifest.json` | 927 | `5cc85ec0c61dbc8d8b9c0010e03da5a598e19672` | Manifiesto o informe de datos V2 |
| `data/curated/README.md` | 2,909 | `fca0b264c7c0d8916af3a462144bf4caccaba6e4` | Documentación V2 |
| `data/curated/wise_ronin_antisycophancy.jsonl` | 5,419 | `dfa681ba2fe7c30969741d82968e3d5624cdb4ae` | Dataset / holdout JSONL V2 |
| `data/curated/wise_ronin_dataset_report.json` | 6,237 | `d23717422b0619cb8f1dc4a72d318ececeed4294` | Manifiesto o informe de datos V2 |
| `data/curated/wise_ronin_factual_clean.jsonl` | 8,245 | `504b8fd45cfefc6960a846abd8d6e237f86aaa36` | Dataset / holdout JSONL V2 |
| `data/curated/wise_ronin_hybrid_controlled.jsonl` | 74,495 | `18dc0b6a2bb185640bad77c933e9229cc00a6116` | Dataset / holdout JSONL V2 |
| `data/curated/wise_ronin_koan_controlled.jsonl` | 14,390 | `b40663137936564918b99ac3dfd68693aaad6ffb` | Dataset / holdout JSONL V2 |
| `data/curated/wise_ronin_preferences.jsonl` | 9,595 | `0579b0414aaf75349d84cc69d02cf4f78d7b6407` | Dataset / holdout JSONL V2 |
| `data/curated/wise_ronin_sft_balanced.jsonl` | 110,794 | `5f3bf9f5ed1570b05bebd0085deb8e0f09cc18f7` | Dataset / holdout JSONL V2 |
| `data/expanded/README.md` | 2,629 | `344e530300b4506d352412bd387ede123be64be9` | Documentación V2 |
| `data/expanded/wise_ronin_expanded.jsonl` | 60,501,292 | `027a73ea7266c5d449349d809f0f39cc1f4d5b07` | Dataset / holdout JSONL V2 |
| `data/expanded/wise_ronin_expanded_report.json` | 687 | `0a4cb4316f8a33d71cfbe24dd57b34450e6a2366` | Manifiesto o informe de datos V2 |
| `data/seeds/gen_2mb_v2.py.zlib.b64` | 6,964 | `03027a26aa2bf80f821cc19965c676753d7796b4` | Artefacto de semilla comprimido V2 |
| `dataset_enano_70mb_parte1.jsonl` | 25,165,782 | `1b00603448b1ecf8cead1caa2af759d40d3765a1` | Dataset / holdout JSONL V2 |
| `dataset_enano_70mb_parte2.jsonl` | 25,165,419 | `92ebcd5777f1aca221fd8ee53fdd1a404788d057` | Dataset / holdout JSONL V2 |
| `dataset_enano_70mb_parte3.jsonl` | 23,069,160 | `5f380e3afb958898d6886b7e5ab33f0e797a9c01` | Dataset / holdout JSONL V2 |
| `docs/CORRECTION_BEFORE_30K.md` | 2,691 | `8c1184185d01f2a205ce53a9885ddea77a731b43` | Documentación V2 |
| `docs/model-card-40k.md` | 6,631 | `dc6f95e42b4f6f9a6fc7bb3a6efec92f323f39b4` | Documentación V2 |
| `evaluate_wise_ronin.py` | 9,786 | `654c3b59403791d28406b2b5a1d5b303fda3b89b` | Script Python V2; revisar imports para dependencias exactas |
| `evaluate_wise_ronin_correction.py` | 7,401 | `47ba06996eb967ee52820fca3252d7b7b55a56ee` | Script Python V2; revisar imports para dependencias exactas |
| `evaluate_wise_ronin_sampling.py` | 6,026 | `e48a8813b296dc9639621f11b5a29ef31f0e8c23` | Script Python V2; revisar imports para dependencias exactas |
| `generate_wise_ronin_capability.py` | 34,664 | `c38f4240f51f6952079f263f94dc76b9e0261675` | Script Python V2; revisar imports para dependencias exactas |
| `generate_wise_ronin_expanded.py` | 28,825 | `f059d8a97419e0cf17cf332a8eefdc248e62aafb` | Script Python V2; revisar imports para dependencias exactas |
| `holdout_wise_ronin_500.jsonl` | 158,935 | `2b771ca6760fddabaa04117d719b506d2ff9568e` | Dataset / holdout JSONL V2 |
| `train_wise_ronin.py` | 18,488 | `3a8699df81c8747633316c2f2254080d819568ff` | Script Python V2; revisar imports para dependencias exactas |
| `validate_wise_ronin_correction.py` | 2,061 | `6c8758c890f833aa3f2ca9b04776c4d2f6253647` | Script Python V2; revisar imports para dependencias exactas |
| `wise-ronin_v2_pesos.npz` | 22,411,578 | `f725841bee32c1b2c75e68dc9773785eecbdaacd` | Pesos del modelo V2 |

## Hallazgos de integridad

- Existe `holdout_wise_ronin_500.jsonl`, con 500 registros e IDs únicos H-001…H-500.
- El holdout contiene 55 prompts distintos; hay variantes repetidas. Se conserva sin modificar.
- El README V2 describe 4.267.520 parámetros, vocabulario BPE real de 1.297 tokens y arquitectura decoder con RoPE/GQA/SwiGLU/RMSNorm.
- El código del trainer declara `vocab=6500`, `dim=256`, `layers=6`, `q_heads=8`, `kv_heads=2`, `hidden=640`; esta diferencia con el README se debe conservar como hallazgo, no corregir en el archivo archivado.
- Hay tres fragmentos JSONL de corpus de ~70 MB, un corpus expandido de ~60,5 MB, datasets curados y de capacidad, workflows y pesos `wise-ronin_v2_pesos.npz`.
- No se ha inspeccionado ni validado aún el solapamiento entre train y holdout.

## Limitaciones del inventario

- Los ficheros V2 se archivan como blobs Git idénticos, sin reserialización.
- Las líneas de fuentes y workflows no se han contado individualmente en esta pasada; los tamaños y hashes sí quedan registrados.
- No se ha ejecutado ningún entrenamiento V2 ni se ha alterado el holdout.