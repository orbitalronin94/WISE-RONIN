# WISE-RONIN

**Modelo de lenguaje pequeño, entrenado y evaluado en español, con prioridad en coherencia, estabilidad y coste real de CPU.**

WISE-RONIN es la línea activa del proyecto. La versión RONIN-10M anterior queda archivada como experimento abandonado: aumentar parámetros y pasos no produjo una mejora útil en generación. En los holdouts de 60k–78k aparecieron bucles, repetición, fugas de plantillas del corpus, fragmentos ingleses irrelevantes y respuestas absurdas o vacías. La decisión fue dejar de perseguir esa rama y concentrar el trabajo en un modelo más pequeño y una evaluación más rigurosa.

> **Estado a 22 de noviembre de 2026:** corpus español combinado generado y auditado localmente por GitHub Actions, pero el artefacto final todavía no se ha publicado porque falló una aserción de validación del manifiesto. No se ha iniciado un entrenamiento con esa mezcla. La arquitectura y los datos deben considerarse experimentales hasta completar la siguiente evaluación.

## Línea vigente

El checkpoint de referencia conocido de WISE-RONIN tiene **4.267.520 parámetros** y utiliza un tokenizer de **1.297 tokens**. La configuración registrada es:

| Parámetro | Valor |
|---|---:|
| Dimensión | 256 |
| Capas | 6 |
| Cabezas Q / KV | 8 / 2 |
| Dimensión por cabeza | 32 |
| Dimensión feed-forward | 640 |
| Contexto máximo | 1.024 |
| Vocabulario | 1.297 |
| Parámetros | 4.267.520 |

El trabajo anterior llegó a una continuación desde 40k hacia 50k pasos. No se debe presentar ese tramo como una campaña final validada: el checkpoint de 40k es el último punto de referencia documentado con evaluación comparativa disponible. La pérdida de entrenamiento, por sí sola, no demuestra capacidad factual ni razonamiento.

### Por qué se abandonó RONIN-10M

La rama de aproximadamente diez millones de parámetros alcanzó 60k–78k pasos, pero los holdouts revelaron degradación generativa: repetición y bucles, plantillas filtradas desde los datos, texto inglés fuera de contexto, respuestas absurdas con tono seguro y salidas vacías. En la evaluación de 78k se observaron además cinco respuestas «ni puta idea». Más parámetros y más entrenamiento no resolvieron esos defectos.

WISE-RONIN se mantuvo como línea activa porque, con menos de la mitad de parámetros efectivos, mostró respuestas más coherentes. Esto no significa que ya sea un modelo competente: las pruebas disponibles también mostraron debilidad factual y aritmética. La decisión es comparativa y provisional, no una declaración de superioridad general.

## Corpus español combinado

La preparación se divide en dos partes: fuentes externas con licencia permitida y material histórico propio. El corpus externo se preparó en PR #64 y la extracción/mezcla con el archivo histórico se implementó en PR #65.

- [PR #64 — corpus español licenciado](https://github.com/orbitalronin94/WISE-RONIN/pull/64)
- [PR #65 — extracción de hechos históricos y mezcla](https://github.com/orbitalronin94/WISE-RONIN/pull/65)
- [Workflow de preparación inicial](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45007651092)
- [Workflow de mezcla](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45008699964)

### Resultado de la extracción

El workflow de mezcla procesó el archivo histórico `archive/v2/data/expanded/wise_ronin_expanded.jsonl`:

| Métrica | Resultado |
|---|---:|
| Filas históricas examinadas | 147.627 |
| Filas clasificadas como español | 147.627 |
| Grupos independientes por `fact_id` | 10.341 |
| Ejemplos históricos incorporados | 10.341 |
| Train combinado | 80.446 |
| Validation | 1.413 |
| Holdout | 2.431 |

Se conservó una variante canónica por grupo `fact_id`, en lugar de incorporar todas las paráfrasis de cada hecho. La respuesta se construyó desde el campo canónico `hecho`, evitando entrenar el modelo con cientos de reformulaciones estilísticas casi idénticas. Los ejemplos históricos se añadieron únicamente a train.

Validation y holdout se copiaron sin modificación desde el corpus base. El holdout contiene 1.931 ejemplos SIQA-es y 500 OpenBookQA-es. No se utiliza para entrenamiento.

### Fuentes externas

| Fuente | Ejemplos en la mezcla |
|---|---:|
| `somosnlp/instruct-legal-refugiados-es` | 9.103 |
| `OysterCoreAI/SFT-General-Spanish-50K` | 24.484 |
| `gplsi/uji_parallel_va_es` | 19.569 |
| `BSC-LT/ALIA-2606-SFT` | 16.949 |
| Archivo histórico propio | 10.341 |

La política de admisión externa permite Apache-2.0 y CC-BY-4.0; se excluyen fuentes NC, ND, CC-BY-SA, desconocidas o con componentes de licencia no verificables. El material histórico se identifica como contenido generado por el proyecto. Las cifras de esta tabla corresponden al manifiesto producido por la ejecución de mezcla, no a estimaciones.

### Estado del artefacto: generado, pero no publicado

La ejecución [45008699964](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45008699964) completó la extracción y produjo los recuentos y hashes siguientes:

- Train: `bea761b56684f6e93d030632e1c464752f45a1dc03067c43692632705a4cf515`
- Validation: `2efacb8236c1041534199709791e63469ce36f64335aef1b60c8ff0c57a986b6`
- Holdout: `e21ae7e292d603f1b8073d8114572c92fb0c6448bfaed9066413ddce09e5a359`

La subida del artefacto se omitió porque el paso de validación intentó leer la clave inexistente `legacy_rows_non_spanish`; el manifiesto sí contiene `legacy_rows_spanish` y `legacy_rows_seen`. Es un defecto en la comprobación del workflow posterior a la extracción, no un fallo del procesamiento del corpus. Hasta que la validación se corrija y el artefacto se publique, estos hashes documentan la salida observada en los logs, pero no sustituyen un artefacto descargable y verificable.

Este conjunto es principalmente **SFT, QA e instrucciones**, junto con texto paralelo y hechos/matemática sintéticos históricos. No debe describirse como un corpus general de preentrenamiento web ni como una muestra representativa de todo el español.

## Lecciones de evaluación

Las evaluaciones anteriores mostraron que una pérdida baja puede coexistir con respuestas incorrectas, repetición o incapacidad para terminar una respuesta. Por ello, ningún resultado se juzga solo por loss.

Se deben registrar como mínimo:

- EOS y longitud de salida, incluidas respuestas vacías o truncadas.
- Exactitud en aritmética y razonamiento, con conjuntos de prueba separados.
- Exactitud factual y resistencia a premisas falsas.
- Repetición de n-gramas, bucles y diversidad de respuestas.
- Fugas de plantillas o memorización de formulaciones del entrenamiento.
- Proporción de respuestas en español y fragmentos fuera de contexto.
- Tiempo real, pasos/s, tokens/s y memoria máxima en CPU.
- Hashes de corpus, tokenizer, checkpoint, configuración y código evaluador.

Los holdouts deben permanecer congelados y fuera del entrenamiento. Toda comparación con Transformer debe usar los mismos datos, tokenizer, semilla, presupuesto de tokens y condiciones de evaluación, dejando explícitas las diferencias arquitectónicas.

## Ingeniería y rendimiento CPU

El objetivo operativo es entrenar en runners CPU de GitHub Actions sin sacrificar las comprobaciones que permiten confiar en el resultado. Los benchmarks cortos sirven para seleccionar candidatos, no para prometer aceleraciones en entrenamientos completos.

La configuración de hilos debe fijarse mediante mediciones reproducibles en el runner utilizado. No se debe asumir que más hilos, menos comprobaciones o menos pérdidas auxiliares equivalen a más rendimiento útil. Las optimizaciones que eliminen observabilidad o cambien el objetivo de aprendizaje requieren una comparación explícita y no se adoptan solo por aumentar pasos/s.

Los workflows de benchmark y los de entrenamiento tienen propósitos distintos. Integrar documentación o preparar datos no debe iniciar accidentalmente un entrenamiento largo.

## Próximos hitos

1. Corregir la aserción de validación del manifiesto y publicar el artefacto combinado, sin volver a descargar ni reconstruir las fuentes ya procesadas.
2. Verificar los tres ficheros, sus hashes, el manifiesto, la separación train/validation/holdout y la procedencia.
3. Auditar el contrato de secuencias: reservar espacio para EOS después de truncar, y mantener idéntico tokenizer/configuración entre entrenamiento y evaluación.
4. Ejecutar primero una prueba corta y recuperable; inspeccionar generaciones y holdout antes de autorizar una campaña larga.
5. Comparar cambios de arquitectura y rendimiento mediante ablaciones, sin mezclar varias hipótesis en un mismo experimento.

## Documentación y código

- [Índice de documentación](docs/README.md)
- [Procedencia y preparación de datos](data/README.md)
- [Contrato del corpus propio en español](data/owner/README_ES.md)
- [Workflow de entrenamiento](.github/workflows/train.yml)
- [Workflow del piloto comparativo](.github/workflows/pilot.yml)
- [Workflow de preparación de datos](.github/workflows/prepare-data.yml)
- [Workflow de mezcla de corpus](.github/workflows/merge-spanish-corpora.yml)

## Reproducir el smoke test

Desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
python -m train.train --dummy-m1 --steps 10 --phase 1
python -m train.train --dummy-m1 --resume checkpoints/step_10.pt --steps 10 --phase 1
```

El modo dummy solo comprueba forward, optimización, checkpoint y reanudación. No utiliza el corpus real y no demuestra calidad generativa.

---

*WISE-RONIN no se valida por el tamaño de sus pesos ni por una curva de pérdida atractiva. Se valida cuando mantiene respuestas coherentes, termina correctamente y resiste pruebas externas congeladas bajo condiciones reproducibles.*
