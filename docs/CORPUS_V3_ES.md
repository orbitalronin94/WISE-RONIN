# Lumen-Ronin Spanish Corpus v3

**Build:** 23 November 2026  
**Branch:** \`corpus/lumen-ronin-spanish-v3\`  
**Commit:** \`7e40727890f21c55197011273d6230b65625dfb5\`  
**Workflow:** [45132496748](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45132496748)  
**Artifact:** [lumen-ronin-spanish-corpus-v3-45132496748](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45132496748/artifacts/14797958579)

## Estado

El artefacto se construyó, congeló y validó correctamente. **No se ha iniciado entrenamiento con v3.**

El manifiesto conserva \`training_authorized=false\`: falta completar la deduplicación exacta entre nuevas fuentes y el train v2, la auditoría global de near-duplicates y los controles globales de contaminación. La literatura importada está admitida únicamente en una partición train *scoped*; su manifiesto de origen declara \`global_g0_pass=false\`. La construcción no convierte esa admisión limitada en autorización global.

## Tamaño y comparación con v2

| Métrica | Corpus v2 | Corpus v3 |
|---|---:|---:|
| Filas train | 9.357.876 | **9.933.007** |
| Incremento neto | — | **575.558 (+6,15 %)** |
| Validación | 1.413 | 1.413 |
| Holdout opción múltiple | 2.431 | 2.431 |
| Vocabulario tokenizer | 4.096 | 4.096 |

Validación y holdout se copiaron byte por byte desde v2. El tokenizer también se conserva sin cambios. No se introdujeron nuevas filas en los splits de evaluación.

## Incorporaciones

### Literatura española

- 608 documentos en la partición train admitida de \`RONIN-IA/literature-es-v1\`.
- Textos de dominio público procedentes de CLiGS ConSSA, CLiGS Textbox y ELTeC-spa.
- 370.228 fragmentos de hasta 126 tokens BPE.
- 7.261 fragmentos exactos duplicados descartados frente a los splits protegidos o a incorporaciones previas.
- Solo se importó train; validation/test no se descargaron ni se incorporaron.
- La fuente está admitida *scoped*; su propio manifiesto indica que el gate G-0 global no ha pasado.

### M-Personas en castellano

- 36.094 conversaciones españolas.
- Fuente: \`BSC-LT/m-personas\`, split \`es\`, revisión \`6fab205\`.
- Licencia declarada en la revisión fijada: Apache-2.0.
- Se conserva el último turno del asistente por conversación; se excluye \`source_text\`.
- La descripción de persona se trata como contexto de estilo/perspectiva, no como sustitución de la identidad del modelo.
- El dataset declara sesgos demográficos y ocupacionales conocidos; revisar muestras antes de entrenar.

### Personalidad factual verificable

- 50.000 ejemplos generados a partir de 40 fichas factuales.
- Se conserva la atribución y licencia por registro.
- No son 50.000 hechos independientes: son combinaciones deterministas de fichas, preguntas, contextos y estilos.
- Se añade la identidad Lumen-Ronin y un estilo erudito, sobrio e incisivo; la ironía no puede alterar los hechos ni la incertidumbre.

### Currículos conductuales y léxicos en español

- 24 datasets fuente de 50.000 registros cada uno (1,2 millones de registros fuente).
- Se seleccionaron hasta 5.000 por dataset, repartidos entre diez casos semilla para evitar concentrar la muestra en un único patrón.
- Tras deduplicación exacta entre las nuevas incorporaciones, quedaron **119.026 registros**.
- Incluyen no repetición, premisas falsas, admitir cuando el usuario tiene razón, contradicción, resistencia a halagos, brevedad, koan, diagnóstico de errores, revisión de arquitectura, detección de buzzwords, verificación factual y otros patrones conductuales/lexicales.
- Cada ejemplo se condiciona explícitamente a la identidad Lumen-Ronin.

### Adaptación de autoidentidad heredada

El train v2 contenía 427 ejemplos del componente \`oasst2-selfidentity-rephrase\` de ALIA. No se copiaron sin cambios:

- 126 respuestas se reescribieron para describir Lumen-Ronin.
- 64 ejemplos genéricos sin afirmaciones de identidad ALIA se conservaron como SFT general.
- 163 ejemplos con afirmaciones de identidad o procedencia ajenas se excluyeron.
- 74 ejemplos adaptados redundantes se descartaron por duplicación.
- Se añadieron además 20 ejemplos canónicos de identidad, arquitectura, límites, incertidumbre, glucosa, dolor, placer, homeostasis, replay y engramas.

La adaptación deja claro que dolor y placer son señales computacionales, no experiencias subjetivas; no atribuye conciencia, credenciales ni financiación institucional.

## Identidad estable aplicada a los ejemplos

Los ejemplos de personalidad, conversación condicionada por persona, conducta y léxico incluyen una instrucción estable:

- El asistente sigue siendo **Lumen-Ronin**.
- El perfil de persona modula el registro o perspectiva, pero no cambia la identidad del modelo.
- Glucosa, dolor y placer describen mecanismos computacionales.
- No se deben inventar experiencias, capacidades, credenciales, recuerdos ni resultados.
- Debe reconocerse la incertidumbre y corregirse una premisa falsa cuando corresponda.

## Integridad

| Archivo | SHA-256 |
|---|---|
| \`train.jsonl\` | \`87595c3217c3b8166eadaf24309cfe4753a6da4e7c1d7c91bfd5322cae114811\` |
| \`validation.jsonl\` | \`2efacb8236c1041534199709791e63469ce36f64335aef1b60c8ff0c57a986b6\` |
| \`holdout_multiple_choice.jsonl\` | \`e21ae7e292d603f1b8073d8114572c92fb0c6448bfaed9066413ddce09e5a359\` |
| \`tokenizer.json\` | \`d7869c11809a28085d5156638086c4833a1dcbacfda4a60b69ec71e26ed3fefe\` |

El SHA-256 del train v2 se verificó durante la copia. Los hashes de validación, holdout y tokenizer coinciden con v2.

## Contenido del artefacto

- \`train.jsonl\`
- \`validation.jsonl\`
- \`holdout_multiple_choice.jsonl\`
- \`tokenizer.json\`
- \`manifest.json\`

Tamaño del artefacto: 1.638.961.161 bytes. Retención de Actions: 90 días desde el 23-11-2026.

## Siguiente gate

Antes de usar v3 en entrenamiento:

1. Deduplicación exacta de las nuevas fuentes contra el train completo v2.
2. Auditoría de near-duplicates y solapamientos globales.
3. Revisión de muestras de M-Personas y de los currículos sintéticos.
4. Confirmación final de derechos y procedencia por fuente.
5. Congelar el manifiesto de autorización.

Hasta superar esos controles, v3 es un **artefacto de construcción validado**, no un corpus declarado globalmente autorizado para entrenamiento. Cuando se entrene, Lumen y el Transformer de control deberán recibir exactamente el mismo corpus y tokenizer.

## Cierre de versión v3 — auditoría 23-11-2026

El snapshot de construcción anterior queda identificado como **Lumen-Ronin Spanish Corpus v3.0.0**. No se modificarán sus datos bajo esta versión. Cualquier limpieza posterior generará una versión nueva (v3.1 o v4), con nuevo manifiesto, hash y artefacto.

- Auditoría: [run 45142717502](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45142717502)
- Informe: [artifact 14805384712](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45142717502/artifacts/14805384712)
- Train rows: 9.933.007.
- Incremento frente a v2: 575.558 filas (+6,15 %).
- SHA-256 train: `87595c3217c3b8166eadaf24309cfe4753a6da4e7c1d7c91bfd5322cae114811`.
- Artefacto comprimido: 1.638.961.161 bytes (1,64 GB decimal; 1,53 GiB).
- Directorio extraído medido por el runner de auditoría: aproximadamente 5,4 GB.
- Validación: 1.413 filas, idéntica a v2.
- Holdout MC: 2.431 filas, idéntico a v2.
- Tokenizer: vocabulario 4.096, idéntico a v2.

### Resultado de controles automáticos

- Cero duplicados exactos de pares prompt/completion frente a v2.
- Cero duplicados exactos internos entre las nuevas incorporaciones.
- Cero coincidencias exactas con validation/holdout.
- 63 completions duplicadas frente a v2 (longitud >=40), pendientes de decidir si son respuestas genéricas legítimas o redundancia.
- 19.740 near-duplicates confirmados por el filtro de alta precisión.
- 157.874 consultas superaron el límite de candidatos del detector; no se consideran aprobadas por ausencia de hallazgo.
- 41 posibles marcadores sensibles detectados automáticamente; son alertas para revisión, no filtraciones confirmadas.
- La trazabilidad individual de los fragmentos literarios hasta documento y hash fuente debe restaurarse.
- M-Personas requiere revisión de muestras y sesgos; las licencias y procedencia deben conservarse por fuente.

**Estado de uso:** v3 queda congelado como snapshot reproducible de datos, no como corpus globalmente certificado ni autorizado para entrenamiento. `training_authorized=false` se mantiene deliberadamente. El conteo exacto de tokens BPE del conjunto completo no figura en el manifiesto ni en la auditoría; no se sustituye por una estimación. Se calculará mediante un pase completo con el tokenizer congelado antes de publicar esa cifra.
