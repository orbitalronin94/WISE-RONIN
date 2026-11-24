# Auditoría y decisiones de admisión del corpus v3.2

**Estado: criba documental inicial. No equivale a lectura completa de los registros descargados.**  
**Entrenamiento autorizado: no.**

Esta primera decisión se basa en los metadatos de la ficha de fuentes, el solapamiento ya conocido con v3.1 y los riesgos de licencia, privacidad, particiones de evaluación y objetivo de tarea. No se afirma que se hayan leído todas las filas de los 20 datasets: esa auditoría de contenido requiere descargar las revisiones fijadas y ejecutar el pipeline reproducible.

## Decisiones iniciales

| Fuente | Decisión | Tratamiento |
|---|---|---|
| SFT-General-Spanish-50K | Mantener para auditoría | Revisar calidad sintética, repeticiones y respuestas |
| mPersonas | Mantener, solo español | Deduplicar perfiles y diálogos por separado |
| Personas_Spanish-40k | Mantener muestra estrecha | No tratar como población representativa |
| StrataSynth Spain | Filtrado | Excluir conditioning psicométrico, demográfico sensible y attachment |
| Big Five traits | **Descartar** | Redundancia conceptual y riesgo de enseñar etiquetas psicométricas |
| LingComp_QA | **No reimportar** | 741 de 803 ya entraron en v3.1; revisar solo el remanente |
| Lenguaje claro | **No reimportar** | 55 filas ya entraron en v3.1; auditar solo el remanente |
| Instruct Legal Refugiados | Mantener train pendiente | Jurisdicción, fecha, derechos y split |
| GovEasy | Mantener pendiente | Procedimientos oficiales vigentes, fecha y URL |
| ESQAD | Cuarentena de particiones | Excluir benchmark/test y separar datos generados |
| TDX | Mantener pendiente | Derechos por documento, privacidad, boilerplate y duplicados |
| BDH-Books | Cuarentena | Derechos por obra, OCR y solapamiento con Spanish-PD-Books |
| Spanish-PD-Books | Cuarentena | Derechos por edición y solapamiento con BDH |
| Common Corpus | **Descartar ingestión masiva** | Solo reconsiderar subconjunto documental con derechos trazables |
| MS MARCO ES | Solo retrieval | No convertir triples en SFT generativa |
| Enfermedades Wiki | **Descartar por ahora** | Licencia MIT del empaquetado no acredita derechos de cada artículo |
| PharmaCoNER | Mantener train pendiente | NER de fármacos, etiquetas y atribución intactas |
| MEDDOCAN | **Descartar por ahora** | Riesgo elevado de privacidad/reidentificación y derechos upstream |
| DisTEMIST | Mantener train pendiente | Extracción de menciones; no diálogo médico |
| BARR2 | **Descartar por ahora** | Falta fijar fuente canónica, revisión y derechos |

Las decisiones estructuradas están en `data/corpus_v3_2_dispositions.json`.

## Duplicación contra v3.1

- LingComp_QA: no volver a importar las 741 filas ya admitidas.
- Lenguaje claro: no volver a importar las 55 filas ya admitidas.
- Libros BDH y Spanish-PD-Books: tratar como un mismo conjunto de deduplicación; no sumar tamaños publicados.
- Common Corpus: no admitir el agregado completo; solapamiento probable con libros, tesis y otras colecciones.
- Las fuentes de personalidad se comparan también por perfil normalizado, no solo por fila completa.

## Auditoría de contenido pendiente

Para declarar una fuente leída y aprobada se exige:
1. Fijar commit/revisión inmutable y guardar SHA-256.
2. Descargar todos los registros del split candidato, o documentar expresamente por qué una fuente masiva queda excluida sin ingestión.
3. Contar filas y tokens españoles reales, no reutilizar cifras promocionales.
4. Inspeccionar esquema, todos los splits y una muestra estratificada de al menos 100 registros; en fuentes pequeñas, todos.
5. Buscar PII, identificadores clínicos, datos sensibles, OCR roto, texto vacío, spam, boilerplate y respuestas de baja calidad.
6. Deduplicar exacto y aproximado contra v3.1 y entre fuentes.
7. Emitir una decisión por fila: `accept`, `reject_duplicate`, `reject_license`, `reject_privacy`, `reject_quality` o `quarantine_review`.
8. Guardar motivos, fuente, revisión y hash para cada decisión.

No se descartarán registros por una simple palabra clave de personalidad: se evalúa su contenido. Los perfiles sintéticos útiles para estilo pueden conservarse sin etiquetas psicométricas ni atributos sensibles.

## Identidad de LUMEN

La identidad canónica se gestiona por separado en `data/identity/`. No se mezcla con Big Five, perfiles de persona ni personalidad sintética. El conjunto de identidad tiene su propio holdout y no autoriza entrenamiento hasta validarlo.

**Gate final:** solo se construye v3.2 después de terminar la auditoría, congelar manifiesto y cache, y comprobar que LUMEN y Transformer reciben exactamente los mismos datos.