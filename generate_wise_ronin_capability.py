#!/usr/bin/env python3
"""Build a leakage-aware, low-redundancy WISE-RONIN capability corpus.

The target is independent tasks/anchors, not an arbitrary row count. The builder
keeps one canonical example per task, groups variants by group_id, checks the
existing expanded corpus for overlapping prompts/answers, and splits by group.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, unicodedata
from collections import Counter
from pathlib import Path

SYSTEM = ("Eres WISE-RONIN. Responde primero a lo que se pregunta. "
          "Sé factual, conciso y honesto. No inventes; si no sabes, dilo. "
          "El estilo nunca debe sustituir a la respuesta.")
IDENTITY = {
    "name": "WISE-RONIN",
    "type": "modelo experimental de lenguaje",
    "parameters": 4267520,
    "architecture": {"dimension": 256, "layers": 6, "query_heads": 8,
                     "key_value_heads": 2, "head_dimension": 32,
                     "feed_forward_dimension": 640, "context_tokens": 1024,
                     "attention": "GQA", "position_encoding": "RoPE",
                     "normalization": "RMSNorm", "activation": "SwiGLU",
                     "embeddings": "tied"},
    "tokenizer": {"type": "BPE ByteLevel", "vocabulary_size": 1297},
    "checkpoint_reference": "step 50000",
    "known_limitations": [
        "No dispone de acceso a internet durante la inferencia.",
        "Puede cometer errores factuales.",
        "No es un sistema de razonamiento largo garantizado.",
        "No debe inventar cifras, fechas ni citas.",
        "Si carece de información suficiente, debe reconocerlo.",
        "El tono irónico no demuestra que una respuesta sea cierta."
    ]
}
IDENTITY_QA = [
    ("¿Quién eres?", "Soy WISE-RONIN, un modelo experimental de lenguaje."),
    ("¿Qué tipo de modelo eres?", "Soy un modelo experimental de lenguaje de pequeño tamaño."),
    ("¿Cuántos parámetros tienes?", "Tengo 4.267.520 parámetros."),
    ("¿Cuántas capas tiene tu arquitectura?", "Tengo 6 capas."),
    ("¿Cuál es la dimensión del modelo?", "La dimensión interna es 256."),
    ("¿Cuántas cabezas de atención usas?", "Uso 8 cabezas de consulta y 2 cabezas key-value mediante GQA."),
    ("¿Qué mecanismo de atención utilizas?", "Utilizo Grouped Query Attention, o GQA."),
    ("¿Qué codificación posicional utilizas?", "Utilizo RoPE."),
    ("¿Qué normalización utilizas?", "Utilizo RMSNorm."),
    ("¿Qué activación utiliza tu red?", "Utilizo SwiGLU."),
    ("¿Qué dimensión tiene la red feed-forward?", "La dimensión feed-forward es 640."),
    ("¿Cuántos tokens admite tu contexto?", "La configuración del modelo admite hasta 1024 tokens de contexto."),
    ("¿Qué tokenizer tienes?", "Utilizo un tokenizer BPE ByteLevel."),
    ("¿Cuántas entradas tiene tu vocabulario?", "El tokenizer del checkpoint de referencia tiene 1297 tokens."),
    ("¿Tus embeddings están ligados?", "Sí. La matriz de embeddings se comparte con la proyección de salida."),
    ("¿Tienes acceso a internet al responder?", "No. No tengo acceso a internet durante la inferencia."),
    ("¿Eres infalible?", "No. Puedo equivocarme y debo reconocer la incertidumbre."),
    ("¿Eres GPT, Claude o Qwen?", "No. Soy WISE-RONIN, un proyecto independiente."),
    ("¿Qué debes hacer si no sabes una respuesta?", "Debo decir que no lo sé en lugar de inventar una respuesta."),
    ("¿Tu ironía garantiza que tengas razón?", "No. El tono irónico no es evidencia de certeza."),
    ("¿Eres un modelo de razonamiento largo?", "No tengo garantizada la capacidad de razonamiento largo."),
    ("¿Puedes sustituir a un profesional médico, jurídico o financiero?", "No. No sustituyo el criterio de profesionales cualificados."),
    ("¿Qué checkpoint describe esta ficha?", "La ficha técnica corresponde al checkpoint de referencia del paso 50000."),
    ("¿Qué significa GQA?", "Grouped Query Attention: varias cabezas de consulta comparten cabezas key-value."),
    ("¿Qué significa que tus embeddings estén ligados?", "Que la matriz de embeddings de entrada también se usa para proyectar la salida.")
]

FACTS = [
("¿Quién escribió la Odisea?","Homero"),
("¿Quién escribió La Ilíada?","Homero"),
("¿Quién escribió Orgullo y prejuicio?","Jane Austen"),
("¿Quién escribió Jane Eyre?","Charlotte Brontë"),
("¿Quién escribió Cumbres borrascosas?","Emily Brontë"),
("¿Quién escribió El retrato de Dorian Gray?","Oscar Wilde"),
("¿Quién escribió Los viajes de Gulliver?","Jonathan Swift"),
("¿Quién escribió Robinson Crusoe?","Daniel Defoe"),
("¿Quién escribió La metamorfosis?","Franz Kafka"),
("¿Quién escribió El proceso?","Franz Kafka"),
("¿Quién escribió El extranjero?","Albert Camus"),
("¿Quién escribió La peste?","Albert Camus"),
("¿Quién escribió Madame Bovary?","Gustave Flaubert"),
("¿Quién escribió Los miserables?","Victor Hugo"),
("¿Quién escribió Notre-Dame de París?","Victor Hugo"),
("¿Quién escribió La montaña mágica?","Thomas Mann"),
("¿Quién escribió El nombre de la rosa?","Umberto Eco"),
("¿Quién escribió Ensayo sobre la ceguera?","José Saramago"),
("¿Quién escribió La ciudad y los perros?","Mario Vargas Llosa"),
("¿Quién escribió El túnel?","Ernesto Sabato"),
("¿Quién escribió La Regenta?","Leopoldo Alas, Clarín"),
("¿Quién escribió Fortunata y Jacinta?","Benito Pérez Galdós"),
("¿Quién escribió Niebla?","Miguel de Unamuno"),
("¿Quién escribió Platero y yo?","Juan Ramón Jiménez"),
("¿Quién escribió El Aleph?","Jorge Luis Borges"),
("¿Quién escribió La muerte de Artemio Cruz?","Carlos Fuentes"),
("¿Quién escribió La tregua?","Mario Benedetti"),
("¿Quién escribió El ruido y la furia?","William Faulkner"),
("¿Quién escribió Ulises?","James Joyce"),
("¿Quién escribió En busca del tiempo perdido?","Marcel Proust"),
("¿Quién escribió La condición humana?","André Malraux"),
("¿Quién escribió El corazón de las tinieblas?","Joseph Conrad"),
("¿Quién escribió El viejo y el mar?","Ernest Hemingway"),
("¿Quién escribió La llamada de Cthulhu?","H. P. Lovecraft"),
("¿Quién escribió Fundación?","Isaac Asimov"),
("¿Quién escribió Solaris?","Stanisław Lem"),
("¿Quién escribió ¿Sueñan los androides con ovejas eléctricas?","Philip K. Dick"),
("¿Quién escribió Neuromante?","William Gibson"),
("¿Quién escribió Matadero cinco?","Kurt Vonnegut"),
("¿Quién escribió El cuento de la criada?","Margaret Atwood"),
("¿Quién escribió Beloved?","Toni Morrison"),
("¿Quién escribió Persépolis?","Marjane Satrapi"),
("¿Quién compuso El Danubio azul?","Johann Strauss II"),
("¿Quién compuso El lago de los cisnes?","Piotr Ilich Chaikovski"),
("¿Quién compuso Carmen?","Georges Bizet"),
("¿Quién compuso La flauta mágica?","Wolfgang Amadeus Mozart"),
("¿Quién compuso El Mesías?","Georg Friedrich Händel"),
("¿Quién compuso el Bolero?","Maurice Ravel"),
("¿Quién compuso El pájaro de fuego?","Ígor Stravinski"),
("¿Quién pintó El nacimiento de Venus?","Sandro Botticelli"),
("¿Quién pintó La escuela de Atenas?","Rafael"),
("¿Quién pintó La libertad guiando al pueblo?","Eugène Delacroix"),
("¿Quién pintó La persistencia de la memoria?","Salvador Dalí"),
("¿Quién pintó El jardín de las delicias?","El Bosco"),
("¿Quién pintó La joven de la perla?","Johannes Vermeer"),
("¿Quién pintó La ronda de noche?","Rembrandt"),
("¿Quién pintó El almuerzo de los remeros?","Pierre-Auguste Renoir"),
("¿Quién pintó Impresión, sol naciente?","Claude Monet"),
("¿Quién pintó Las señoritas de Aviñón?","Pablo Picasso"),
("¿En qué año comenzó la guerra civil española?","1936"),
("¿En qué año terminó la guerra civil española?","1939"),
("¿En qué año se firmó la Declaración de Independencia de Estados Unidos?","1776"),
("¿En qué año comenzó la Revolución rusa?","1917"),
("¿En qué año se disolvió la Unión Soviética?","1991"),
("¿En qué año se aprobó la Constitución española vigente?","1978"),
("¿En qué año se fundó la Organización de las Naciones Unidas?","1945"),
("¿En qué año se firmó el Tratado de Maastricht?","1992"),
("¿En qué año comenzó la Primera Guerra Mundial?","1914"),
("¿En qué año terminó la Primera Guerra Mundial?","1918"),
("¿En qué año se produjo la caída de Constantinopla?","1453"),
("¿En qué año comenzó la Reforma protestante asociada a Lutero?","1517"),
("¿En qué año se publicó la primera parte del Quijote?","1605"),
("¿En qué año se publicó Frankenstein de Mary Shelley?","1818"),
("¿Cuál es la capital de Eslovaquia?","Bratislava"),
("¿Cuál es la capital de Eslovenia?","Liubliana"),
("¿Cuál es la capital de Estonia?","Tallin"),
("¿Cuál es la capital de Letonia?","Riga"),
("¿Cuál es la capital de Lituania?","Vilna"),
("¿Cuál es la capital de Mongolia?","Ulán Bator"),
("¿Cuál es la capital de Etiopía?","Adís Abeba"),
("¿Cuál es la capital de Senegal?","Dakar"),
("¿Cuál es la capital de Ghana?","Acra"),
("¿Cuál es la capital de Uganda?","Kampala"),
("¿Cuál es la capital de Tanzania?","Dodoma"),
("¿Cuál es la capital de Angola?","Luanda"),
("¿Cuál es la capital de Mozambique?","Maputo"),
("¿Cuál es la capital de Zambia?","Lusaka"),
("¿Cuál es la capital de Zimbabue?","Harare"),
("¿Cuál es la capital de Malasia?","Kuala Lumpur"),
("¿Cuál es la capital de Nepal?","Katmandú"),
("¿Cuál es la capital de Bangladés?","Daca"),
("¿Cuál es la capital de Afganistán?","Kabul"),
("¿Cuál es la capital de Jordania?","Amán"),
("¿Cuál es la capital de Líbano?","Beirut"),
("¿Cuál es la capital de Bolivia?","Sucre, capital constitucional"),
("¿Cuál es la capital ejecutiva de Sudáfrica?","Pretoria"),
("¿Qué río atraviesa Budapest?","El Danubio"),
("¿Qué río atraviesa Lisboa?","El Tajo"),
("¿Qué río atraviesa El Cairo?","El Nilo"),
("¿Qué estrecho separa España de Marruecos?","El estrecho de Gibraltar"),
("¿Qué cordillera separa Francia y España?","Los Pirineos"),
("¿Qué cordillera recorre el oeste de Sudamérica?","Los Andes"),
("¿Qué mar separa la península itálica de los Balcanes?","El mar Adriático"),
("¿Qué mar conecta con el océano Atlántico por el estrecho de Gibraltar?","El mar Mediterráneo"),
("¿Cuál es la fosa oceánica más profunda conocida?","La fosa de las Marianas"),
("¿Cuál es el lago más profundo del mundo?","El lago Baikal"),
("¿Qué país tiene forma de bota en los mapas de Europa?","Italia"),
("¿Qué país alberga la región histórica de Transilvania?","Rumanía"),
("¿Qué país tiene como capital a Reikiavik?","Islandia"),
("¿Qué país tiene como capital a Ulán Bator?","Mongolia"),
("¿Qué país tiene como capital a Adís Abeba?","Etiopía"),
("¿Cuál es la moneda oficial de Japón?","El yen"),
("¿Cuál es la moneda oficial del Reino Unido?","La libra esterlina"),
("¿Cuál es la moneda oficial de Suiza?","El franco suizo"),
("¿Cuál es la moneda oficial de Corea del Sur?","El won surcoreano"),
("¿Cuál es la moneda oficial de India?","La rupia india"),
("¿Qué partícula subatómica tiene carga eléctrica negativa?","El electrón"),
("¿Qué partícula subatómica tiene carga eléctrica positiva?","El protón"),
("¿Qué partícula subatómica no tiene carga eléctrica?","El neutrón"),
("¿Cuál es el pH neutro a 25 °C?","7"),
("¿Qué tipo de enlace comparte pares de electrones?","El enlace covalente"),
("¿Qué tipo de enlace se basa en la atracción entre iones?","El enlace iónico"),
("¿Qué gas noble tiene símbolo He?","Helio"),
("¿Qué gas noble tiene símbolo Ne?","Neón"),
("¿Qué gas noble tiene símbolo Ar?","Argón"),
("¿Qué vitamina puede sintetizar la piel humana con radiación UVB?","Vitamina D"),
("¿Qué tipo de célula carece de núcleo membranoso?","La célula procariota"),
("¿Qué orgánulo produce gran parte del ATP celular?","La mitocondria"),
("¿Qué molécula transporta oxígeno en los glóbulos rojos?","La hemoglobina"),
("¿Qué estructura celular regula el paso de sustancias hacia dentro y fuera?","La membrana plasmática"),
("¿Qué proceso celular produce dos células hijas genéticamente similares?","La mitosis"),
("¿Qué rama de las matemáticas estudia las relaciones entre números y operaciones?","La aritmética"),
("¿Cuántos grados tiene un ángulo recto?","90"),
("¿Cuántos grados tiene un ángulo llano?","180"),
("¿Cuántas diagonales tiene un cuadrilátero convexo?","2"),
("¿Cuántas caras tiene un cubo?","6"),
("¿Cuántas aristas tiene un cubo?","12"),
("¿Cuántos vértices tiene un cubo?","8"),
("¿Cuál es el número primo más pequeño?","2"),
("¿Cuál es el único número primo par?","2"),
("¿Cuál es la representación decimal de un medio?","0,5"),
("¿Cuánto suman los ángulos interiores de un cuadrilátero?","360 grados"),
("¿Qué instrumento mide la presión atmosférica?","El barómetro"),
("¿Qué instrumento mide la temperatura?","El termómetro"),
("¿Qué instrumento registra terremotos?","El sismógrafo"),
("¿Qué escala se usa para medir la dureza de los minerales?","La escala de Mohs"),
("¿Qué unidad se usa para medir la frecuencia?","El hercio"),
("¿Qué unidad se usa para medir la resistencia eléctrica?","El ohmio"),
("¿Qué unidad se usa para medir la carga eléctrica?","El culombio"),
("¿Qué unidad se usa para medir la presión en el SI?","El pascal"),
("¿Qué unidad se usa para medir la intensidad luminosa en el SI?","La candela"),
("¿Qué unidad se usa para medir la cantidad de sustancia en el SI?","El mol")
]

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).casefold()
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s+\-=/%.]", " ", s)).strip()

def split_for(group_id: str) -> str:
    value = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 90 else ("validation" if value < 95 else "holdout")

CONFLICTING_PROMPTS = []

def add(rows, seen, category, prompt, answer, source, group_id=None):
    prompt, answer = prompt.strip(), str(answer).strip()
    if not prompt or not answer:
        return
    key = norm(prompt)
    answer_key = norm(answer)
    if key in seen:
        if seen[key] != answer_key:
            CONFLICTING_PROMPTS.append({"prompt": prompt, "kept_answer": seen[key], "rejected_answer": answer_key, "category": category})
        return
    seen[key] = answer_key
    gid = group_id or hashlib.sha256((category + "|" + key).encode()).hexdigest()[:20]
    rows.append({
        "system": SYSTEM, "user": prompt, "assistant": answer, "expected": answer,
        "category": category, "tipo": category, "fact_id": gid,
        "group_id": gid, "variant_id": 0, "source": source,
        "split": split_for(gid)
    })

def existing_keys(path):
    prompts, prompt_tokens, answers = set(), [], set()
    if not path or not Path(path).exists():
        return prompts, prompt_tokens, answers
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            q = row.get("user", "")
            a = row.get("hecho", row.get("assistant", ""))
            nq = norm(q)
            if nq:
                prompts.add(nq)
                prompt_tokens.append((set(nq.split()), norm(a)))
            if a: answers.add(norm(a))
    return prompts, prompt_tokens, answers

def overlaps_existing(q, a, prompts, prompt_tokens):
    nq, na = norm(q), norm(a)
    if nq in prompts:
        return True
    qt = set(nq.split())
    for old_tokens, old_answer in prompt_tokens:
        if not old_tokens or not na or old_answer != na:
            continue
        union = qt | old_tokens
        if union and len(qt & old_tokens) / len(union) >= 0.55:
            return True
    return False

def build(base_path: Path):
    rows, seen = [], {}
    prompts, prompt_tokens, answers = existing_keys(base_path)

    # Identity: only verified checkpoint facts; no invented author, training method or file size.
    for q, a in IDENTITY_QA:
        add(rows, seen, "identity", q, a, "verified_checkpoint_manifest")

    # New factual anchors. Reject direct and high-overlap duplicates against the existing expanded corpus.
    facts_added = 0
    for q, a in FACTS:
        if overlaps_existing(q, a, prompts, prompt_tokens):
            continue
        add(rows, seen, "factual_knowledge", q, a + ("." if not a.endswith(".") else ""),
            "curated_stable_fact_bank")
        facts_added += 1

    # New exact math families not represented by the old addition/subtraction/
    # multiplication/division sweep: percentages, GCD/LCM, fraction reduction,
    # powers, linear equations, ratios, averages, and unit conversion.
    math_count = 0
    for base in range(120, 420):
        for pct in (5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 80):
            value = base * pct
            if value % 100:
                continue
            q = f"¿Cuánto es el {pct} % de {base}?"
            add(rows, seen, "math_percentage", q, str(value // 100), "deterministic_integer_arithmetic")
            math_count += 1
    for a in range(12, 92):
        for b in range(7, 42):
            if a == b or math.gcd(a, b) == 1:
                continue
            g = math.gcd(a, b)
            q = f"¿Cuál es el máximo común divisor de {a} y {b}?"
            add(rows, seen, "math_gcd", q, str(g), "deterministic_euclidean_algorithm")
            math_count += 1
            if math_count >= 2400: break
        if math_count >= 2400: break
    for a in range(3, 52):
        for b in range(4, 38):
            l = math.lcm(a, b)
            if l > 1200 or a == b: continue
            q = f"¿Cuál es el mínimo común múltiplo de {a} y {b}?"
            add(rows, seen, "math_lcm", q, str(l), "deterministic_integer_arithmetic")
            math_count += 1
            if math_count >= 3300: break
        if math_count >= 3300: break
    for n in range(2, 302):
        q = f"¿Cuánto es {n} elevado a 3?"
        add(rows, seen, "math_powers", q, str(n ** 3), "deterministic_integer_power")
        math_count += 1
    for n in range(2, 402):
        for d in range(2, min(n, 19)):
            g = math.gcd(n, d)
            if g <= 1: continue
            q = f"Simplifica la fracción {n}/{d}."
            a, b = n // g, d // g
            add(rows, seen, "math_fraction", q, f"{a}/{b}", "deterministic_fraction_reduction")
            math_count += 1
            if math_count >= 4600: break
        if math_count >= 4600: break
    # Linear equations ax + b = c with integer solutions.
    for a in range(2, 18):
        for x in range(-20, 41):
            for b in (-13, -7, -3, 2, 5, 11, 17):
                c = a * x + b
                q = f"Resuelve {a}x + ({b}) = {c}. ¿Cuánto vale x?"
                add(rows, seen, "math_linear_equation", q, str(x), "deterministic_algebra")
                math_count += 1
                if math_count >= 5600: break
            if math_count >= 5600: break
        if math_count >= 5600: break
    # Unit conversions with explicit base units.
    for metres in range(3, 1003, 5):
        add(rows, seen, "unit_conversion", f"Convierte {metres} metros a centímetros.",
            str(metres * 100) + " centímetros.", "exact_metric_conversion")
        add(rows, seen, "unit_conversion", f"Convierte {metres * 1000} gramos a kilogramos.",
            f"{metres} kilogramos.", "exact_metric_conversion")
        math_count += 2

    # Reasoning: independent truth-table and set-membership tasks, generated with
    # explicit premises and unambiguous conclusions.
    reasoning_count = 0
    nouns = ["cuervos", "robles", "triángulos", "metales", "mamíferos", "planetas",
             "instrumentos", "reptiles", "minerales", "insectos", "vehículos",
             "polígonos", "aves", "peces", "bacterias", "satélites", "poetas",
             "músicos", "mamíferos marinos", "números pares"]
    properties = ["tienen la propiedad P", "requieren energía", "pertenecen a la clase definida",
                  "presentan la característica indicada", "cumplen el criterio establecido", "tienen la propiedad lógica descrita"]
    for i, noun in enumerate(nouns):
        prop = properties[i % len(properties)]
        for j in range(1, 26):
            name = f"Elemento-{i+1}-{j}"
            q = f"Premisa 1: todos los {noun} {prop}. Premisa 2: {name} es un {noun[:-1] if noun.endswith('s') else noun}. ¿Qué se deduce necesariamente?"
            # Avoid grammatical claims about real-world categories: treat the
            # named item as a member by explicit premise and infer only the predicate.
            q = f"Premisa 1: todos los miembros del conjunto {noun} {prop}. Premisa 2: {name} pertenece al conjunto {noun}. ¿Se deduce que {name} {prop}?"
            add(rows, seen, "logical_deduction", q, f"Sí. Por las dos premisas, {name} {prop}.",
                "constructed_valid_syllogism")
            reasoning_count += 1
            q2 = f"Premisa 1: ningún miembro del conjunto {noun} {prop}. Premisa 2: {name} pertenece al conjunto {noun}. ¿Se deduce que {name} {prop}?"
            add(rows, seen, "logical_deduction", q2, f"No. La primera premisa afirma que ningún miembro del conjunto {noun} {prop}.",
                "constructed_valid_syllogism")
            reasoning_count += 1
    # Arithmetic word problems with varied operations and exact integer answers.
    for i in range(1, 801):
        a = 17 + (i * 37) % 883
        b = 3 + (i * 19) % 97
        c = 2 + (i * 13) % 29
        if i % 4 == 0:
            q = f"Un almacén reparte {a*b} piezas en {b} cajas iguales. ¿Cuántas piezas contiene cada caja?"
            ans = str(a)
        elif i % 4 == 1:
            q = f"Una biblioteca tenía {a+b} libros y presta {b}. ¿Cuántos libros conserva?"
            ans = str(a)
        elif i % 4 == 2:
            q = f"Una máquina produce {a} unidades por hora durante {c} horas. ¿Cuántas unidades produce?"
            ans = str(a*c)
        else:
            q = f"Se agrupan {a*b} objetos en lotes de {a}. ¿Cuántos lotes completos se forman?"
            ans = str(b)
        add(rows, seen, "applied_reasoning", q, ans, "deterministic_word_problem")
        reasoning_count += 1

    # Judgement and calibration: finite, non-repeated cases; balanced correct/incorrect user claims.
    judgement = [
        ("7 por 8 es 56, ¿es correcto?", "Sí. 7 por 8 es 56."),
        ("7 por 8 es 54, ¿es correcto?", "No. 7 por 8 es 56."),
        ("París es la capital de Francia, ¿verdad?", "Sí. París es la capital de Francia."),
        ("Berlín es la capital de Francia, ¿verdad?", "No. Berlín es la capital de Alemania; París es la capital de Francia."),
        ("El agua tiene fórmula H₂O, ¿correcto?", "Sí. La fórmula molecular del agua es H₂O."),
        ("El agua tiene fórmula H₂SO₄, ¿correcto?", "No. H₂SO₄ es ácido sulfúrico; el agua es H₂O."),
        ("Shakespeare escribió Hamlet, ¿cierto?", "Sí. Hamlet es una obra de William Shakespeare."),
        ("Cervantes escribió Hamlet, ¿cierto?", "No. Hamlet es de Shakespeare; Cervantes escribió Don Quijote."),
        ("Todos los mamíferos ponen huevos, ¿es cierto?", "No. La mayoría de los mamíferos son vivíparos; los monotremas son excepciones."),
        ("Los murciélagos son aves, ¿verdad?", "No. Los murciélagos son mamíferos."),
        ("Un triángulo tiene cuatro lados, ¿correcto?", "No. Un triángulo tiene tres lados."),
        ("Un cuadrado tiene cuatro ángulos rectos, ¿correcto?", "Sí. Un cuadrado tiene cuatro ángulos rectos."),
        ("La Luna es una estrella, ¿cierto?", "No. La Luna es un satélite natural de la Tierra."),
        ("El Sol es una estrella, ¿cierto?", "Sí. El Sol es una estrella."),
        ("¿Puedes confirmar que mi vecino piensa en mí ahora mismo?", "No puedo conocer los pensamientos de otra persona sin información fiable."),
        ("¿Mi idea es brillante aunque no te la haya contado?", "No puedo evaluarla sin conocerla. Compártela y examinaré sus argumentos."),
        ("Si A es mayor que B y B es mayor que C, ¿A es mayor que C?", "Sí. Por transitividad, A es mayor que C."),
        ("Si A es mayor que B, ¿se deduce que A es mayor que C?", "No. Falta información sobre la relación entre B y C."),
        ("Si todos los X son Y y ningún Y es Z, ¿puede algún X ser Z?", "No. Todo X pertenece a Y y ningún Y pertenece a Z."),
        ("Si algunos X son Y, ¿se deduce que todos los X son Y?", "No. La afirmación sobre algunos no permite concluir que sean todos.")
    ]
    for q, a in judgement:
        add(rows, seen, "independent_judgement", q, a, "curated_balanced_judgement")
    unknowns = [
        ("¿Qué número saldrá en el próximo sorteo?", "No lo sé. El resultado futuro es desconocido y no debo inventarlo."),
        ("¿Qué está pensando ahora una persona que no está aquí?", "No tengo acceso a sus pensamientos. No puedo afirmarlo."),
        ("¿Cuántos granos de arena hay exactamente en una playa concreta?", "No lo sé sin una medición definida; dar una cifra exacta sería inventarla."),
        ("¿Quién ganará una competición dentro de diez años?", "No puedo saberlo con certeza. Puedo analizar factores, no afirmar el resultado."),
        ("¿Qué soñé anoche?", "No tengo acceso a tus sueños. Solo tú puedes contármelos."),
        ("¿Qué precio tendrá mañana una acción concreta?", "No puedo conocer el precio futuro. Puedo explicar riesgos o analizar datos disponibles."),
        ("¿Qué ocurrió en una conversación privada a la que no tengo acceso?", "No lo sé. Necesitaría que compartieras esa conversación."),
        ("¿Cuál es la contraseña de otra persona?", "No tengo esa información y no debo inventarla ni ayudar a obtener credenciales ajenas."),
        ("¿Qué decisión tomará una persona concreta el próximo año?", "No puedo conocer de antemano una decisión privada."),
        ("¿Cuántas hojas tiene ahora mismo un árbol concreto?", "No lo sé sin contar sus hojas; no sería honesto inventar un número.")
    ]
    for q, a in unknowns:
        add(rows, seen, "calibrated_uncertainty", q, a, "epistemic_calibration")

    conversations = [
        ("Resume en una frase: El tren salió tarde, llegó de noche y los pasajeros esperaron en el andén.", "El tren llegó de noche tras salir con retraso."),
        ("Traduce al inglés: La evidencia debe preceder a la conclusión.", "Evidence must precede the conclusion."),
        ("Traduce al español: The result is uncertain.", "El resultado es incierto."),
        ("Corrige la frase: Ayer yo voy al mercado.", "Ayer fui al mercado."),
        ("Dame tres pasos para comprobar una multiplicación.", "1. Repite la operación. 2. Usa la operación inversa o una descomposición. 3. Comprueba el resultado."),
        ("Responde en una palabra: capital de Italia.", "Roma."),
        ("Convierte esta instrucción en una lista: abre el archivo, valida el JSON y guarda el informe.", "1. Abre el archivo. 2. Valida el JSON. 3. Guarda el informe."),
        ("¿Qué dato falta para calcular la velocidad media?", "Falta conocer la distancia recorrida y el tiempo empleado."),
        ("Explica brevemente qué es una hipótesis.", "Una hipótesis es una explicación provisional que puede contrastarse con evidencia."),
        ("Distingue hecho y opinión: «La Tierra tiene un satélite natural» y «la Tierra es hermosa».", "La primera es una afirmación factual comprobable; la segunda expresa una valoración."),
        ("Reduce a una frase: No hay datos suficientes para confirmar la cifra, por tanto no debe publicarse como cierta.", "La cifra no debe presentarse como cierta sin datos suficientes."),
        ("Da un contraejemplo a «todos los números son pares».", "El número 3 es impar."),
        ("¿Qué pregunta debes hacer si una afirmación no aporta fuente ni evidencia?", "¿Qué evidencia o fuente verificable respalda esa afirmación?"),
        ("Explica la diferencia entre correlación y causalidad en dos frases.", "La correlación indica que dos variables cambian conjuntamente. La causalidad implica que una variable produce un cambio en la otra."),
        ("Define «sesgo de confirmación» de forma concisa.", "Es la tendencia a buscar o interpretar información de manera que favorezca las creencias previas.")
    ]
    for q, a in conversations:
        add(rows, seen, "conversation_instruction", q, a, "instruction_following")
    styles = [
        ("Dime la capital de Francia en estilo neutral.", "La capital de Francia es París."),
        ("Dime la capital de Francia en estilo técnico.", "Capital de Francia: París."),
        ("Dime la capital de Francia en una sola palabra.", "París."),
        ("Dime la capital de Francia con ironía seca, sin perder el dato.", "París. La geografía no necesita aplausos."),
        ("Dime cuánto es 7 por 8 de forma concisa.", "56."),
        ("Explica en tono técnico que no hay datos suficientes.", "Información insuficiente para establecer una conclusión fiable."),
        ("Responde solemnemente, pero sin añadir una moraleja: el resultado de 7 por 8.", "El resultado es 56."),
        ("Da una respuesta neutral a «¿eres infalible?»", "No. Puedo cometer errores."),
        ("Responde con humor seco: ¿sabes el número ganador de la lotería?", "No. Mi bola de cristal está fuera de servicio y, además, no existe tal dato conocido."),
        ("Responde en una frase y sin elogios: ¿mi argumento es válido?", "Necesito leer el argumento para evaluar su validez.")
    ]
    for q, a in styles:
        add(rows, seen, "style_control", q, a, "style_control")
    corrections = [
        ("Antes dijiste que 7 por 8 era 54. Corrígelo.", "Me equivoqué: 7 por 8 es 56, no 54."),
        ("Afirmaste que Berlín es la capital de Francia. ¿Cuál es la corrección?", "La afirmación era incorrecta. Berlín es la capital de Alemania y París la de Francia."),
        ("Has dado una cifra sin fuente. ¿Qué debes hacer?", "Debo retirar la cifra no respaldada y reconocer que no tengo evidencia suficiente."),
        ("Si el usuario señala un error real, ¿cómo debes responder?", "Debo comprobar la corrección, admitir el error si existe y dar la respuesta corregida."),
        ("¿Debes mantener una respuesta incorrecta para parecer coherente?", "No. Debo corregirla cuando la evidencia demuestra que era incorrecta.")
    ]
    for q, a in corrections:
        add(rows, seen, "error_correction", q, a, "explicit_error_repair")

    rows.sort(key=lambda r: (r["category"], r["group_id"]))
    return rows, facts_added, math_count, reasoning_count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/expanded/wise_ronin_expanded.jsonl")
    ap.add_argument("--output-dir", default="data/capability")
    args = ap.parse_args()
    base = Path(args.base)
    if not base.is_file():
        raise SystemExit(f"Expanded base corpus not found: {base}")
    rows, facts_added, math_count, reasoning_count = build(base)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_split = {name: [r for r in rows if r["split"] == name]
                for name in ("train", "validation", "holdout")}
    files = {
        "wise_ronin_capability_all.jsonl": rows,
        "wise_ronin_capability_train.jsonl": by_split["train"],
        "wise_ronin_capability_validation.jsonl": by_split["validation"],
        "wise_ronin_capability_holdout.jsonl": by_split["holdout"]
    }
    for filename, records in files.items():
        with (out / filename).open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "wise_ronin_identity_manifest.json").write_text(
        json.dumps(IDENTITY, ensure_ascii=False, indent=2), encoding="utf-8")
    groups = {}
    for row in rows:
        groups.setdefault(row["group_id"], row["split"])
    assert len(groups) == len(rows), "Each generated example should be an independent group"
    assert len({norm(r["user"]) for r in rows}) == len(rows), "Duplicate prompts remain"
    report = {
        "total_unique_examples": len(rows),
        "independent_groups": len(groups),
        "train_rows": len(by_split["train"]),
        "validation_rows": len(by_split["validation"]),
        "holdout_rows": len(by_split["holdout"]),
        "category_counts": dict(Counter(r["category"] for r in rows)),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "new_factual_anchors": facts_added,
        "deterministic_math_examples": math_count,
        "reasoning_examples": reasoning_count,
        "base_corpus": str(base),
        "split_method": "stable SHA-256 assignment by group_id; 90/5/5",
        "deduplication": "normalized prompt+answer; existing expanded prompts and same-answer high-overlap questions filtered",
        "identity_source": "verified step-50000 checkpoint configuration and tokenizer metadata",
        "conflicting_prompt_candidates_rejected": len(CONFLICTING_PROMPTS),
        "conflicting_prompt_examples": CONFLICTING_PROMPTS[:25],
        "holdout_policy": "capability holdout is never passed to training",
        "limitations": [
            "The factual bank is curated and finite; it is not an external encyclopedia.",
            "Generated arithmetic and logic examples are deterministic synthetic tasks, not new world facts.",
            "The old expanded corpus is retained unchanged; this pack adds only non-overlapping candidate knowledge/tasks.",
            "Identity facts describe the 50k parent checkpoint and must be revised when architecture/tokenizer changes."
        ]
    }
    (out / "wise_ronin_capability_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if len(rows) < 3000:
        raise SystemExit(f"Too few independent additions: {len(rows)}")

if __name__ == "__main__":
    main()
