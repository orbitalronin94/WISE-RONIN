#!/usr/bin/env python3
"""Generate a large, deterministic WISE-RONIN corpus with factual anchors and varied style.

The row count is not presented as the number of independent facts: every record carries
fact_id/base_id and variant_id so splits and audits can group paraphrases safely.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path

SYSTEM = ("Eres WISE-RONIN. Respondón y antisicofántico. Da primero el hecho cierto; "
         "después, si procede, un rodeo seco y distinto. Nunca adules, nunca inventes "
         "datos ni conviertas una duda en certeza. Si una afirmación es discutible, "
         "explica el matiz. No des moralejas ni hables como un asistente genérico.")

# Stable, low-ambiguity capital facts. Deliberately omit disputed/administrative edge cases.
CAPITALS = [
("Japón","Tokio"),("Francia","París"),("España","Madrid"),("Italia","Roma"),("Alemania","Berlín"),
("Portugal","Lisboa"),("Irlanda","Dublín"),("Bélgica","Bruselas"),("Suiza","Berna"),("Austria","Viena"),
("Grecia","Atenas"),("Turquía","Ankara"),("Egipto","El Cairo"),("Marruecos","Rabat"),("México","Ciudad de México"),
("Brasil","Brasilia"),("Argentina","Buenos Aires"),("Chile","Santiago"),("Perú","Lima"),("Colombia","Bogotá"),
("Canadá","Ottawa"),("Estados Unidos","Washington D. C."),("China","Pekín"),("India","Nueva Delhi"),
("Rusia","Moscú"),("Australia","Canberra"),("Corea del Sur","Seúl"),("Tailandia","Bangkok"),
("Noruega","Oslo"),("Suecia","Estocolmo"),("Finlandia","Helsinki"),("Dinamarca","Copenhague"),
("Polonia","Varsovia"),("Hungría","Budapest"),("Rumanía","Bucarest"),("Bulgaria","Sofía"),
("Croacia","Zagreb"),("Serbia","Belgrado"),("Islandia","Reikiavik"),("Cuba","La Habana"),
("Venezuela","Caracas"),("Ecuador","Quito"),("Paraguay","Asunción"),("Uruguay","Montevideo"),
("Kenia","Nairobi"),("Nigeria","Abuya"),("Etiopía","Adís Abeba"),("Irán","Teherán"),
("Irak","Bagdad"),("Pakistán","Islamabad"),("Vietnam","Hanói"),("Filipinas","Manila"),
("Nueva Zelanda","Wellington"),("Países Bajos","Ámsterdam"),("Chequia","Praga"),("Eslovaquia","Bratislava"),
("Eslovenia","Liubliana"),("Estonia","Tallin"),("Letonia","Riga"),("Lituania","Vilna"),
("Mongolia","Ulán Bator"),("Nepal","Katmandú"),("Bangladés","Daca"),("Afganistán","Kabul"),
("Arabia Saudí","Riad"),("Jordania","Amán"),("Líbano","Beirut"),("Túnez","Túnez"),
("Senegal","Dakar"),("Ghana","Acra"),("Uganda","Kampala"),("Tanzania","Dodoma"),
("Angola","Luanda"),("Mozambique","Maputo"),("Zambia","Lusaka"),("Zimbabue","Harare"),
("Bolivia","Sucre, capital constitucional"),("Sudáfrica","Pretoria, capital ejecutiva")
]

# Element, symbol, atomic number. Atomic numbers and symbols are stable reference facts.
ELEMENTS = [
("Hidrógeno","H"),("Helio","He"),("Litio","Li"),("Berilio","Be"),("Boro","B"),("Carbono","C"),
("Nitrógeno","N"),("Oxígeno","O"),("Flúor","F"),("Neón","Ne"),("Sodio","Na"),("Magnesio","Mg"),
("Aluminio","Al"),("Silicio","Si"),("Fósforo","P"),("Azufre","S"),("Cloro","Cl"),("Argón","Ar"),
("Potasio","K"),("Calcio","Ca"),("Escandio","Sc"),("Titanio","Ti"),("Vanadio","V"),("Cromo","Cr"),
("Manganeso","Mn"),("Hierro","Fe"),("Cobalto","Co"),("Níquel","Ni"),("Cobre","Cu"),("Zinc","Zn"),
("Galio","Ga"),("Germanio","Ge"),("Arsénico","As"),("Selenio","Se"),("Bromo","Br"),("Kriptón","Kr"),
("Rubidio","Rb"),("Estroncio","Sr"),("Itrio","Y"),("Circonio","Zr"),("Niobio","Nb"),("Molibdeno","Mo"),
("Tecnecio","Tc"),("Rutenio","Ru"),("Rodio","Rh"),("Paladio","Pd"),("Plata","Ag"),("Cadmio","Cd"),
("Indio","In"),("Estaño","Sn"),("Antimonio","Sb"),("Telurio","Te"),("Yodo","I"),("Xenón","Xe"),
("Cesio","Cs"),("Bario","Ba"),("Lantano","La"),("Cerio","Ce"),("Praseodimio","Pr"),("Neodimio","Nd"),
("Prometio","Pm"),("Samario","Sm"),("Europio","Eu"),("Gadolinio","Gd"),("Terbio","Tb"),("Disprosio","Dy"),
("Holmio","Ho"),("Erbio","Er"),("Tulio","Tm"),("Iterbio","Yb"),("Lutecio","Lu"),("Hafnio","Hf"),
("Tantalio","Ta"),("Wolframio","W"),("Renio","Re"),("Osmio","Os"),("Iridio","Ir"),("Platino","Pt"),
("Oro","Au"),("Mercurio","Hg"),("Talio","Tl"),("Plomo","Pb"),("Bismuto","Bi"),("Polonio","Po"),
("Astato","At"),("Radón","Rn"),("Francio","Fr"),("Radio","Ra"),("Actinio","Ac"),("Torio","Th"),
("Protactinio","Pa"),("Uranio","U"),("Neptunio","Np"),("Plutonio","Pu"),("Americio","Am"),("Curio","Cm"),
("Berkelio","Bk"),("Californio","Cf"),("Einsteinio","Es"),("Fermio","Fm"),("Mendelevio","Md"),("Nobelio","No"),
("Lawrencio","Lr"),("Rutherfordio","Rf"),("Dubnio","Db"),("Seaborgio","Sg"),("Bohrio","Bh"),("Hassio","Hs"),
("Meitnerio","Mt"),("Darmstadtio","Ds"),("Roentgenio","Rg"),("Copernicio","Cn"),("Nihonio","Nh"),
("Flerovio","Fl"),("Moscovio","Mc"),("Livermorio","Lv"),("Teneso","Ts"),("Oganesón","Og")
]

GENERAL = [
("¿En qué año llegó el ser humano a la Luna?","1969"),
("¿Quién escribió Hamlet?","William Shakespeare"),
("¿Quién escribió Don Quijote de la Mancha?","Miguel de Cervantes"),
("¿Quién escribió Cien años de soledad?","Gabriel García Márquez"),
("¿Quién escribió Rayuela?","Julio Cortázar"),
("¿Quién escribió la Divina comedia?","Dante Alighieri"),
("¿Quién pintó la Gioconda?","Leonardo da Vinci"),
("¿Quién pintó el Guernica?","Pablo Picasso"),
("¿Quién compuso la Novena Sinfonía?","Ludwig van Beethoven"),
("¿Cuál es el océano más grande?","El océano Pacífico"),
("¿Cuál es el planeta más cercano al Sol?","Mercurio"),
("¿Cuál es el planeta más grande del sistema solar?","Júpiter"),
("¿Cuál es el planeta más lejano del Sol entre los ocho reconocidos?","Neptuno"),
("¿Cuántos planetas reconoce la Unión Astronómica Internacional en el sistema solar?","Ocho"),
("¿Cuál es la fórmula molecular del agua?","H₂O"),
("¿Cuál es la fórmula molecular de la glucosa?","C₆H₁₂O₆"),
("¿Cuál es el símbolo químico del oro?","Au"),
("¿Cuál es el símbolo químico de la plata?","Ag"),
("¿Qué elemento tiene número atómico 1?","Hidrógeno"),
("¿Qué elemento tiene número atómico 6?","Carbono"),
("¿Qué elemento tiene número atómico 8?","Oxígeno"),
("¿Qué elemento tiene número atómico 26?","Hierro"),
("¿Qué elemento tiene número atómico 79?","Oro"),
("¿Qué elemento tiene número atómico 92?","Uranio"),
("¿Cuál es la unidad del SI para la fuerza?","Newton"),
("¿Cuál es la unidad del SI para la energía?","Julio"),
("¿Cuál es la unidad del SI para la potencia?","Vatio"),
("¿Cuál es la velocidad de la luz en el vacío, aproximadamente?","299.792.458 metros por segundo"),
("¿Cuál es el gas más abundante de la atmósfera terrestre seca?","Nitrógeno"),
("¿Qué gas necesitan las células humanas para la respiración aeróbica?","Oxígeno"),
("¿Qué molécula almacena gran parte de la información genética?","ADN"),
("¿Qué proceso permite a las plantas convertir energía luminosa en energía química?","Fotosíntesis"),
("¿Qué fuerza mantiene a los planetas en órbita alrededor del Sol?","Gravedad"),
("¿Qué científico formuló las leyes clásicas del movimiento?","Isaac Newton"),
("¿Quién desarrolló la teoría de la relatividad general?","Albert Einstein"),
("¿Quién propuso la selección natural como mecanismo de evolución?","Charles Darwin"),
("¿Quién descubrió la penicilina?","Alexander Fleming"),
("¿En qué año comenzó la Segunda Guerra Mundial en Europa?","1939"),
("¿En qué año terminó la Segunda Guerra Mundial?","1945"),
("¿En qué año cayó el Muro de Berlín?","1989"),
("¿Quién fue el primer ser humano en viajar al espacio?","Yuri Gagarin"),
("¿En qué año viajó Yuri Gagarin al espacio?","1961"),
("¿Cuál es el animal terrestre más rápido en carrera corta?","El guepardo"),
("¿Cuál es el animal más grande conocido actualmente?","La ballena azul"),
("¿Cuál es el mamífero que puede realizar vuelo activo?","El murciélago"),
("¿Qué mamífero pone huevos y tiene pico?","El ornitorrinco"),
("¿Cuántos huesos tiene normalmente el esqueleto adulto humano?","206"),
("¿Cuál es el órgano más extenso del cuerpo humano?","La piel"),
("¿Cuál es el hueso más largo del cuerpo humano?","El fémur"),
("¿Cuántos lados tiene un hexágono?","Seis"),
("¿Cuánto suman los ángulos interiores de un triángulo euclídeo?","180 grados"),
("¿Cuál es el valor aproximado de pi a cuatro decimales?","3,1416"),
("¿Cuál es la raíz cuadrada principal de 144?","12"),
("¿Cuánto vale 2 elevado a 10?","1024"),
("¿Cuál es el río que atraviesa París?","El Sena"),
("¿Cuál es el río que atraviesa Londres?","El Támesis"),
("¿Cuál es el río que atraviesa Roma?","El Tíber"),
("¿Qué mar se encuentra entre el sur de Europa y el norte de África?","El mar Mediterráneo"),
("¿Cuál es el país de mayor superficie territorial?","Rusia"),
("¿Cuál es el país soberano de menor superficie?","Ciudad del Vaticano"),
("¿Cuál es la isla más grande del mundo, excluyendo continentes?","Groenlandia"),
("¿Cuál es el desierto cálido más extenso?","El Sáhara"),
("¿Qué lengua tiene más hablantes nativos según estimaciones habituales?","Chino mandarín"),
("¿Qué idioma es oficial en Brasil?","Portugués"),
("¿En qué país se desarrolló la fabricación del papel en la Antigüedad?","China"),
("¿Quién es tradicionalmente asociado con la imprenta europea de tipos móviles?","Johannes Gutenberg"),
("¿En qué siglo se difundió la imprenta de tipos móviles de Gutenberg en Europa?","Siglo XV"),
("¿Quién escribió 1984?","George Orwell"),
("¿Quién escribió El principito?","Antoine de Saint-Exupéry"),
("¿Quién escribió Orgullo y prejuicio?","Jane Austen"),
("¿Quién escribió Moby-Dick?","Herman Melville"),
("¿Quién escribió Crimen y castigo?","Fiódor Dostoyevski"),
("¿Quién escribió Guerra y paz?","León Tolstói"),
("¿Quién escribió El señor de los anillos?","J. R. R. Tolkien"),
("¿Quién escribió La casa de Bernarda Alba?","Federico García Lorca"),
("¿Quién pintó Las meninas?","Diego Velázquez"),
("¿Quién pintó La noche estrellada?","Vincent van Gogh"),
("¿Quién pintó El grito?","Edvard Munch"),
("¿Quién compuso Las cuatro estaciones?","Antonio Vivaldi"),
("¿Quién escribió El arte de la guerra?","Sun Tzu"),
("¿Quién escribió El príncipe?","Nicolás Maquiavelo"),
("¿Quién escribió La República?","Platón"),
("¿Quién escribió Ética a Nicómaco?","Aristóteles"),
("¿Quién escribió Meditaciones?","Marco Aurelio"),
("¿Cuál es la capital de Aragón?","Zaragoza"),
("¿Cuál es la capital de Cataluña?","Barcelona"),
("¿Cuál es la capital de Galicia?","Santiago de Compostela"),
("¿Cuál es la capital de Andalucía?","Sevilla"),
("¿Cuál es la capital de la Comunidad Valenciana?","Valencia"),
("¿Cuál es la capital del País Vasco?","Vitoria-Gasteiz"),
("¿Cuál es la capital de Asturias?","Oviedo"),
("¿Cuál es la capital de Cantabria?","Santander"),
("¿Cuál es la capital de Extremadura?","Mérida"),
("¿Cuál es la capital de Castilla y León?","Valladolid")
]

# Each pattern puts the answer first or very near the beginning. No generic fallback.
OPENERS = [
"El hecho es {a}.", "Dato confirmado: {a}.", "La respuesta corta es {a}.",
"Queda así: {a}.", "Apunta el dato: {a}.", "Sin incienso: {a}.",
"El registro dice {a}.", "La cifra correcta es {a}.", "No hay misterio: {a}.",
"Respuesta: {a}.", "Te dejo el dato: {a}.", "En limpio: {a}.",
"El archivo señala {a}.", "La cosa es {a}.", "No es una opinión: {a}.",
"Primero el hecho: {a}.", "La cuenta da {a}.", "El nombre correcto es {a}.",
"Eso tiene nombre: {a}.", "Queda escrito: {a}."
]
TAILS = [
"El resto es ruido de cubiertos.", "No hace falta ponerle una corona.",
"El espejo no participa en la medición.", "La palmadita no cambia el resultado.",
"El cajón puede quedarse cerrado.", "No es una moraleja: es el dato.",
"El taller entrega hechos, no aplausos.", "La novela empieza después del punto.",
"Ni incienso ni ceremonia: compruébalo.", "El portazo es opcional; la cifra no.",
"El universo no ha firmado una excepción para tu ego.", "No confundas seguridad con volumen.",
"El dato no necesita que lo quieras.", "La llave no altera la cerradura.",
"Si buscabas incienso, te has equivocado de mostrador.", "El papel aguanta la cifra.",
"El aplauso no forma parte del método.", "No le pongas bigote a la evidencia.",
"El resultado no depende del humor del espejo.", "Y ahí termina el teatro.",
"El dato no lleva uniforme ni pide permiso.", "No confundas el envoltorio con el contenido.",
"El taller no certifica egos.", "La evidencia no se arrodilla ante el ruido.",
"Puedes discutir el método; la cuenta sigue ahí.", "El mapa no cambia para consolar al viajero.",
"El dato cabe en una línea; el teatro, en un almacén.", "No hay premio por adornar la respuesta.",
"El número no ha venido a caerte bien.", "La respuesta no necesita escolta.",
"El archivo no registra aplausos.", "No hace falta convertirlo en leyenda.",
"Que el dato sea seco no lo vuelve menos cierto.", "La mesa aguanta el papel, no la fantasía.",
"Ni incienso ni humo: una comprobación basta.", "La respuesta no es una invitación a inventar.",
"El margen queda libre para tus objeciones.", "El dato está servido; el ego que espere."
]
OBJECTS = [
"una taza fría","un mapa doblado","un candado sin llave","una pizarra limpia","un reloj parado",
"una servilleta arrugada","un libro abierto","una llave inglesa","un vaso vacío","un rotulador seco",
"una corona de cartón","un billete usado","una brújula quieta","un cajón cerrado","una lámpara apagada",
"un papel cuadriculado","un banco de taller","una ventana abierta","un saco vacío","un espejo sin marco"
]
QUESTION_PREFIXES = [
"", "Oye, ", "Dime, ", "Sin rodeos: ", "A ver, ", "Una pregunta: ",
"Te pregunto una cosa: ", "Vamos al dato: ", "Compruébame esto: ", "En serio, ",
"Necesito un dato: ", "Contesta con precisión: ", "Una comprobación: ",
"Resuelve esta duda: ", "Quiero verificar algo: ", "Pregunta directa: ",
"Sin ceremonia: ", "Vamos a lo concreto: ", "No te me escapes: ",
"Consulta breve: ", "Dame el dato exacto: ", "A ver si lo sabes: ",
"En el registro: ", "Para dejarlo escrito: ", "Una de cultura general: ",
"Comprueba el dato: ", "Queda una pregunta: ", "Vamos por partes: ",
"Esto tiene respuesta: ", "No es una adivinanza: ", "En términos claros: ",
"Que conste en acta: ", "Pregunta de examen: ", "Te pongo a prueba: ",
"Al grano: ", "Sin humo: ", "Responde sin adornos: ",
"Una cuestión factual: ", "Abre el atlas: ", "Mira la tabla: "
]

def stable_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def fact_rows():
    facts=[]
    for country,capital in CAPITALS:
        facts.append((f"¿Cuál es la capital de {country}?",capital,"geography"))
    for number,(element,symbol) in enumerate(ELEMENTS,1):
        facts.append((f"¿Cuál es el símbolo químico de {element}?",symbol,"chemistry"))
        facts.append((f"¿Qué elemento tiene número atómico {number}?",element,"chemistry"))
    for q,a in GENERAL:
        facts.append((q,a,"general"))
    # Remove duplicate question/answer pairs without silently resolving conflicts.
    seen={}
    for q,a,cat in facts:
        if q in seen and seen[q][0] != a:
            raise ValueError(f"Conflicting fact for {q}: {seen[q][0]} / {a}")
        seen[q]=(a,cat)
    return [(q,a,cat) for q,(a,cat) in sorted(seen.items())]

def arithmetic_rows(limit: int):
    rows=[]
    # Distinct, exactly solvable tasks across operations and difficulty bands.
    for a in range(17,217):
        for b in range(13,113):
            rows.append((f"¿Cuánto es {a} más {b}?",str(a+b),"addition"))
            rows.append((f"¿Cuánto es {a} menos {b}?",str(a-b),"subtraction"))
            if b <= 60:
                rows.append((f"¿Cuánto es {a} por {b}?",str(a*b),"multiplication"))
            if a % b == 0:
                rows.append((f"¿Cuánto es {a} dividido entre {b}?",str(a//b),"division"))
            if len(rows) >= limit: return rows[:limit]
    for n in range(2,500):
        rows.append((f"¿Cuál es el cuadrado de {n}?",str(n*n),"square"))
        rows.append((f"¿Cuál es el doble de {n} elevado al cuadrado?",str(2*n*n),"composite_math"))
        if n <= 100:
            rows.append((f"¿Cuál es el cubo de {n}?",str(n**3),"cube"))
        if len(rows) >= limit: return rows[:limit]
    return rows[:limit]

def render(q: str, answer: str, idx: int, variant: int, base_id: str):
    # A combinatorial phrase space: formula + tail + object + connective.
    opener=OPENERS[(idx*7+variant*11) % len(OPENERS)]
    tail=TAILS[(idx*13+variant*3) % len(TAILS)]
    obj=OBJECTS[(idx*5+variant*7) % len(OBJECTS)]
    mode=(idx+variant) % 5
    fact=opener.format(a=answer)
    if mode==0: out=f"{fact} {tail}"
    elif mode==1: out=f"{fact} Sobre la mesa hay {obj}; no es una prueba, es atrezo. {tail}"
    elif mode==2: out=f"{fact} {tail} La pizarra sigue siendo una pizarra."
    elif mode==3: out=f"{fact} Te lo dejo junto a {obj}. {tail}"
    else: out=f"{fact} Ni corona ni incienso: {tail}"
    # Ensure response-level uniqueness through meaningful formula selection, not hash noise.
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="data/expanded/wise_ronin_expanded.jsonl")
    ap.add_argument("--target-rows",type=int,default=147627)
    ap.add_argument("--variants-per-base",type=int,default=10)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    if args.target_rows < 1000 or args.variants_per_base < 1:
        ap.error("target-rows debe ser >=1000 y variants-per-base >=1")
    rng=random.Random(args.seed)
    facts=fact_rows()
    # Keep every factual anchor, then fill the requested volume with distinct,
    # exactly solvable arithmetic questions. Never inflate the factual count by relabeling math.
    fact_variants=40
    factual=[]
    for i,(q,a,cat) in enumerate(facts):
        base=stable_id("fact|"+q+"|"+a)
        for v in range(fact_variants):
            pref=QUESTION_PREFIXES[(i*7+v) % len(QUESTION_PREFIXES)]
            user=pref+q
            factual.append({"system":SYSTEM,"user":user,"assistant":render(q,a,i,v,base),
                            "tipo":"factual_"+cat,"fact_id":base,"variant_id":v,"hecho":a})
    if len(factual)>=args.target_rows:
        raise SystemExit("El banco factual por sí solo supera target-rows; aumenta el objetivo.")
    math_target=args.target_rows-len(factual)
    math_rows=[]
    need=math_target
    # Enough unique base problems; each gets at most ten paraphrases.
    math_variants=min(args.variants_per_base,10)
    math_facts=arithmetic_rows((need+math_variants-1)//math_variants+500)
    for j,(q,a,cat) in enumerate(math_facts):
        base=stable_id("math|"+q+"|"+a)
        for v in range(math_variants):
            if len(math_rows)>=need: break
            pref=QUESTION_PREFIXES[(j*5+v*3) % len(QUESTION_PREFIXES)]
            math_rows.append({"system":SYSTEM,"user":pref+q,"assistant":render(q,a,j+len(facts),v,base),
                              "tipo":"math_"+cat,"fact_id":base,"variant_id":v,"hecho":a})
        if len(math_rows)>=need: break
    rows=factual+math_rows
    seen_user=set(); seen_pair=set(); clean=[]
    for row in rows:
        u=row["user"].strip()
        pair=(u,row["assistant"])
        if u in seen_user or pair in seen_pair:
            continue
        seen_user.add(u); seen_pair.add(pair); clean.append(row)
    # If paraphrase collisions occur, deterministically add unused arithmetic tasks.
    if len(clean)<args.target_rows:
        extra=arithmetic_rows(args.target_rows*2)
        for j,(q,a,cat) in enumerate(extra):
            for v in range(args.variants_per_base):
                if len(clean)>=args.target_rows: break
                user=QUESTION_PREFIXES[(j+v*7)%len(QUESTION_PREFIXES)]+q
                if user in seen_user: continue
                base=stable_id("extra|"+q+"|"+a)
                row={"system":SYSTEM,"user":user,"assistant":render(q,a,j+99999,v,base),
                     "tipo":"math_"+cat,"fact_id":base,"variant_id":v,"hecho":a}
                if row["assistant"] in {x["assistant"] for x in clean[-500:]}:
                    continue
                clean.append(row); seen_user.add(user)
            if len(clean)>=args.target_rows: break
    if len(clean)<args.target_rows:
        raise SystemExit(f"No se alcanzó target: {len(clean)}/{args.target_rows}")
    clean=clean[:args.target_rows]
    rng.shuffle(clean)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8") as f:
        for row in clean: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    counts=Counter(r["tipo"] for r in clean)
    base_counts=Counter(r["fact_id"] for r in clean)
    report={"rows":len(clean),"unique_prompts":len({r["user"] for r in clean}),
            "unique_responses":len({r["assistant"] for r in clean}),
            "independent_fact_ids":len(base_counts),"max_variants_per_fact":max(base_counts.values()),"factual_anchor_count":len(facts),"arithmetic_base_count":sum(1 for k in base_counts if k not in {stable_id("fact|"+q+"|"+a) for q,a,_ in facts}),
            "type_counts":dict(counts),"target_rows":args.target_rows,
            "seed":args.seed,"source":"deterministic built-in fact bank + exact arithmetic",
            "warning":"Row count is not independent knowledge count; group by fact_id for splits/evaluation."}
    report_path=out.with_name("wise_ronin_expanded_report.json")
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
