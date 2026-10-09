"""Declaración de los análisis (igual que PU.DEF_ANALISIS de la versión de escritorio): métodos,
columnas que necesita cada uno, opciones, efectos fijos y pedigrí aparte."""


def _rol(id_, etiqueta, tipo="id", ayuda=None, **extra):
    rol = dict(id=id_, etiqueta=etiqueta, tipo=tipo, ayuda=ayuda, requerido=False, auto=True)
    rol.update(extra)
    return rol


R = {
    "padre": _rol("padre", "Padre", ayuda="Identificación del padre (toro, carnero…)"),
    "madre": _rol("madre", "Madre", ayuda="Identificación de la madre"),
    "animal": _rol("animal", "Animal", ayuda="Identificación de cada individuo"),
    "id": _rol("id", "Animal (ID)", ayuda="Identificación de cada individuo"),
    "rasgo": _rol("rasgo", "Rasgo", "num", "Variable medida (numérica)"),
    "sexo": _rol("sexo", "Sexo", ayuda="1/M = macho, 2/H/0 = hembra"),
    "fecha": _rol("fecha", "Fecha o año de nacimiento", ayuda="Fecha (dd/mm/aaaa o de Excel) o solo el año"),
    "referencia": _rol("referencia", "Población de referencia",
                       ayuda="Columna con 1 en los animales de referencia (p. ej. los vivos). Vacío = valor por defecto."),
    "subpoblacion": _rol("subpoblacion", "Subpoblación", ayuda="Área, rebaño, línea, color… que agrupa a los animales"),
    "rebano": _rol("rebano", "Rebaño de nacimiento", ayuda="Rebaño, hato o criador donde nació cada animal"),
}


def req(r, **extra):
    return dict(R[r], requerido=True, **extra)


def opc(r, **extra):
    return dict(R[r], **extra)


ROLES_PED = [req("id"), opc("padre"), opc("madre")]
OPC_LRT = dict(clave="lrt", etiqueta="Prueba de razón de verosimilitud (un ajuste extra)", tipo="check", defecto=True)
TIPO_REG = [("padre", "Padre (h² = 2b)"), ("madre", "Madre (h² = 2b)"), ("promedio", "Promedio de los padres (h² = b)"),
            ("madre_intra_padre", "Madre, dentro de padre (h² = 2b)")]

DEF = {
    "descriptiva": dict(
        titulo="Estadística descriptiva", eyebrow="Datos", icono=":material/bar_chart:",
        sub="Resumen, distribución, normalidad y correlaciones fenotípicas de las variables numéricas, con opción de resumir por grupo.",
        metodos=[dict(id="descriptiva", nombre="Resumen, histogramas y correlaciones",
                      roles=[_rol("grupo", "Agrupar por (opcional)", ayuda="p. ej. padre, hato o sexo", auto=False)],
                      multi=dict(clave="columnas", etiqueta="Variables numéricas", tipo="num", min=1, max=30), faltantes=True)]),
    "heredabilidad": dict(
        titulo="Heredabilidad (h²)", eyebrow="Parámetros genéticos", icono=":material/genetics:",
        sub="Proporción de la varianza fenotípica debida a efectos genéticos aditivos. Elija el método según la estructura familiar de sus datos.",
        metodos=[
            dict(id="medios_hermanos", nombre="Medios hermanos paternos", etiqueta="ANOVA",
                 resumen="Crías de varios padres. h² = 4σ²padre / σ²P. Con la columna animal y registros repetidos usa el modelo anidado y da también r.",
                 roles=[req("padre"), req("rasgo"), opc("animal", etiqueta="Animal (si hay registros repetidos)")], fijos=True, confianza=True),
            dict(id="hermanos_completos", nombre="Hermanos completos (padre / madre)", etiqueta="ANOVA",
                 resumen="Familias de hermanos completos anidadas en padres: separa los componentes de padre y de madre.",
                 roles=[req("padre"), req("madre"), req("rasgo"), opc("animal", etiqueta="Animal (si hay registros repetidos)")], fijos=True, confianza=True),
            dict(id="regresion", nombre="Regresión progenie–progenitor", etiqueta="Regresión",
                 resumen="Padres e hijos con fenotipo. h² = 2b (un progenitor) o b (promedio de padres).",
                 roles=[req("animal"), opc("padre"), opc("madre"), req("rasgo")], pedigri=True, confianza=True,
                 opciones=[dict(clave="tipo_regresion", etiqueta="Progenitor", tipo="select", defecto="padre", opciones=TIPO_REG)]),
            dict(id="reml_padre", nombre="REML · modelo de padre", etiqueta="Recomendado",
                 resumen="Exacto con datos desbalanceados y efectos fijos. Da errores estándar, prueba de significancia y valores de cría de los padres.",
                 roles=[req("padre"), req("rasgo"), opc("animal", etiqueta="Animal (ambiente permanente si hay repetidos)"),
                        opc("madre", etiqueta="Madre (efecto de madre, opcional)", auto=False)], fijos=True, confianza=True, opciones=[OPC_LRT]),
            dict(id="reml_animal", nombre="REML · modelo animal", etiqueta="Más completo",
                 resumen="Usa todo el pedigrí (matriz de parentesco A). Da h², r si hay repetidos y valores de cría de todos los animales.",
                 roles=[req("animal"), req("padre"), opc("madre"), req("rasgo")], fijos=True, pedigri=True, confianza=True, opciones=[OPC_LRT]),
        ]),
    "repetibilidad": dict(
        titulo="Repetibilidad (r)", eyebrow="Parámetros genéticos", icono=":material/repeat:",
        sub="Correlación entre mediciones repetidas del mismo animal (lactancias, partos, esquilas). Incluye la producción más probable (PMP).",
        metodos=[
            dict(id="anova", nombre="ANOVA de una vía (animales)", etiqueta="ANOVA", resumen="Varios registros por animal. r = σ²entre animales / σ²P.",
                 roles=[req("animal"), req("rasgo")], fijos=True, confianza=True),
            dict(id="anova_anidado", nombre="ANOVA anidado padre / animal", etiqueta="ANOVA",
                 resumen="Animales agrupados por padre con registros repetidos: estima r y h² a la vez.",
                 roles=[req("padre"), req("animal"), req("rasgo")], fijos=True, confianza=True),
            dict(id="reml", nombre="REML · modelo de repetibilidad", etiqueta="Recomendado",
                 resumen="Animal como efecto aleatorio (y padre si se indica). PMP calculada por BLUP.",
                 roles=[req("animal"), req("rasgo"), opc("padre", etiqueta="Padre (opcional: estima también h²)")], fijos=True, confianza=True),
            dict(id="reml_animal", nombre="REML · modelo animal + ambiente permanente", etiqueta="Más completo",
                 resumen="Con pedigrí: separa la varianza aditiva del ambiente permanente.",
                 roles=[req("animal"), req("padre"), opc("madre"), req("rasgo")], fijos=True, pedigri=True, confianza=True),
        ]),
    "correlacion": dict(
        titulo="Correlación genética (rG)", eyebrow="Parámetros genéticos", icono=":material/scatter_plot:",
        sub="Grado en que dos rasgos comparten genes: indica la respuesta correlacionada a la selección. También se calculan la repetibilidad "
            "de cada rasgo y las correlaciones fenotípica y ambiental.",
        metodos=[
            dict(id="ancova", nombre="Covarianza de medios hermanos", etiqueta="ANCOVA",
                 resumen="rG = σs(xy) / √(σ²s(x)·σ²s(y)). Admite varios rasgos a la vez.",
                 roles=[req("padre"), opc("animal", etiqueta="Animal (si hay registros repetidos)")],
                 multi=dict(clave="rasgos", etiqueta="Rasgos", tipo="num", min=2, max=8), fijos=True, confianza=True),
            dict(id="reml_padre", nombre="REML bivariado · modelo de padre", etiqueta="Recomendado",
                 resumen="Ajusta ambos rasgos a la vez con efectos fijos; errores estándar por la matriz de información.",
                 roles=[req("padre"), opc("animal", etiqueta="Animal (ambiente permanente si hay repetidos)")],
                 multi=dict(clave="rasgos", etiqueta="Rasgos", tipo="num", min=2, max=4), fijos=True, confianza=True),
            dict(id="reml_animal", nombre="REML bivariado · modelo animal", etiqueta="Más completo", resumen="Con el pedigrí completo (matriz A).",
                 roles=[req("animal"), req("padre"), opc("madre")], multi=dict(clave="rasgos", etiqueta="Rasgos", tipo="num", min=2, max=4),
                 fijos=True, pedigri=True, confianza=True),
            dict(id="cruzada", nombre="Covarianza cruzada progenitor–progenie", etiqueta="Regresión",
                 resumen="Requiere ambos rasgos medidos en padres e hijos.",
                 roles=[req("animal"), opc("padre"), opc("madre")], multi=dict(clave="rasgos", etiqueta="Rasgos (dos)", tipo="num", min=2, max=2),
                 pedigri=True, confianza=True,
                 opciones=[dict(clave="tipo_regresion", etiqueta="Progenitor", tipo="select", defecto="padre",
                                opciones=[("padre", "Padre"), ("madre", "Madre"), ("ambos", "Padre y madre")])]),
        ]),
    "pedigri": dict(
        titulo="Calidad y verificación del pedigrí", eyebrow="Pedigrí", icono=":material/verified:",
        sub="Revisa errores (duplicados, ciclos, sexo, fechas), mide la integridad del pedigrí por generación (MacCluer), cuenta los "
            "apareamientos entre parientes cercanos y prepara el pedigrí ordenado y renumerado.",
        metodos=[dict(id="pedigri", nombre="Verificación e integridad", roles=ROLES_PED + [opc("sexo"), opc("fecha"), opc("referencia")],
                      uno_de=["padre", "madre"],
                      opciones=[dict(clave="generaciones", etiqueta="Generaciones para medir la integridad", tipo="numero", defecto=5)])]),
    "endogamia": dict(
        titulo="Endogamia y tamaño efectivo", eyebrow="Diversidad genética", icono=":material/hub:",
        sub="Endogamia (F), relación promedio (AR), incremento individual de endogamia (ΔFi) y tamaño efectivo (Ne) por los métodos de ENDOG.",
        metodos=[dict(id="endogamia", nombre="F, AR, ΔFi y Ne", roles=ROLES_PED + [opc("sexo"), opc("fecha"), opc("referencia")], uno_de=["padre", "madre"],
                      opciones=[dict(clave="k_reciente", etiqueta="Endogamia reciente: generaciones a considerar", tipo="numero",
                                     ayuda="Vacío = no calcular. Con 3 a 5 se separa la endogamia reciente de la antigua."),
                                dict(clave="coascendencia", etiqueta="Ne por incremento de coascendencia entre pares (más lento)", tipo="check", defecto=False)])]),
    "fundadores": dict(
        titulo="Fundadores y ancestros", eyebrow="Diversidad genética", icono=":material/park:",
        sub="Origen de los genes: fundadores (fe), ancestros (fa, Boichard), genomas fundadores equivalentes (fg), rebaños fundadores, "
            "índice de conservación genética (GCI) y endogamia parcial (Lacy).",
        metodos=[dict(id="fundadores", nombre="Probabilidades de origen de los genes",
                      roles=ROLES_PED + [opc("referencia", ayuda="Vacío = animales con ambos padres conocidos (como ENDOG)."), opc("rebano", auto=False)],
                      uno_de=["padre", "madre"],
                      opciones=[dict(clave="gci", etiqueta="Calcular el índice de conservación genética (GCI)", tipo="check", defecto=True),
                                dict(clave="parcial_n", etiqueta="Endogamia parcial: número de fundadores y ancestros", tipo="numero", defecto=0,
                                     ayuda="0 = no calcular. Requiere poblaciones de referencia moderadas.")])]),
    "intervalos": dict(
        titulo="Intervalo generacional", eyebrow="Diversidad genética", icono=":material/schedule:",
        sub="Edad de los padres al nacer su progenie por las cuatro vías, intervalo generacional, Ne por la fecha de nacimiento y Ne de Hill.",
        metodos=[dict(id="intervalos", nombre="Intervalos y Ne demográfico", roles=ROLES_PED + [req("sexo"), req("fecha"), opc("referencia")],
                      uno_de=["padre", "madre"])]),
    "subpoblaciones": dict(
        titulo="Subpoblaciones y estadísticos F", eyebrow="Diversidad genética", icono=":material/groups:",
        sub="Estructura genética entre áreas, rebaños o líneas: FIS, FST y FIT de Wright, distancias de Nei y contribución a la diversidad.",
        metodos=[dict(id="subpoblaciones", nombre="Estadísticos F genealógicos", roles=ROLES_PED + [req("subpoblacion")], uno_de=["padre", "madre"],
                      opciones=[dict(clave="tamano_comun", etiqueta="Ajustar a un tamaño común de subpoblación", tipo="numero",
                                     ayuda="Vacío = sin ajuste (Bartolomé et al., 2010)."),
                                dict(clave="contribuciones", etiqueta="Contribución de cada subpoblación a la diversidad", tipo="check", defecto=True)])]),
    "rebanos": dict(
        titulo="Estructura de rebaños", eyebrow="Diversidad genética", icono=":material/agriculture:",
        sub="Clasifica los rebaños en núcleo, multiplicador, comercial o aislado (Vassallo) y calcula cuántos rebaños aportan padres (Robertson).",
        metodos=[dict(id="rebanos", nombre="Origen de los padres por rebaño", roles=ROLES_PED + [req("rebano"), opc("referencia")], uno_de=["padre", "madre"])]),
    "parentesco": dict(
        titulo="Parentesco y coascendencia", eyebrow="Pedigrí", icono=":material/family_restroom:",
        sub="Parentesco aditivo (a) y coascendencia (f = a/2) entre animales elegidos, o de un animal con todo el pedigrí.",
        metodos=[dict(id="parentesco", nombre="Matriz de parentesco", roles=ROLES_PED + [opc("sexo")], uno_de=["padre", "madre"],
                      opciones=[dict(clave="contra_todos", etiqueta="Comparar un animal contra todo el pedigrí", tipo="check", defecto=False),
                                dict(clave="sexo_opuesto", etiqueta="Solo animales del sexo opuesto (posibles parejas)", tipo="check", defecto=False),
                                dict(clave="animales", etiqueta="Animales", tipo="ids", requerido=True,
                                     ayuda="Separe los ID con comas, espacios o saltos de línea (2 a 80; 1 si compara contra todos).")])]),
    "apareamientos": dict(
        titulo="Apareamientos: endogamia esperada", eyebrow="Pedigrí", icono=":material/join:",
        sub="Coeficiente de endogamia que tendría la cría de cada pareja macho × hembra y el macho menos emparentado para cada hembra.",
        metodos=[dict(id="apareamientos", nombre="Matriz machos × hembras", roles=[req("id"), opc("padre"), opc("madre")], uno_de=["padre", "madre"],
                      opciones=[dict(clave="machos", etiqueta="Machos candidatos", tipo="ids", ayuda="Vacío = los 30 padres más recientes."),
                                dict(clave="hembras", etiqueta="Hembras candidatas", tipo="ids", ayuda="Vacío = las 60 madres más recientes."),
                                dict(clave="limite_f", etiqueta="Límite de endogamia aceptable", tipo="numero", defecto=0.0625,
                                     ayuda="0,0625 equivale a aparear primos hermanos.")])]),
    "blup": dict(
        titulo="Evaluación genética BLUP", eyebrow="Pedigrí", icono=":material/trending_up:",
        sub="Valores de cría (VC) de todos los animales del pedigrí con el modelo animal, a partir de una heredabilidad (y repetibilidad) conocida.",
        metodos=[dict(id="blup", nombre="BLUP · modelo animal", roles=[req("animal"), opc("padre"), opc("madre"), req("rasgo")], fijos=True, pedigri=True,
                      opciones=[dict(clave="h2", etiqueta="Heredabilidad (h²)", tipo="numero", requerido=True, sugerir="h2",
                                     ayuda="Entre 0 y 1. Puede usar la estimada en «Heredabilidad»."),
                                dict(clave="r", etiqueta="Repetibilidad (r)", tipo="numero", sugerir="r",
                                     ayuda="Solo si hay registros repetidos por animal (r ≥ h²).")])]),
}
