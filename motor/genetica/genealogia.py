"""Análisis basados en el pedigrí: verificación y estadísticas, endogamia, parentesco,
endogamia esperada de apareamientos y evaluación genética BLUP con parámetros conocidos."""
import math
from collections import Counter, defaultdict

import numpy as np

from . import reml
from .datos import ErrorDatos, Registros, a_numero, codificar, columna_ids, matriz_fija, residuos_fijos
from .parametros import _faltantes, _opciones, _registros, _repetidos, _tabla_vc
from .pedigri import Pedigri, normalizar_sexo
from .resultados import Resultado, histograma


def _pedigri_payload(payload, ctx):
    """Construye el pedigrí desde la hoja (roles id/padre/madre/sexo/fecha)."""
    tabla = payload.get("tabla") or {}
    roles = payload.get("roles") or {}
    vals = tabla.get("valores") or {}
    if not roles.get("id"):
        raise ErrorDatos("Asigne la columna de identificación del animal.")
    if not roles.get("padre") and not roles.get("madre"):
        raise ErrorDatos("Asigne al menos la columna de padre o la de madre.")
    ids = columna_ids(vals.get(roles["id"], []))
    n = len(ids)
    pad = columna_ids(vals[roles["padre"]]) if roles.get("padre") else [None] * n
    mad = columna_ids(vals[roles["madre"]]) if roles.get("madre") else [None] * n
    sexos = vals.get(roles["sexo"]) if roles.get("sexo") else None
    ped = Pedigri(ids, pad, mad, sexos=sexos, progreso=ctx.progreso, cancelado=ctx.cancelado)
    return ped, ids, vals, roles


def _anio(v):
    """Extrae el año de un valor de fecha (número, '2015', '2015-03-02', '02/03/2015')."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if v != v:
            return None
        x = float(v)
        if 1800 <= x <= 2200:
            return int(x)
        if 20000 <= x <= 80000:  # número de serie de Excel
            return 1900 + int((x - 1) / 365.2425)
        return None
    s = str(v).strip()
    import re
    m = re.search(r"(18|19|20|21)\d{2}", s)
    return int(m.group(0)) if m else None


def pedigri(payload, ctx):
    from . import diversidad as dv
    from .poblacion import DatosPedigri
    datos = DatosPedigri(payload, ctx)
    ped, div, roles = datos.ped, datos.div, datos.roles
    ids, vals = datos.ids, datos.vals
    F = div.F
    ctx.progreso(0.6, "Estadísticas e integridad del pedigrí…")
    est = ped.estadisticas()
    r = Resultado("pedigri", "pedigri", "Calidad y verificación del pedigrí",
                  f"{est['animales']} animales · {est['generaciones']} generaciones")

    # verificación de fechas: progenitor nacido el mismo día o después que la cría
    nac = datos.nacimiento()
    anios = {}
    if nac is not None:
        malos = []
        for k, i in enumerate(ped.ids):
            if nac[k] == nac[k]:
                anios[i] = dv.anio_de(nac[k])
            for p in (div.s[k], div.d[k]):
                if p >= 0 and nac[k] == nac[k] and nac[p] == nac[p] and nac[p] >= nac[k]:
                    malos.append(i)
        if malos:
            ped._aviso("error", f"{len(set(malos))} animales tienen un progenitor nacido el mismo día o después que ellos.",
                       sorted(set(malos), key=str))
        sin_fecha = int(np.sum(np.isnan(nac)))
        if sin_fecha:
            ped._aviso("info", f"{sin_fecha} animales no tienen una fecha de nacimiento reconocible.",
                       [i for k, i in enumerate(ped.ids) if nac[k] != nac[k]][:200])

    pct = lambda x: 100 * x / est["animales"] if est["animales"] else 0
    r.indicador("Animales en el pedigrí", est["animales"], decimales=0,
                descripcion=f"{est['agregados']} agregados como base (progenitores sin fila propia)")
    r.indicador("Endogamia media (F)", est["f_media"], decimales=4, destacado=True,
                descripcion=f"{est['consanguineos']} animales consanguíneos ({pct(est['consanguineos']):.1f} %); "
                            f"F máximo = {est['f_max']:.4f}")
    r.indicador("Generaciones completas equivalentes", est["ecg_media"], decimales=2,
                descripcion="Profundidad promedio del pedigrí conocido")
    if est["ne_realizado"] == est["ne_realizado"]:
        r.indicador("Ne realizado (todo el pedigrí)", est["ne_realizado"], decimales=1,
                    descripcion="Ne = 1/(2·ΔF̄) con ΔFi de todos los animales con GE > 1. Para la población de "
                                "referencia y otros métodos, vea «Endogamia y Ne».")
    if est["ne_sexos"] == est["ne_sexos"]:
        r.indicador("Ne por número de padres", est["ne_sexos"], decimales=1,
                    descripcion=f"4·Nm·Nf/(Nm+Nf) con {est['padres']} padres y {est['madres']} madres")

    errores = [a for a in ped.avisos if a["tipo"] == "error"]
    r.tabla("Verificación del pedigrí", [("tipo", "Tipo"), ("mensaje", "Hallazgo"), ("total", "Casos", "entero"),
                                         ("ejemplos", "Ejemplos")],
            [{"tipo": {"error": "Error", "advertencia": "Advertencia", "info": "Información"}[a["tipo"]],
              "mensaje": a["mensaje"], "total": a["total"], "ejemplos": ", ".join(a["ids"][:12])}
             for a in ped.avisos] or [{"tipo": "Correcto", "mensaje": "No se encontraron problemas.", "total": 0,
                                       "ejemplos": ""}], id="verificacion")
    if errores:
        r.nota(f"Se encontraron {len(errores)} tipos de error en el pedigrí; revise la tabla de verificación.",
               "advertencia")

    r.tabla("Completitud", [("dato", "Dato"), ("valor", "Animales", "entero"), ("pct", "%", "porcentaje", 1)], [
        {"dato": "Con padre y madre conocidos", "valor": est["ambos_padres"], "pct": pct(est["ambos_padres"])},
        {"dato": "Solo padre conocido", "valor": est["solo_padre"], "pct": pct(est["solo_padre"])},
        {"dato": "Solo madre conocida", "valor": est["solo_madre"], "pct": pct(est["solo_madre"])},
        {"dato": "Sin padres conocidos (base)", "valor": est["sin_padres"], "pct": pct(est["sin_padres"])},
        {"dato": "Padres distintos", "valor": est["padres"], "pct": None},
        {"dato": "Madres distintas", "valor": est["madres"], "pct": None},
    ], id="completitud")
    r.tabla("Estructura familiar", [("dato", "Dato"), ("valor", "Valor", "numero", 2)], [
        {"dato": "Crías por padre (promedio)", "valor": est["progenie_por_padre_media"]},
        {"dato": "Crías por padre (máximo)", "valor": est["progenie_por_padre_max"]},
        {"dato": "Crías por madre (promedio)", "valor": est["progenie_por_madre_media"]},
        {"dato": "Crías por madre (máximo)", "valor": est["progenie_por_madre_max"]},
        {"dato": "Generaciones máximas trazadas (promedio)", "valor": est["gen_max_media"]},
        {"dato": "Generaciones completas (promedio)", "valor": est["gen_completas_media"]},
        {"dato": "Generaciones completas equivalentes (promedio)", "valor": est["ecg_media"]},
        {"dato": "Incremento individual de endogamia (ΔF̄)", "valor": est["delta_f_media"]},
    ], id="estructura")
    r.tabla("Endogamia por generación", [("generacion", "Generación", "entero"), ("animales", "Animales", "entero"),
                                         ("consanguineos", "Consanguíneos", "entero"),
                                         ("f_media", "F medio", "numero", 4), ("f_max", "F máximo", "numero", 4)],
            est["por_generacion"], id="por_generacion",
            nota="Generación 0 = animales base; cada cría está una generación por debajo de su progenitor más joven.")
    r.grafico("lineas", "Endogamia media por generación",
              series=[{"nombre": "F medio", "puntos": [[g["generacion"], g["f_media"]] for g in est["por_generacion"]]},
                      {"nombre": "F máximo", "puntos": [[g["generacion"], g["f_max"]] for g in est["por_generacion"]]}],
              eje_x="Generación", eje_y="Coeficiente de endogamia")
    if anios:
        por_anio = defaultdict(list)
        for i, f in zip(ped.ids, F):
            if i in anios:
                por_anio[anios[i]].append(f)
        filas = [{"anio": a, "animales": len(v), "f_media": float(np.mean(v)), "f_max": float(np.max(v))}
                 for a, v in sorted(por_anio.items())]
        r.tabla("Endogamia por año de nacimiento", [("anio", "Año", "entero"), ("animales", "Animales", "entero"),
                                                    ("f_media", "F medio", "numero", 4),
                                                    ("f_max", "F máximo", "numero", 4)], filas, id="por_anio")
        if len(filas) > 1:
            r.grafico("lineas", "Endogamia media por año de nacimiento",
                      series=[{"nombre": "F medio", "puntos": [[f["anio"], f["f_media"]] for f in filas]}],
                      eje_x="Año", eje_y="F medio")
    r.tabla("Padres con más crías", [("animal", "Padre"), ("crias", "Crías", "entero")],
            [{"animal": a, "crias": c} for a, c in est["top_padres"]], id="top_padres")
    if est["top_madres"]:
        r.tabla("Madres con más crías", [("animal", "Madre"), ("crias", "Crías", "entero")],
                [{"animal": a, "crias": c} for a, c in est["top_madres"]], id="top_madres")
    consang = [(i, f) for i, f in zip(ped.ids, F) if f > 1e-12]
    consang.sort(key=lambda x: -x[1])
    if consang:
        r.tabla("Animales más consanguíneos", [("animal", "Animal"), ("padre", "Padre"), ("madre", "Madre"),
                                                ("f", "F", "numero", 4)],
                [{"animal": i, "padre": ped.filas[i][0] or "", "madre": ped.filas[i][1] or "", "f": f}
                 for i, f in consang[:100]], id="consanguineos")
        r.grafico("histograma", "Distribución de F (solo consanguíneos)", **histograma([f for _, f in consang]),
                  eje="Coeficiente de endogamia")
    ecg = est["ecg"]

    # integridad del pedigrí por generación (PedKnow) e índice de MacCluer et al. (1983)
    try:
        G = max(1, min(int((datos.op or {}).get("generaciones") or 5), 8))
    except (TypeError, ValueError):
        G = 5
    comp = div.completitud(G)
    ref = datos.referencia()
    filas = []
    for g in range(G):
        fila = {"gen": g + 1, "nombre": ["Padres", "Abuelos", "Bisabuelos", "Tatarabuelos"][g] if g < 4 else f"Generación {g + 1}",
                "todo": 100 * float(comp["por_generacion"][g].mean())}
        if ref is not None:
            fila["ref"] = 100 * float(comp["por_generacion"][g][ref].mean())
        filas.append(fila)
    cols = [("gen", "Generación", "entero"), ("nombre", "Ancestros"), ("todo", "% conocidos (todos)", "numero", 1)]
    if ref is not None:
        cols.append(("ref", "% conocidos (referencia)", "numero", 1))
    r.tabla("Integridad del pedigrí por generación", cols, filas, id="integridad",
            nota="Proporción de los 2^g ancestros de cada generación que son conocidos (tabla PedKnow de ENDOG).")
    series = [{"nombre": "Todos los animales", "puntos": [[f["gen"], f["todo"]] for f in filas]}]
    if ref is not None:
        series.append({"nombre": "Población de referencia", "puntos": [[f["gen"], f["ref"]] for f in filas]})
    r.grafico("lineas", "Ancestros conocidos por generación (%)", series=series, eje_x="Generación ancestral",
              eje_y="% de ancestros conocidos")
    pec = comp["pec"]
    r.indicador("Índice de integridad de MacCluer (PEC)", float(pec.mean()), decimales=3,
                descripcion=f"Media de {G} generaciones; referencia: {float(pec[ref].mean()):.3f}" if ref is not None
                else f"Media de {G} generaciones (0 = sin pedigrí, 1 = completo)")
    codigos = ["P", "M"]
    filas = []
    nombres_pos = {"P": "Padre", "M": "Madre", "PP": "Abuelo paterno", "PM": "Abuela paterna", "MP": "Abuelo materno",
                   "MM": "Abuela materna"}
    prev = codigos
    for g, conocidos in enumerate(comp["posiciones"], 1):
        cod = codigos if g == 1 else [c + "P" for c in prev] + [c + "M" for c in prev]
        # las columnas vienen como [línea paterna…, línea materna…]
        orden = [c for c in cod if c[0] == "P"] + [c for c in cod if c[0] == "M"]
        for j, c in enumerate(orden):
            filas.append({"gen": g, "posicion": nombres_pos.get(c, _nombre_posicion(c)), "codigo": c,
                          "pct": 100 * float(conocidos[:, j].mean())})
        prev = cod
    r.tabla("Ancestros conocidos por posición", [("gen", "Generación", "entero"), ("posicion", "Ancestro"),
                                                 ("codigo", "Ruta"), ("pct", "% conocidos", "numero", 1)], filas,
            id="posiciones", nota="Ruta: P = padre, M = madre, leída desde el animal (PM = madre del padre).")
    r.columna_nueva("PEC", "id", {i: float(v) for i, v in zip(ped.ids, pec)}, "Índice de integridad del pedigrí (MacCluer)")

    # apareamientos entre parientes cercanos (HighInbred de ENDOG)
    ap = div.apareamientos_cercanos()
    n = div.n
    r.tabla("Apareamientos entre parientes cercanos", [("tipo", "Apareamiento"), ("n", "Crías", "entero"),
                                                       ("pct", "% del pedigrí", "numero", 2), ("ejemplos", "Ejemplos")], [
        {"tipo": "Hermanos completos", "n": ap["hermanos_completos"], "pct": 100 * ap["hermanos_completos"] / n,
         "ejemplos": ", ".join(ped.ids[i] for i in ap["ejemplos"]["hc"][:8])},
        {"tipo": "Medios hermanos", "n": ap["medios_hermanos"], "pct": 100 * ap["medios_hermanos"] / n,
         "ejemplos": ", ".join(ped.ids[i] for i in ap["ejemplos"]["mh"][:8])},
        {"tipo": "Padre o madre con su cría", "n": ap["padre_hijo"], "pct": 100 * ap["padre_hijo"] / n,
         "ejemplos": ", ".join(ped.ids[i] for i in ap["ejemplos"]["pp"][:8])},
    ], id="cercanos", nota="Número de animales nacidos de cada tipo de apareamiento (crías con F ≥ 0,125 o 0,25).")

    # pedigrí ordenado y renumerado (como MiDef / el archivo secuencial de Pedigree Viewer)
    sexo = datos.sexo()
    r.d["hoja_nueva"] = {
        "nombre": "Pedigri_renumerado",
        "descripcion": "Pedigrí ordenado (padres antes que crías) con identificadores secuenciales 1…n",
        "columnas": ["I0", "P0", "M0", "animal", "padre", "madre", "sexo", "generacion", "F", "GE"],
        "filas": [[k + 1, int(div.s[k] + 1), int(div.d[k] + 1), ped.ids[k],
                   ped.ids[div.s[k]] if div.s[k] >= 0 else None, ped.ids[div.d[k]] if div.d[k] >= 0 else None,
                   {"M": 1, "H": 2}.get(sexo[k]), int(ped.generacion[k + 1]), round(float(F[k]), 8),
                   round(float(ecg[k]), 6)] for k in range(n)],
    }
    r.columna_nueva("F_endogamia", "id", {i: float(f) for i, f in zip(ped.ids, F)},
                    "Coeficiente de endogamia (Meuwissen y Luo, 1992)")
    r.columna_nueva("Gen_equivalentes", "id", {i: float(g) for i, g in zip(ped.ids, ecg)},
                    "Generaciones completas equivalentes")
    r.formula("Endogamia", "F_x = ½ a_padre,madre  (Meuwissen y Luo, 1992)")
    r.formula("Generaciones equivalentes", "GE = Σ (½)^n  sobre los ancestros conocidos")
    r.formula("Ne realizado", "ΔF_i = 1 − (1 − F_i)^(1/(GE_i − 1));  Ne = 1 / (2·ΔF̄)")
    r.formula("Índice de MacCluer", "PEC = 4·Cp·Cm/(Cp + Cm),  C = (1/d)·Σ a_g  (a_g: ancestros conocidos de la línea)")
    # pedigrí codificado para el diagrama (el navegador ya tiene el resto de campos)
    r.d["pedigri"] = {
        "ids": ped.ids, "padre": ped.s[1:].tolist(), "madre": ped.d[1:].tolist(),
        "generacion": ped.generacion[1:].tolist(), "f": F.tolist(),
        "agregado": [i in ped.agregados for i in ped.ids],
        "sexo": [_sexo_inferido(ped, i, k) for k, i in enumerate(ped.ids, 1)],
    }
    return r.json()


def _nombre_posicion(ruta):
    nombres = {1: ("Padre", "Madre"), 2: ("Abuelo", "Abuela"), 3: ("Bisabuelo", "Bisabuela")}
    g = len(ruta)
    base = nombres.get(g, ("Ancestro", "Ancestra"))[0 if ruta[-1] == "P" else 1]
    return f"{base} · línea {'paterna' if ruta[0] == 'P' else 'materna'}"


def _sexo_inferido(ped, i, codigo):
    s = ped.sexo_registrado.get(i)
    if s:
        return s
    return None


def _usados(ped):
    como_padre = set(int(x) for x in ped.s[1:] if x)
    como_madre = set(int(x) for x in ped.d[1:] if x)
    return como_padre, como_madre


def parentesco(payload, ctx):
    """Matriz de parentesco (A) y coascendencia entre animales elegidos, o de un animal con todos."""
    ped, ids, vals, roles = _pedigri_payload(payload, ctx)
    op = _opciones(payload)
    elegidos = [str(x).strip() for x in (op.get("animales") or []) if str(x).strip()]
    faltan = [x for x in elegidos if x not in ped.codigo]
    if faltan:
        raise ErrorDatos("No están en el pedigrí: " + ", ".join(faltan[:10]))
    ctx.progreso(0.3, "Calculando parentescos…")
    F = ped.F()
    if len(elegidos) == 1 or op.get("contra_todos"):
        foco = elegidos[0] if elegidos else None
        if not foco:
            raise ErrorDatos("Indique un animal.")
        from .diversidad import Diversidad
        from .poblacion import sexo_por_animal
        col = ped.columnas_a([ped.codigo[foco]])[:, 0]
        k_foco = ped.codigo[foco] - 1
        div = Diversidad(ped, cancelado=ctx.cancelado)
        col_sexo = None
        if roles.get("sexo") and roles["sexo"] in vals:
            fila = {}
            for k, i in enumerate(ids):
                if i is not None and i not in fila:
                    fila[i] = k
            col_sexo = [vals[roles["sexo"]][fila[i]] if i in fila else None for i in ped.ids]
        sexo = sexo_por_animal(div, col_sexo)
        ar = div.ar()
        opuesto = bool(op.get("sexo_opuesto")) and sexo[k_foco] in ("M", "H")
        filas = []
        for k, i in enumerate(ped.ids):
            if i == foco:
                continue
            if opuesto:
                if sexo[k] not in ("M", "H") or sexo[k] == sexo[k_foco]:
                    continue
            elif col[k] <= 1e-12:
                continue
            filas.append({"animal": i, "sexo": {"M": "Macho", "H": "Hembra"}.get(sexo[k], ""), "a": float(col[k]),
                          "coasc": float(col[k] / 2), "ar": float(ar[k]), "relacion": _nombre_relacion(col[k])})
        filas.sort(key=lambda f: (f["coasc"], f["ar"]) if opuesto else (-f["a"],))
        titulo = (f"Posibles parejas de {foco} y endogamia de su cría" if opuesto
                  else f"Parentesco de {foco} con todo el pedigrí")
        r = Resultado("parentesco", "uno_contra_todos", titulo,
                      f"{len(filas)} animales del sexo opuesto" if opuesto else f"{len(filas)} parientes con parentesco mayor que cero")
        r.indicador("Endogamia de " + foco, float(F[k_foco]), decimales=4)
        r.indicador("Relación promedio (AR) de " + foco, float(ar[k_foco]), decimales=4,
                    descripcion="Representación del animal en todo el pedigrí; los de menor AR conservan más diversidad")
        r.indicador("Parientes", int(np.sum(col > 1e-12)) - 1, decimales=0)
        r.indicador("Parentesco medio con la población", float(np.mean(np.delete(col, k_foco))),
                    decimales=4, descripcion="Contribución genética media del animal a la población")
        cols = [("animal", "Animal"), ("sexo", "Sexo"),
                ("coasc", "F de la cría (coascendencia)" if opuesto else "Coascendencia (f = a/2)", "numero", 4),
                ("a", "Parentesco aditivo (a)", "numero", 4), ("ar", "AR", "numero", 4),
                ("relacion", "Equivalente aproximado")]
        r.tabla("Posibles parejas ordenadas por endogamia esperada" if opuesto else "Parientes ordenados por parentesco",
                cols, filas[:200_000], id="parientes",
                nota=("Equivale a la tabla Parent de ENDOG: la coascendencia con el animal es la endogamia que tendría "
                      "su cría. Ante empates, se prefieren parejas con menor AR.") if opuesto else None)
        r.columna_nueva(f"Parentesco_{foco}", "id", {i: float(col[k]) for k, i in enumerate(ped.ids)},
                        f"Parentesco aditivo con {foco}")
        return r.json()
    if len(elegidos) > 80:
        raise ErrorDatos("Elija como máximo 80 animales para la matriz de parentesco.")
    if len(elegidos) < 2:
        raise ErrorDatos("Elija al menos dos animales.")
    cods = [ped.codigo[x] for x in elegidos]
    A = ped.submatriz_a(cods)
    r = Resultado("parentesco", "matriz", "Matriz de parentesco aditivo (A)", f"{len(elegidos)} animales")
    filas = []
    for i, a in enumerate(elegidos):
        fila = {"animal": a}
        for j in range(len(elegidos)):
            fila[f"c{j}"] = float(A[i, j])
        filas.append(fila)
    r.tabla("Matriz A (diagonal = 1 + F)", [("animal", "")] + [(f"c{j}", e, "numero", 4) for j, e in enumerate(elegidos)],
            filas, id="matriz_a")
    pares = []
    for i in range(len(elegidos)):
        for j in range(i + 1, len(elegidos)):
            pares.append({"a1": elegidos[i], "a2": elegidos[j], "a": float(A[i, j]), "coasc": float(A[i, j] / 2),
                          "relacion": _nombre_relacion(A[i, j])})
    pares.sort(key=lambda f: -f["a"])
    r.tabla("Pares ordenados por parentesco", [("a1", "Animal 1"), ("a2", "Animal 2"), ("a", "Parentesco (a)", "numero", 4),
                                               ("coasc", "Coascendencia (f)", "numero", 4),
                                               ("relacion", "Equivalente aproximado")], pares, id="pares")
    if len(elegidos) <= 40:
        r.grafico("matriz_calor", "Parentesco aditivo", nombres=elegidos, valores=A.tolist(), rango=[0, max(1.0, float(A.max()))])
    off = A[~np.eye(len(elegidos), dtype=bool)]
    r.indicador("Parentesco medio entre ellos", float(off.mean()), decimales=4, destacado=True)
    r.indicador("Coascendencia media", float(off.mean() / 2), decimales=4,
                descripcion="Endogamia esperada si se aparearan al azar entre sí")
    r.indicador("Endogamia media de los elegidos", float(np.mean(np.diag(A) - 1)), decimales=4)
    r.formula("Parentesco y coascendencia", "a_xy = 2 f_xy;  F de la cría de x × y = f_xy")
    return r.json()


def _nombre_relacion(a):
    tabla = [(1.0, "Mismo animal / gemelo idéntico"), (0.5, "Padre–hijo o hermanos completos"),
             (0.25, "Medios hermanos, abuelo–nieto, tío–sobrino"), (0.125, "Primos hermanos, bisabuelo"),
             (0.0625, "Primos segundos aprox.")]
    mejor = min(tabla, key=lambda x: abs(math.log(max(a, 1e-6)) - math.log(x[0])))
    return mejor[1] if a >= 0.04 else "Lejano"


def apareamientos(payload, ctx):
    """Endogamia esperada de la progenie para machos × hembras candidatos."""
    ped, ids, vals, roles = _pedigri_payload(payload, ctx)
    op = _opciones(payload)
    machos = [str(x).strip() for x in (op.get("machos") or []) if str(x).strip()]
    hembras = [str(x).strip() for x in (op.get("hembras") or []) if str(x).strip()]
    if not machos or not hembras:
        como_padre, como_madre = _usados(ped)
        if not machos:
            machos = [ped.ids[c - 1] for c in sorted(como_padre)][-30:]
        if not hembras:
            hembras = [ped.ids[c - 1] for c in sorted(como_madre)][-60:]
    if not machos:
        raise ErrorDatos("No hay machos candidatos: escríbalos en «Machos candidatos» o asigne la columna de padre.")
    if not hembras:
        raise ErrorDatos("No hay hembras candidatas: escríbalas en «Hembras candidatas» o asigne la columna de madre.")
    faltan = [x for x in machos + hembras if x not in ped.codigo]
    if faltan:
        raise ErrorDatos("No están en el pedigrí: " + ", ".join(faltan[:10]))
    if len(machos) > 100 or len(hembras) > 1000:
        raise ErrorDatos("Máximo 100 machos y 1000 hembras por consulta.")
    ctx.progreso(0.3, "Calculando la coascendencia de cada pareja…")
    cols = ped.columnas_a([ped.codigo[m] for m in machos])  # n x machos
    idx_h = np.array([ped.codigo[h] - 1 for h in hembras])
    Fprog = cols[idx_h, :].T / 2  # machos x hembras
    try:
        limite = float(op.get("limite_f") or 0.0625)
    except (TypeError, ValueError):
        limite = 0.0625
    r = Resultado("apareamientos", "apareamientos", "Endogamia esperada de la progenie",
                  f"{len(machos)} machos × {len(hembras)} hembras")
    filas = []
    for i, m in enumerate(machos):
        fila = {"macho": m}
        for j in range(len(hembras)):
            fila[f"h{j}"] = float(Fprog[i, j])
        filas.append(fila)
    if len(hembras) <= 60:
        r.tabla("F esperado de la cría (macho × hembra)", [("macho", "Macho \\ Hembra")]
                + [(f"h{j}", h, "numero", 4) for j, h in enumerate(hembras)], filas, id="matriz_f")
        if len(machos) <= 40:
            r.grafico("matriz_calor", "Endogamia esperada de la cría", nombres=machos, nombres_col=hembras,
                      valores=Fprog.tolist(), rango=[0, max(0.25, float(Fprog.max()))])
    mejores = []
    for j, h in enumerate(hembras):
        orden = np.argsort(Fprog[:, j])
        mejor = machos[orden[0]]
        mejores.append({"hembra": h, "macho": mejor, "f": float(Fprog[orden[0], j]),
                        "alternativa": machos[orden[1]] if len(orden) > 1 else "",
                        "f_alt": float(Fprog[orden[1], j]) if len(orden) > 1 else None,
                        "riesgosos": int(np.sum(Fprog[:, j] > limite))})
    r.tabla("Macho con menor endogamia esperada para cada hembra",
            [("hembra", "Hembra"), ("macho", "Macho sugerido"), ("f", "F esperado", "numero", 4),
             ("alternativa", "Alternativa"), ("f_alt", "F alternativa", "numero", 4),
             ("riesgosos", f"Machos con F > {limite:g}", "entero")], mejores, id="sugeridos")
    r.indicador("F esperado medio (todas las parejas)", float(Fprog.mean()), decimales=4)
    r.indicador("Parejas por encima del límite", int(np.sum(Fprog > limite)), decimales=0,
                descripcion=f"Límite F = {limite:g} (≈ cruce de primos hermanos = 0,0625)")
    r.indicador("F esperado con el macho sugerido", float(np.mean([m["f"] for m in mejores])), decimales=4,
                destacado=True)
    r.nota("La sugerencia solo minimiza la endogamia; combínela con los valores de cría para decidir.")
    r.formula("Endogamia de la cría", "F_cría = f_macho,hembra = ½ a_macho,hembra")
    return r.json()


def blup(payload, ctx):
    """Evaluación genética BLUP (modelo animal) con h² (y r) conocidos."""
    op = _opciones(payload)
    try:
        h2 = float(op.get("h2"))
    except (TypeError, ValueError):
        raise ErrorDatos("Indique la heredabilidad (h²) a usar en la evaluación.")
    if not 0 < h2 < 1:
        raise ErrorDatos("La heredabilidad debe estar entre 0 y 1 (sin incluirlos).")
    rep = op.get("r")
    rep = float(rep) if rep not in (None, "") else None
    reg = _registros(payload, ["rasgo"], ["animal"], ["padre", "madre"])
    nombre = reg.nombres["rasgo"]
    X, nombres_x, notas_x = matriz_fija(reg)
    y = reg.rasgos["rasgo"]
    # varianza fenotípica corregida por efectos fijos
    res_y = residuos_fijos(X, y)[:, 0]
    var_p = float(res_y @ res_y) / max(len(y) - X.shape[1], 1)
    if not var_p > 0:
        var_p = float(np.var(y, ddof=1))
    from .parametros import _pedigri_de_registros
    ctx.progreso(0.1, "Armando el pedigrí y A⁻¹…")
    ped = _pedigri_de_registros(reg, payload, ctx)
    Ainv = ped.a_inversa()
    codigos = np.array([ped.codigo[a] - 1 for a in reg.ids["animal"]])
    aleatorios = [reml.Aleatorio("animal", codigos, ped.n, Ainv=Ainv, etiquetas=ped.ids)]
    repetidos = _repetidos(reg.ids["animal"])[0] > 0
    var_a = h2 * var_p
    G = [np.array([[var_a]])]
    if repetidos:
        if rep is None:
            raise ErrorDatos("Hay registros repetidos por animal: indique también la repetibilidad (r).")
        if not h2 <= rep < 1:
            raise ErrorDatos("La repetibilidad debe ser mayor o igual que h² y menor que 1.")
        pc, pl = codificar(reg.ids["animal"])
        aleatorios.append(reml.Aleatorio("pe", pc, len(pl)))
        G.append(np.array([[max((rep - h2) * var_p, 1e-8 * var_p)]]))
        var_e = (1 - rep) * var_p
    else:
        var_e = (1 - h2) * var_p
    modelo = reml.ModeloMixto(y, X, aleatorios)
    ctx.progreso(0.4, "Resolviendo las ecuaciones del modelo mixto…")
    esc = modelo.de[0] ** 2
    R0 = np.array([[var_e / esc]])
    Gs = [g / esc for g in G]
    calcular_pev = ped.n <= 20000
    sols = modelo.soluciones(R0, Gs, varianza_error=calcular_pev)
    r = Resultado("blup", "blup", "Evaluación genética BLUP · modelo animal",
                  f"{nombre} · {reg.n} registros · {ped.n} animales · h² = {h2:g}"
                  + (f" · r = {rep:g}" if repetidos else ""))
    r.nota(reg.nota_descartados(), "advertencia")
    for n in notas_x:
        r.nota(n, "advertencia")
    ctx.progreso(0.8, "Ordenando resultados…")
    filas = _tabla_vc(r, ped, reg, sols, 0, var_a, nombre, "animal")
    vcs = np.array([f["vc"] for f in filas])
    exact = [f["exactitud"] for f in filas if f["exactitud"] is not None]
    r.indicador("Animales evaluados", ped.n, decimales=0, descripcion=f"{len(set(reg.ids['animal']))} con registros")
    r.indicador("Varianza aditiva usada (σ²A)", var_a, decimales=4, descripcion=f"σ²P = {var_p:.4g} (corregida por efectos fijos)")
    r.indicador("DE de los valores de cría", float(vcs.std(ddof=1)) if len(vcs) > 1 else None, decimales=4)
    if exact:
        r.indicador("Exactitud media", float(np.mean(exact)), decimales=3, destacado=True)
    _fijos = sols["fijos"][:, 0]
    r.tabla("Soluciones de efectos fijos (BLUE)", [("efecto", "Efecto"), ("solucion", "Solución", "numero", 4)],
            [{"efecto": nombres_x[i], "solucion": float(_fijos[i])} for i in range(min(len(nombres_x), 300))], id="fijos")
    r.formula("Modelo animal", "y = Xb + Za + " + ("Wpe + " if repetidos else "") + "e")
    r.formula("Ecuaciones del modelo mixto", "[X'X  X'Z; Z'X  Z'Z + A⁻¹·α] [b; a] = [X'y; Z'y],  α = σ²e/σ²A")
    r.formula("Exactitud", "r_TI = √(1 − PEV / σ²A)")
    r.parametro("h2", h2)
    if repetidos:
        r.parametro("r", rep)
    return r.json()
