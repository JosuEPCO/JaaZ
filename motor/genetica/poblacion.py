"""Análisis demográfico y de diversidad genética del pedigrí (herramientas de ENDOG):
endogamia y tamaño efectivo, fundadores y ancestros, intervalo generacional,
subpoblaciones (estadísticos F) y estructura de rebaños."""
import math
from collections import Counter, defaultdict

import numpy as np

from . import diversidad as dv
from .datos import ErrorDatos, a_id, columna_ids
from .pedigri import Pedigri, normalizar_sexo
from .resultados import Resultado, histograma

LIMITE_FILAS = 200_000


# ================================================================== carga común
class DatosPedigri:
    """Pedigrí de la hoja + columnas auxiliares alineadas con el orden del pedigrí."""

    def __init__(self, payload, ctx):
        tabla = payload.get("tabla") or {}
        self.roles = payload.get("roles") or {}
        self.op = payload.get("opciones") or {}
        vals = tabla.get("valores") or {}
        self.vals = vals
        if not self.roles.get("id"):
            raise ErrorDatos("Asigne la columna de identificación del animal.")
        if not self.roles.get("padre") and not self.roles.get("madre"):
            raise ErrorDatos("Asigne al menos la columna de padre o la de madre.")
        ids = columna_ids(vals.get(self.roles["id"], []))
        n = len(ids)
        pad = columna_ids(vals[self.roles["padre"]]) if self.roles.get("padre") else [None] * n
        mad = columna_ids(vals[self.roles["madre"]]) if self.roles.get("madre") else [None] * n
        sexos = vals.get(self.roles["sexo"]) if self.roles.get("sexo") else None
        ctx.progreso(0.03, "Ordenando y verificando el pedigrí…")
        self.ped = Pedigri(ids, pad, mad, sexos=sexos, progreso=ctx.progreso, cancelado=ctx.cancelado)
        ctx.progreso(0.25, "Endogamia y generaciones…")
        self.div = dv.Diversidad(self.ped, cancelado=ctx.cancelado, progreso=ctx.progreso)
        self.fila = {}
        for k, i in enumerate(ids):
            if i is not None and i not in self.fila:
                self.fila[i] = k
        self.n = self.ped.n
        self.ids = self.ped.ids

    def columna(self, rol):
        nombre = self.roles.get(rol)
        if not nombre or nombre not in self.vals:
            return None
        col = self.vals[nombre]
        return [col[self.fila[i]] if i in self.fila else None for i in self.ids]

    def sexo(self):
        return sexo_por_animal(self.div, self.columna("sexo"))

    def nacimiento(self):
        col = self.columna("fecha")
        if col is None:
            return None
        return np.array([dv.anio_decimal(v) for v in col])

    def referencia(self):
        """Máscara de la población de referencia (columna con 1) o None."""
        col = self.columna("referencia")
        if col is None:
            return None
        ref = np.array([dv.es_referencia(v) for v in col])
        if not ref.any():
            raise ErrorDatos(f"La columna «{self.roles['referencia']}» no tiene ningún animal marcado con 1.")
        return ref

    def grupos(self, rol):
        col = self.columna(rol)
        if col is None:
            return None
        return [a_id(v) for v in col]

    def nombre(self, i):
        return self.ids[i] if i >= 0 else ""


def sexo_por_animal(div, columna=None):
    """'M', 'H' o None por animal (orden del pedigrí). Sin dato, se infiere por su uso
    como padre o como madre."""
    out = np.array([None] * div.n, dtype=object)
    if columna is not None:
        out[:] = [normalizar_sexo(v) for v in columna]
    s, d = div.s, div.d
    for i in np.unique(s[s >= 0]):
        if out[i] is None:
            out[i] = "M"
    for i in np.unique(d[d >= 0]):
        if out[i] is None:
            out[i] = "H"
    return out


def _ref_o_defecto(datos, defecto):
    ref = datos.referencia()
    if ref is not None:
        return ref, f"animales con 1 en «{datos.roles['referencia']}» ({int(ref.sum())})"
    if defecto == "ambos":
        div = datos.div
        ref = (div.s >= 0) & (div.d >= 0)
        if not ref.any():
            raise ErrorDatos("Ningún animal tiene ambos padres conocidos: defina una población de referencia.")
        return ref, f"animales con ambos padres conocidos ({int(ref.sum())})"
    return np.ones(datos.n, bool), f"todos los animales ({datos.n})"


def _pct(x):
    return 100 * x if x == x else None


def _nivel_ne(ne):
    if ne is None or ne != ne:
        return None
    if ne < 50:
        return "Crítico"
    if ne < 100:
        return "En riesgo"
    return "Adecuado"


NOTA_NE = ("Referencia FAO: Ne < 50 indica riesgo inmediato por endogamia; se recomienda Ne ≥ 50 a corto plazo y "
           "≥ 500 para conservar el potencial evolutivo. Los métodos difieren en sus supuestos: compare varios y "
           "prefiera los basados en ΔFi o en coascendencias con pedigríes profundos.")


# ================================================================== ENDOGAMIA Y Ne
def endogamia(payload, ctx):
    datos = DatosPedigri(payload, ctx)
    div, op = datos.div, datos.op
    n = div.n
    F = div.F
    ctx.progreso(0.35, "Relación promedio (AR)…")
    ar = div.ar()
    df9, df8 = div.delta_f(2009), div.delta_f(2008)
    hijos = div.progenie()
    ref, desc_ref = _ref_o_defecto(datos, "todos")
    r = Resultado("endogamia", "endogamia", "Endogamia y tamaño efectivo de la población",
                  f"{n} animales · población de referencia: {desc_ref}")

    reg = div.ne_regresion_generaciones()
    ne09 = div.ne_desde_media(df9[ref])
    ne08 = div.ne_desde_media(df8[ref])
    ne_ref_ge = div.ne_regresion_ref(ref)
    consang = F > 1e-12
    r.indicador("Endogamia media (F)", float(F.mean()), decimales=4, destacado=True,
                descripcion=f"{int(consang.sum())} animales consanguíneos ({100 * consang.mean():.1f} %); "
                            f"F máximo = {F.max():.4f}; F medio de los consanguíneos = "
                            f"{(F[consang].mean() if consang.any() else 0):.4f}")
    r.indicador("Relación promedio (AR)", float(ar.mean()), decimales=4,
                descripcion="Probabilidad de que un alelo tomado al azar de la población pertenezca al animal")
    r.indicador("Ne realizado (ΔFi)", ne09["ne"], ne09["ee"], decimales=1, nivel=_nivel_ne(ne09["ne"]),
                descripcion=f"Gutiérrez et al. (2009), {ne09['n']} animales de la referencia con GE > 1")
    r.indicador("Generaciones completas equivalentes", float(div.ecg.mean()), decimales=2,
                descripcion=f"Máximas {div.gmax.mean():.2f} · completas {div.gcom.mean():.2f}")

    # resumen tipo Populat.txt
    filas = [
        {"dato": "Número de animales", "valor": n, "dF": None, "ne": None},
        {"dato": "Endogamia media (%)", "valor": 100 * F.mean(), "dF": None, "ne": None},
        {"dato": "Relación promedio media (%)", "valor": 100 * ar.mean(), "dF": None, "ne": None},
        {"dato": "Generaciones máximas trazadas (media)", "valor": reg["maximas"]["media"],
         "dF": _pct(reg["maximas"]["b"]), "ne": reg["maximas"]["ne"]},
        {"dato": "Generaciones completas (media)", "valor": reg["completas"]["media"],
         "dF": _pct(reg["completas"]["b"]), "ne": reg["completas"]["ne"]},
        {"dato": "Generaciones completas equivalentes (media)", "valor": reg["equivalentes"]["media"],
         "dF": _pct(reg["equivalentes"]["b"]), "ne": reg["equivalentes"]["ne"]},
    ]
    r.tabla("Resumen de la población", [("dato", "Dato"), ("valor", "Valor", "numero", 3),
                                         ("dF", "ΔF por generación (%)", "numero", 3), ("ne", "Ne", "numero", 2)],
            filas, id="resumen",
            nota="ΔF = pendiente de la regresión de F individual sobre el número de generaciones; Ne = 1/(2ΔF).")

    # tamaño efectivo por métodos
    metodos = [
        {"metodo": "Regresión de F sobre generaciones máximas", "poblacion": "Todo el pedigrí", "n": n,
         "dF": reg["maximas"]["b"], "ne": reg["maximas"]["ne"], "ee": None, "ref": "Gutiérrez et al. (2003)"},
        {"metodo": "Regresión de F sobre generaciones completas", "poblacion": "Todo el pedigrí", "n": n,
         "dF": reg["completas"]["b"], "ne": reg["completas"]["ne"], "ee": None, "ref": "Gutiérrez et al. (2003)"},
        {"metodo": "Regresión de F sobre generaciones equivalentes", "poblacion": "Todo el pedigrí", "n": n,
         "dF": reg["equivalentes"]["b"], "ne": reg["equivalentes"]["ne"], "ee": None, "ref": "Gutiérrez et al. (2003)"},
        {"metodo": "Incremento individual ΔFi (t = GE − 1)", "poblacion": "Referencia", "n": ne09["n"],
         "dF": ne09["media"], "ne": ne09["ne"], "ee": ne09["ee"], "ref": "Gutiérrez et al. (2009)"},
        {"metodo": "Incremento individual ΔFi (t = GE)", "poblacion": "Referencia", "n": ne08["n"],
         "dF": ne08["media"], "ne": ne08["ne"], "ee": ne08["ee"], "ref": "Gutiérrez et al. (2008)"},
        {"metodo": "Regresión de F sobre GE en la referencia", "poblacion": "Referencia", "n": int(ref.sum()),
         "dF": ne_ref_ge.get("delta_f"), "ne": ne_ref_ge["ne"], "ee": None, "ref": "ENDOG"},
    ]
    if op.get("coascendencia"):
        ctx.progreso(0.55, "Ne por incremento de coascendencia entre pares (puede tardar)…")
        nc = div.ne_coascendencia(ref)
        metodos.append({"metodo": "Incremento de coascendencia entre pares Δc", "poblacion": "Referencia",
                        "n": nc["n"], "dF": nc.get("media"), "ne": nc["ne"], "ee": nc["ee"],
                        "ref": "Cervantes et al. (2010)"})
        if nc.get("muestreo"):
            r.nota(f"Para Δc se usó una muestra aleatoria de {nc['n']} animales de la referencia.", "info")
        r.indicador("Ne por coascendencia (Δc)", nc["ne"], nc["ee"], decimales=1, nivel=_nivel_ne(nc["ne"]),
                    descripcion=f"{nc['pares']:,} pares de la referencia".replace(",", " "))
    nac = datos.nacimiento()
    L = None
    if nac is not None and np.sum(~np.isnan(nac)) > 2:
        sexo = datos.sexo()
        _, L = dv.intervalos(div, nac, sexo, solo_reproductores=True)
        for etiqueta, mascara in (("Todo el pedigrí", np.ones(n, bool)), ("Referencia", ref)):
            if etiqueta == "Referencia" and ref.all():
                continue
            nf = dv.ne_fecha(div, nac, L, mascara)
            metodos.append({"metodo": "Regresión de F sobre la fecha de nacimiento", "poblacion": etiqueta,
                            "n": nf["n"], "dF": nf.get("delta_f"), "ne": nf["ne"], "ee": None, "ref": "Gutiérrez et al. (2003)"})
            metodos.append({"metodo": "Regresión de ln(1 − F) sobre la fecha", "poblacion": etiqueta,
                            "n": nf["n"], "dF": None, "ne": nf["ne_log"], "ee": None, "ref": "Pérez-Enciso (1995)"})
    padres_ref = set(div.s[ref & (div.s >= 0)]) if ref.any() else set()
    madres_ref = set(div.d[ref & (div.d >= 0)]) if ref.any() else set()
    if padres_ref and madres_ref:
        nm, nf_ = len(padres_ref), len(madres_ref)
        metodos.append({"metodo": f"Número de padres ({nm}) y madres ({nf_}) de la referencia", "poblacion": "Referencia",
                        "n": nm + nf_, "dF": None, "ne": 4 * nm * nf_ / (nm + nf_), "ee": None, "ref": "Wright (1931)"})
    for m in metodos:
        m["nivel"] = _nivel_ne(m["ne"])
    r.tabla("Tamaño efectivo (Ne) por distintos métodos",
            [("metodo", "Método"), ("poblacion", "Población"), ("n", "n", "entero"), ("dF", "ΔF", "numero", 5),
             ("ne", "Ne", "numero", 2), ("ee", "EE", "numero", 2), ("nivel", "Situación"), ("ref", "Referencia")],
            metodos, id="ne", nota=NOTA_NE + (f" Intervalo generacional usado: {L:.2f} años." if L else ""))
    r.grafico("barras", "Tamaño efectivo según el método",
              barras=[{"etiqueta": (m["metodo"].replace("Regresión de ", "Reg. ").replace("Incremento ", "Incr. ")[:34]
                                    + (" (ref.)" if m["poblacion"] == "Referencia" else "")), "valor": m["ne"]}
                      for m in metodos if m["ne"] == m["ne"] and m["ne"] is not None and m["ne"] < 1e5],
              eje="Ne")

    # tablas por generación (PorG / PorC)
    cols = [("generacion", "Generación", "entero"), ("n", "Animales", "entero"), ("f", "F medio", "numero", 4),
            ("pct_consanguineos", "% consanguíneos", "numero", 1), ("f_consanguineos", "F de consanguíneos", "numero", 4),
            ("ar", "AR medio", "numero", 4), ("delta_f", "ΔF", "numero", 4), ("ne", "Ne", "numero", 1)]
    por_max = div.ne_por_generacion(div.gmax)
    por_com = div.ne_por_generacion(div.gcom)
    r.tabla("Endogamia por generaciones máximas trazadas", cols, por_max, id="por_max",
            nota="Ne = 1/(2ΔF) con ΔF = (Ft − Ft−1)/(1 − Ft−1), solo cuando F aumenta.")
    r.tabla("Endogamia por generaciones completas", cols, por_com, id="por_com")
    r.grafico("lineas", "Endogamia y relación promedio por generaciones máximas",
              series=[{"nombre": "F medio", "puntos": [[f["generacion"], f["f"]] for f in por_max]},
                      {"nombre": "AR medio", "puntos": [[f["generacion"], f["ar"]] for f in por_max]}],
              eje_x="Generaciones máximas trazadas", eje_y="Coeficiente")
    if nac is not None:
        por_anio = defaultdict(list)
        for i in range(n):
            if nac[i] == nac[i]:
                por_anio[dv.anio_de(nac[i])].append(i)
        filas = []
        for a in sorted(por_anio):
            ix = np.array(por_anio[a])
            filas.append({"anio": a, "n": len(ix), "f": float(F[ix].mean()), "ar": float(ar[ix].mean()),
                          "pct": 100 * float((F[ix] > 1e-12).mean()), "df": float(np.nanmean(df9[ix])) if np.any(~np.isnan(df9[ix])) else None})
        r.tabla("Endogamia por año de nacimiento", [("anio", "Año", "entero"), ("n", "Animales", "entero"),
                                                    ("f", "F medio", "numero", 4), ("pct", "% consanguíneos", "numero", 1),
                                                    ("ar", "AR medio", "numero", 4), ("df", "ΔFi medio", "numero", 4)],
                filas, id="por_anio")
        if len(filas) > 1:
            r.grafico("lineas", "Evolución de la endogamia por año de nacimiento",
                      series=[{"nombre": "F medio", "puntos": [[f["anio"], f["f"]] for f in filas]},
                              {"nombre": "AR medio", "puntos": [[f["anio"], f["ar"]] for f in filas]}],
                      eje_x="Año de nacimiento", eje_y="Coeficiente")

    # endogamia reciente
    Fk = None
    try:
        k = int(op.get("k_reciente") or 0)
    except (TypeError, ValueError):
        k = 0
    if k > 0:
        k = min(k, 10)
        ctx.progreso(0.75, f"Endogamia reciente ({k} generaciones)…")
        Fk = div.endogamia_reciente(k, np.arange(n))
        r.tabla(f"Endogamia reciente ({k} generaciones)", [("dato", "Dato"), ("todo", "Todo el pedigrí", "numero", 4),
                                                            ("ref", "Referencia", "numero", 4)], [
            {"dato": "F con todo el pedigrí", "todo": float(F.mean()), "ref": float(F[ref].mean())},
            {"dato": f"F con {k} generaciones (reciente)", "todo": float(Fk.mean()), "ref": float(Fk[ref].mean())},
            {"dato": "F antigua (diferencia)", "todo": float((F - Fk).mean()), "ref": float((F - Fk)[ref].mean())},
        ], id="reciente", nota="La endogamia reciente se debe a apareamientos de las últimas generaciones; la antigua "
                               "proviene de ancestros lejanos y suele estar más purgada de alelos deletéreos.")
        r.columna_nueva(f"F_{k}gen", "id", {datos.ids[i]: float(Fk[i]) for i in range(n)},
                        f"Endogamia con {k} generaciones")

    # parámetros individuales (MiDef)
    filas = []
    for i in range(n):
        f = {"animal": datos.ids[i], "padre": datos.nombre(div.s[i]), "madre": datos.nombre(div.d[i]),
             "f": float(F[i]), "ar": float(ar[i]), "df9": None if df9[i] != df9[i] else float(df9[i]),
             "df8": None if df8[i] != df8[i] else float(df8[i]), "gmax": int(div.gmax[i]),
             "gcom": int(div.gcom[i]), "ge": float(div.ecg[i]), "crias": int(hijos[i]), "ref": "sí" if ref[i] else ""}
        if Fk is not None:
            f["fk"] = float(Fk[i])
        filas.append(f)
    cols = [("animal", "Animal"), ("padre", "Padre"), ("madre", "Madre"), ("f", "F", "numero", 4),
            ("ar", "AR", "numero", 4), ("df9", "ΔFi (t−1)", "numero", 5), ("df8", "ΔFi (t)", "numero", 5),
            ("gmax", "Gen. máx.", "entero"), ("gcom", "Gen. compl.", "entero"), ("ge", "Gen. equiv.", "numero", 2),
            ("crias", "Crías", "entero"), ("ref", "Referencia")]
    if Fk is not None:
        cols.insert(4, ("fk", f"F {k} gen.", "numero", 4))
    r.tabla("Parámetros individuales", cols, filas[:LIMITE_FILAS], id="individuales",
            nota="Equivale a la tabla MiDef de ENDOG. Use «Agregar a la hoja» para llevar F, AR, ΔFi y GE a sus datos.")
    altos = sorted(range(n), key=lambda i: -F[i])[:100]
    if F[altos[0]] > 0:
        r.grafico("histograma", "Distribución de F (animales consanguíneos)", **histograma(F[consang]), eje="F")
    r.columna_nueva("F", "id", {datos.ids[i]: float(F[i]) for i in range(n)}, "Coeficiente de endogamia")
    r.columna_nueva("AR", "id", {datos.ids[i]: float(ar[i]) for i in range(n)}, "Relación promedio")
    r.columna_nueva("dFi", "id", {datos.ids[i]: (None if df9[i] != df9[i] else float(df9[i])) for i in range(n)},
                    "Incremento individual de endogamia (t = GE − 1)")
    r.columna_nueva("GE", "id", {datos.ids[i]: float(div.ecg[i]) for i in range(n)}, "Generaciones completas equivalentes")
    r.formula("Endogamia", "F (Meuwissen y Luo, 1992)")
    r.formula("Relación promedio", "c' = (1/n)·1'A,  A = (I − ½P)⁻¹ D (I − ½P')⁻¹")
    r.formula("Incremento individual", "ΔFi = 1 − (1 − Fi)^(1/(t − 1)),  t = generaciones equivalentes")
    r.formula("Ne realizado", "Ne = 1/(2·ΔF̄);  EE = 2·Ne²·σΔF/√n")
    r.formula("Ne por coascendencia", "Δc_jk = 1 − (1 − c_jk)^(2/(g_j + g_k));  Ne = 1/(2·Δc̄)")
    r.formula("Regresión en la referencia", "ΔF = b / (1 − (F̄ − b)),  b = pendiente de F sobre GE")
    r.parametro("ne_realizado", ne09["ne"])
    r.parametro("f_media", float(F.mean()))
    _avisos(r, datos)
    return r.json()


def _avisos(r, datos):
    errores = [a for a in datos.ped.avisos if a["tipo"] == "error"]
    if errores:
        r.nota("El pedigrí tiene inconsistencias (" + "; ".join(a["mensaje"] for a in errores[:3])
               + "). Revíselas en «Calidad y verificación del pedigrí».", "advertencia")
    if datos.ped.agregados:
        r.nota(f"{len(datos.ped.agregados)} progenitores sin fila propia se agregaron como animales base.")


# ================================================================== FUNDADORES Y ANCESTROS
def fundadores(payload, ctx):
    datos = DatosPedigri(payload, ctx)
    div, op = datos.div, datos.op
    n = div.n
    ref, desc_ref = _ref_o_defecto(datos, "ambos")
    ctx.progreso(0.35, "Contribuciones de los fundadores…")
    fund_total = div.fundadores(np.ones(n, bool))
    fund_ref = div.fundadores(ref)
    ctx.progreso(0.45, "Ancestros (Boichard et al., 1997)…")
    anc = div.ancestros(ref)
    fbar = div.coascendencia_media(ref)
    fg = 1 / (2 * fbar) if fbar > 0 else math.nan
    fe, fa = fund_ref["fe"], anc["fa"]
    r = Resultado("fundadores", "fundadores", "Fundadores, ancestros y origen de los genes",
                  f"{n} animales · población de referencia: {desc_ref}")
    r.indicador("Número efectivo de fundadores (fe)", fe, decimales=1, destacado=True,
                descripcion=f"Población de referencia; {fund_ref['contribuyentes']} fundadores contribuyen. "
                            f"Con toda la población: fe = {fund_total['fe']:.2f}")
    r.indicador("Número efectivo de ancestros (fa)", fa, decimales=1,
                descripcion=f"{len(anc['elegidos'])} ancestros explican el {100 * anc['acumulado']:.1f} % de los genes; "
                            f"{anc['explican_50'] or '—'} explican el 50 %")
    r.indicador("Genomas fundadores equivalentes (fg)", fg, decimales=1,
                descripcion=f"fg = 1/(2·f̄), coascendencia media de la referencia f̄ = {fbar:.4f}")
    if fa == fa and fe == fe:
        r.indicador("fe / fa", fe / fa, decimales=2,
                    descripcion="Mayor que 1 indica cuellos de botella (uso desbalanceado de reproductores)")

    resumen = [
        {"dato": "Tamaño de la población", "valor": n},
        {"dato": "Población base (uno o ambos padres desconocidos)", "valor": fund_total["base"]},
        {"dato": "Población base real (medio fundador = ½)", "valor": fund_total["base_real"]},
        {"dato": "Número efectivo de fundadores, toda la población (fe)", "valor": fund_total["fe"]},
        {"dato": "Endogamia esperada por desbalance de fundadores (%)", "valor": 100 / (2 * fund_total["fe"])},
        {"dato": "Endogamia media observada (%)", "valor": 100 * float(div.F.mean())},
        {"dato": "Animales en la población de referencia", "valor": int(ref.sum())},
        {"dato": "Fundadores que contribuyen a la referencia", "valor": fund_ref["contribuyentes"]},
        {"dato": "Número efectivo de fundadores de la referencia (fe)", "valor": fe},
        {"dato": "Ancestros que contribuyen a la referencia", "valor": len(anc["elegidos"])},
        {"dato": "Número efectivo de ancestros (fa)", "valor": fa},
        {"dato": "Ancestros que explican el 50 % de los genes", "valor": anc["explican_50"]},
        {"dato": "Genomas fundadores equivalentes (fg)", "valor": fg},
    ]
    r.tabla("Resumen del origen de los genes", [("dato", "Dato"), ("valor", "Valor", "numero", 2)], resumen,
            id="resumen", nota="Equivale a Founders.txt y Ancestor.txt de ENDOG. fe ≥ fa ≥ fg: la caída de fe a fa refleja "
                               "cuellos de botella y la de fa a fg la pérdida de genes por deriva.")
    # fundadores
    filas = []
    acum = 0.0
    for f in fund_ref["filas"]:
        if f["q"] <= 0:
            continue
        acum += f["q"]
        filas.append({"animal": datos.ids[f["i"]], "tipo": "Fantasma (medio fundador)" if f["fantasma"] else "Fundador",
                      "q": 100 * f["q"], "acum": 100 * acum})
    r.tabla("Contribución de los fundadores a la referencia", [("animal", "Fundador"), ("tipo", "Tipo"),
                                                               ("q", "Contribución (%)", "numero", 3),
                                                               ("acum", "Acumulada (%)", "numero", 2)],
            filas, id="fundadores",
            nota="Cuando un animal tiene un solo padre conocido, el desconocido se trata como un fundador «fantasma».")
    # ancestros
    rebano = datos.grupos("rebano")
    filas = []
    acum = 0.0
    completo, _ = div.tipos_base()
    for orden, (i, p, b) in enumerate(zip(anc["elegidos"], anc["marginal"], anc["bruta"]), 1):
        acum += p
        filas.append({"orden": orden, "animal": datos.ids[i], "marginal": 100 * p, "acum": 100 * acum,
                      "bruta": 100 * b, "fundador": "sí" if completo[i] else "", "rebano": rebano[i] if rebano else None})
    cols = [("orden", "Orden", "entero"), ("animal", "Ancestro"), ("marginal", "Contribución marginal (%)", "numero", 3),
            ("acum", "Acumulada (%)", "numero", 2), ("bruta", "Contribución total (%)", "numero", 3),
            ("fundador", "Fundador")]
    if rebano:
        cols.append(("rebano", "Rebaño"))
    r.tabla("Ancestros que explican la diversidad (Boichard et al., 1997)", cols, filas, id="ancestros",
            nota="La contribución marginal es la parte de los genes de la referencia que aporta el ancestro y no está "
                 "explicada por los ancestros elegidos antes.")
    top = filas[:20]
    if top:
        r.grafico("barras", "Ancestros con mayor contribución marginal (%)",
                  barras=[{"etiqueta": f["animal"], "valor": f["marginal"]} for f in top], eje="Contribución (%)")
        r.grafico("lineas", "Contribución acumulada de los ancestros",
                  series=[{"nombre": "Acumulada (%)", "puntos": [[f["orden"], f["acum"]] for f in filas[:300]]}],
                  eje_x="Número de ancestros", eje_y="% de los genes de la referencia")
    # rebaños fundadores
    if rebano:
        por_rebano = defaultdict(float)
        for i, p in zip(anc["elegidos"], anc["marginal"]):
            por_rebano[rebano[i] if rebano[i] is not None else "(sin rebaño)"] += p
        tot = sum(por_rebano.values())
        sh = np.array([v / tot for v in por_rebano.values()])
        fh = 1 / float(np.sum(sh ** 2))
        r.indicador("Número efectivo de rebaños fundadores (fh)", fh, decimales=1)
        r.tabla("Rebaños de origen de los ancestros", [("rebano", "Rebaño"), ("pct", "Contribución (%)", "numero", 2)],
                sorted([{"rebano": h, "pct": 100 * v / tot} for h, v in por_rebano.items()], key=lambda x: -x["pct"]),
                id="rebanos_fund", nota=f"fh = 1/Σ(contribución del rebaño)² = {fh:.2f}")
    # GCI
    if op.get("gci", True):
        ctx.progreso(0.75, "Índice de conservación genética (GCI)…")
        g = div.gci()
        if g is None:
            r.nota("El GCI no se calculó: el pedigrí es demasiado grande (fundadores × animales).", "info")
        else:
            r.columna_nueva("GCI", "id", {datos.ids[i]: (None if g[i] != g[i] else float(g[i])) for i in range(n)},
                            "Índice de conservación genética (Alderson, 1992)")
            mejores = sorted(np.flatnonzero(ref), key=lambda i: -(g[i] if g[i] == g[i] else 0))[:50]
            r.tabla("Animales de la referencia con mayor GCI", [("animal", "Animal"), ("gci", "GCI", "numero", 2),
                                                               ("f", "F", "numero", 4)],
                    [{"animal": datos.ids[i], "gci": float(g[i]), "f": float(div.F[i])} for i in mejores], id="gci",
                    nota="Un GCI alto indica que el animal reúne genes de muchos fundadores en proporciones parejas: "
                         "valioso para programas de conservación.")
            r.indicador("GCI medio de la referencia", float(np.nanmean(g[ref])), decimales=2)
    # endogamia parcial
    try:
        npar = int(op.get("parcial_n") or 0)
    except (TypeError, ValueError):
        npar = 0
    if npar > 0:
        npar = min(npar, 10)
        ctx.progreso(0.85, "Endogamia parcial (Lacy et al., 1996)…")
        focos_f = [f["i"] for f in fund_ref["filas"] if not f["fantasma"]][:npar]
        focos_a = [i for i in anc["elegidos"] if i not in focos_f and not completo[i]][:npar]
        idx, Pf = div.endogamia_parcial(focos_f, ref)
        Pa = div.endogamia_parcial(focos_a, ref, cortar=True)[1] if focos_a else np.zeros((len(idx), 0))
        Fm = float(div.F[idx].mean())
        resumen = []
        for j, i in enumerate(focos_f):
            resumen.append({"foco": datos.ids[i], "tipo": "Fundador", "media": float(Pf[:, j].mean()),
                            "pct": 100 * float(Pf[:, j].mean()) / Fm if Fm > 0 else None})
        for j, i in enumerate(focos_a):
            resumen.append({"foco": datos.ids[i], "tipo": "Ancestro", "media": float(Pa[:, j].mean()),
                            "pct": 100 * float(Pa[:, j].mean()) / Fm if Fm > 0 else None})
        r.tabla("Endogamia parcial por fundador y ancestro", [("foco", "Animal"), ("tipo", "Tipo"),
                                                              ("media", "F parcial media", "numero", 5),
                                                              ("pct", "% de la F media", "numero", 1)], resumen, id="parcial",
                nota="Probabilidad de que un animal de la referencia sea homocigoto idéntico por descendencia para un alelo "
                     "del fundador (o del ancestro, tratado como fundador). La suma sobre todos los fundadores es F.")
        filas = []
        for k, i in enumerate(idx):
            f = {"animal": datos.ids[i], "f": float(div.F[i])}
            for j in range(len(focos_f)):
                f[f"pf{j}"] = float(Pf[k, j])
            for j in range(len(focos_a)):
                f[f"pa{j}"] = float(Pa[k, j])
            filas.append(f)
        cols = [("animal", "Animal"), ("f", "F", "numero", 4)] + \
               [(f"pf{j}", f"F por {datos.ids[i]}", "numero", 5) for j, i in enumerate(focos_f)] + \
               [(f"pa{j}", f"F por {datos.ids[i]} (anc.)", "numero", 5) for j, i in enumerate(focos_a)]
        r.tabla("Endogamia parcial de cada animal de la referencia", cols, filas, id="parcial_ind")
    r.formula("Fundadores efectivos", "fe = 1 / Σ q_k²  (q_k: contribución del fundador k)")
    r.formula("Ancestros efectivos", "fa = 1 / Σ p_j²  (p_j: contribución marginal, Boichard et al., 1997)")
    r.formula("Genomas fundadores equivalentes", "fg = 1 / (2·f̄)  (Caballero y Toro, 2000)")
    r.formula("GCI", "GCI = 1 / Σ p_i²  (p_i: proporción de genes del fundador i; Alderson, 1992)")
    _avisos(r, datos)
    r.parametro("fe", fe)
    r.parametro("fa", fa)
    r.parametro("fg", fg)
    return r.json()


# ================================================================== INTERVALO GENERACIONAL
def intervalo_generacional(payload, ctx):
    datos = DatosPedigri(payload, ctx)
    div = datos.div
    n = div.n
    nac = datos.nacimiento()
    if nac is None:
        raise ErrorDatos("Asigne la columna con la fecha (o el año) de nacimiento.")
    if np.sum(~np.isnan(nac)) < 3:
        raise ErrorDatos("Casi ningún animal tiene una fecha de nacimiento reconocible "
                         "(use fechas, número de serie de Excel o el año).")
    sexo = datos.sexo()
    ref = datos.referencia()
    ctx.progreso(0.4, "Edades de los padres por vía…")
    gen, L = dv.intervalos(div, nac, sexo, solo_reproductores=True)
    edad, Lt = dv.intervalos(div, nac, sexo, solo_reproductores=False)
    r = Resultado("intervalos", "intervalos", "Intervalo generacional y tamaño efectivo",
                  f"{n} animales · {int(np.sum(~np.isnan(nac)))} con fecha de nacimiento")
    nombres = {"padre-hijo": "Padre → hijo", "padre-hija": "Padre → hija", "madre-hijo": "Madre → hijo",
               "madre-hija": "Madre → hija", "total": "Total", "sexo de la cría desconocido": "Cría de sexo desconocido"}
    cols = [("via", "Vía"), ("n", "N", "entero"), ("media", "Media (años)", "numero", 2), ("de", "DE", "numero", 2),
            ("ee", "EE", "numero", 3)]
    for f in gen + edad:
        f["via"] = nombres.get(f["via"], f["via"])
    via_p = [f for f in gen if f["via"].startswith("Padre")]
    via_m = [f for f in gen if f["via"].startswith("Madre")]

    def media_vias(fs):
        tot = sum(f["n"] for f in fs)
        return sum(f["n"] * f["media"] for f in fs if f["n"]) / tot if tot else math.nan
    r.indicador("Intervalo generacional (L)", L, decimales=2, destacado=True, unidad="años",
                descripcion=f"Edad media de los padres al nacer la progenie que luego se reproduce "
                            f"({gen[-1]['n']} pares). Vía paterna {media_vias(via_p):.2f} · materna {media_vias(via_m):.2f}")
    r.indicador("Edad media de los padres al parto", Lt, decimales=2, unidad="años",
                descripcion="Toda la progenie, se reproduzca o no")
    r.tabla("Intervalo generacional por vía (progenie que se reproduce)", cols, gen, id="intervalo",
            nota="Intervalo generacional: edad media de los padres al nacimiento de las crías que llegan a ser "
                 "progenitores (James, 1977).")
    r.tabla("Edad de los padres al nacimiento de toda su progenie", cols, edad, id="edad")
    if gen[-1].get("negativas"):
        r.nota(f"{gen[-1]['negativas']} pares padre–cría tienen al progenitor nacido después que la cría y se "
               "excluyeron. Revise esas fechas.", "advertencia")
    if ref is not None:
        gref, Lref = dv.intervalos(div, nac, sexo, solo_reproductores=False, mascara=ref)
        for f in gref:
            f["via"] = nombres.get(f["via"], f["via"])
        r.tabla("Edad de los padres de la población de referencia", cols, gref, id="intervalo_ref")
    # Ne por fecha
    filas = []
    for etiqueta, mascara in (("Todo el pedigrí", np.ones(n, bool)), ("Referencia", ref)):
        if mascara is None:
            continue
        nf = dv.ne_fecha(div, nac, L, mascara)
        filas.append({"poblacion": etiqueta, "n": nf["n"], "b": nf.get("b"), "ne": nf["ne"], "ne_log": nf["ne_log"]})
    r.tabla("Ne por regresión sobre la fecha de nacimiento", [
        ("poblacion", "Población"), ("n", "Animales", "entero"), ("b", "Pendiente de F por año", "numero", 6),
        ("ne", "Ne (regresión lineal)", "numero", 2), ("ne_log", "Ne (regresión logarítmica)", "numero", 2)],
        filas, id="ne_fecha",
        nota=f"ΔF = L·b / (1 − (F̄ − L·b)) con L = {L:.2f} años; logarítmica: Ne = −1/(2·L·b'), b' = pendiente de "
             "ln(1 − F) sobre la fecha (Pérez-Enciso, 1995).")
    if filas:
        r.indicador("Ne por regresión sobre la fecha", filas[0]["ne"], decimales=1, nivel=_nivel_ne(filas[0]["ne"]))
    # Ne por varianza del tamaño de familia (Hill)
    filas = []
    if L == L and L > 0:
        periodo = max(1, int(round(L)))
        validos = ~np.isnan(nac)
        if validos.any():
            inicio = math.floor(np.nanmin(nac))
            fin = math.floor(np.nanmax(nac))
            for a in range(inicio, fin + 1, periodo):
                conj = validos & (nac >= a) & (nac < a + periodo)
                h = dv.ne_hill(div, sexo, conj)
                if h["machos"] >= 2 and h["hembras"] >= 2:
                    filas.append({"periodo": f"{dv.anio_de(a + 0.5)}–{dv.anio_de(a + periodo - 0.5)}", "machos": h["machos"],
                                  "hembras": h["hembras"], "hm": h["media_hijos_machos"], "hh": h["media_hijos_hembras"],
                                  "ne": h["ne"]})
    if ref is not None:
        h = dv.ne_hill(div, sexo, ref)
        filas.append({"periodo": "Población de referencia", "machos": h["machos"], "hembras": h["hembras"],
                      "hm": h.get("media_hijos_machos"), "hh": h.get("media_hijos_hembras"), "ne": h["ne"]})
    if filas:
        r.tabla("Ne por varianza del tamaño de familia (Hill, 1979)", [
            ("periodo", "Nacidos en"), ("machos", "Machos reproductores", "entero"),
            ("hembras", "Hembras reproductoras", "entero"), ("hm", "Hijos reprod. por macho", "numero", 2),
            ("hh", "Hijos reprod. por hembra", "numero", 2), ("ne", "Ne", "numero", 1)], filas, id="ne_hill",
            nota="Por períodos de nacimiento de longitud ≈ L (cada período se trata como una generación). El tamaño de "
                 "familia cuenta solo los hijos e hijas que a su vez se reproducen.")
    # edad media de los padres por año
    por_anio = defaultdict(lambda: {"p": [], "m": []})
    for i in range(n):
        if nac[i] != nac[i]:
            continue
        a = dv.anio_de(nac[i])
        for p, k in ((div.s[i], "p"), (div.d[i], "m")):
            if p >= 0 and nac[p] == nac[p] and nac[i] >= nac[p]:
                por_anio[a][k].append(nac[i] - nac[p])
    anios = sorted(a for a, v in por_anio.items() if v["p"] or v["m"])
    if len(anios) > 1:
        r.grafico("lineas", "Edad media de los padres según el año de nacimiento de la cría",
                  series=[{"nombre": "Padre", "puntos": [[a, float(np.mean(por_anio[a]["p"]))] for a in anios if por_anio[a]["p"]]},
                          {"nombre": "Madre", "puntos": [[a, float(np.mean(por_anio[a]["m"]))] for a in anios if por_anio[a]["m"]]}],
                  eje_x="Año de nacimiento de la cría", eje_y="Edad (años)")
    r.formula("Intervalo generacional", "L = media de las edades de los padres al nacer su progenie reproductora")
    r.formula("Hill (1979)", "1/Ne = [2 + σ²mm + 2(M/F)cov(mm,mf) + (M/F)²σ²mf]/(16M) + "
                             "[2 + (F/M)²σ²fm + 2(F/M)cov(fm,ff) + σ²ff]/(16F)")
    _avisos(r, datos)
    r.parametro("L", L)
    return r.json()


# ================================================================== SUBPOBLACIONES
def subpoblaciones(payload, ctx):
    datos = DatosPedigri(payload, ctx)
    div, op = datos.div, datos.op
    grupos = datos.grupos("subpoblacion")
    if grupos is None:
        raise ErrorDatos("Asigne la columna que define las subpoblaciones (área, rebaño, línea…).")
    try:
        M = float(op.get("tamano_comun")) if op.get("tamano_comun") not in (None, "") else None
    except (TypeError, ValueError):
        raise ErrorDatos("El tamaño común debe ser un número.")
    if M is not None and M < 2:
        raise ErrorDatos("El tamaño común debe ser al menos 2.")
    ctx.progreso(0.5, "Coascendencias dentro y entre subpoblaciones…")
    res = dv.estadisticos_f(div, grupos, M)
    et = [str(e) for e in res["etiquetas"]]
    k = len(et)
    r = Resultado("subpoblaciones", "subpoblaciones", "Estructura de la población: estadísticos F y distancias",
                  f"{k} subpoblaciones según «{datos.roles['subpoblacion']}» · {int(res['N'].sum())} animales"
                  + (f" · ajustado a tamaño {M:g}" if M else ""))
    r.indicador("FST (diferenciación entre subpoblaciones)", res["fst"], decimales=4, destacado=True,
                descripcion=_texto_fst(res["fst"]))
    r.indicador("FIS (dentro de subpoblaciones)", res["fis"], decimales=4,
                descripcion="Negativo: se evitan los apareamientos entre parientes; positivo: subdivisión interna")
    r.indicador("FIT (población total)", res["fit"], decimales=4)
    r.indicador("Diversidad génica (1 − f̃)", 1 - res["f_meta"], decimales=4)
    r.tabla("Metapoblación", [("dato", "Dato"), ("valor", "Valor", "numero", 6)], [
        {"dato": "Coascendencia media dentro de subpoblaciones", "valor": res["f_dentro"]},
        {"dato": "Autocoascendencia media", "valor": res["s_meta"]},
        {"dato": "Endogamia media", "valor": res["F_meta"]},
        {"dato": "Distancia de Nei media", "valor": res["d_nei"]},
        {"dato": "Coascendencia media de la metapoblación", "valor": res["f_meta"]},
        {"dato": "FIS", "valor": res["fis"]}, {"dato": "FST", "valor": res["fst"]}, {"dato": "FIT", "valor": res["fit"]},
    ], id="metapoblacion", nota="Equivale a MatFst de ENDOG. (1 − FIT) = (1 − FIS)(1 − FST).")
    filas = []
    for i in range(k):
        filas.append({"sub": et[i], "n": int(res["N"][i]), "F": res["F"][i], "fii": res["fij"][i, i], "s": res["s"][i],
                      "fg": res["fg"][i], "fis": res["fst_par"][i, i], "ge": res["ge"][i]})
    r.tabla("Subpoblaciones", [("sub", "Subpoblación"), ("n", "Animales", "entero"), ("F", "F medio", "numero", 4),
                               ("fii", "Coascendencia media", "numero", 4), ("s", "Autocoascendencia", "numero", 4),
                               ("fg", "fg", "numero", 1), ("fis", "FIS", "numero", 4),
                               ("ge", "Gen. equivalentes", "numero", 2)], filas, id="subpob")

    def matriz(nombre, Mx, id, decimales=4, diag_nota=""):
        filas = []
        for i in range(k):
            f = {"sub": et[i]}
            for j in range(k):
                f[f"c{j}"] = float(Mx[i, j]) if j <= i else None
            filas.append(f)
        r.tabla(nombre, [("sub", "")] + [(f"c{j}", et[j], "numero", decimales) for j in range(k)], filas, id=id,
                nota=diag_nota)
    matriz("Coascendencia media entre subpoblaciones (fij)", res["fij"], "matriz_fij")
    matriz("Distancia mínima de Nei (DA)", res["Dnei"], "matriz_nei", diag_nota="DA = (fii + fjj)/2 − fij")
    matriz("Distancia de parentesco (Dk)", res["Dk"], "matriz_dk", diag_nota="Dk = (si + sj)/2 − fij")
    matriz("FST entre pares de subpoblaciones (FIS en la diagonal)", res["fst_par"], "matriz_fst")
    if k <= 40:
        r.grafico("matriz_calor", "FST entre pares de subpoblaciones", nombres=et,
                  valores=[[float(res["fst_par"][i, j]) if i != j else 0.0 for j in range(k)] for i in range(k)],
                  rango=[0, max(0.05, float(np.max(res["fst_par"][~np.eye(k, dtype=bool)])))])
    if op.get("contribuciones", True):
        g = dv.contribucion_diversidad(res)
        r.tabla("Contribución de cada subpoblación a la diversidad génica", [
            ("subpoblacion", "Subpoblación"), ("gd_sin", "GD sin ella", "numero", 5),
            ("interna", "Diversidad interna (%)", "numero", 3), ("distancia", "Distancia entre subpob. (%)", "numero", 3),
            ("total", "Cambio total (%)", "numero", 3)],
            [dict(f, subpoblacion=str(f["subpoblacion"])) for f in g["filas"]], id="ganancia",
            nota=f"GD = 1 − f̃ = {g['gd']:.5f}. Cambio en la diversidad génica al quitar cada subpoblación (Caballero y "
                 "Toro, 2002): negativo = la subpoblación aporta diversidad (conviene conservarla).")
        r.grafico("barras", "Cambio en la diversidad génica al quitar cada subpoblación (%)",
                  barras=[{"etiqueta": str(f["subpoblacion"]), "valor": f["total"]} for f in g["filas"]], eje="%")
    if M:
        r.nota(f"Las coascendencias dentro de cada subpoblación se ajustaron a un tamaño común de {M:g} animales "
               "(Bartolomé et al., 2010) para que el tamaño de muestra no sesgue las comparaciones.")
    r.formula("Distancia de Nei", "DA_ij = (fii + fjj)/2 − fij")
    r.formula("Estadísticos F", "FIS = (F̃ − f̄)/(1 − f̄);  FST = (f̄ − f̃)/(1 − f̃);  FIT = (F̃ − f̃)/(1 − f̃)")
    r.formula("Ajuste por tamaño", "f_ii(M) = s_i/M + (1 − 1/M)·f̄_ii fuera de la diagonal")
    _avisos(r, datos)
    r.parametro("fst", res["fst"])
    return r.json()


def _texto_fst(fst):
    if fst < 0.05:
        return "Diferenciación pequeña (< 0,05): las subpoblaciones comparten casi todo su acervo genético."
    if fst < 0.15:
        return "Diferenciación moderada (0,05–0,15)."
    if fst < 0.25:
        return "Diferenciación grande (0,15–0,25)."
    return "Diferenciación muy grande (> 0,25)."


# ================================================================== REBAÑOS
def rebanos(payload, ctx):
    datos = DatosPedigri(payload, ctx)
    div = datos.div
    reb = datos.grupos("rebano")
    if reb is None:
        raise ErrorDatos("Asigne la columna con el rebaño de nacimiento de cada animal.")
    ref = datos.referencia()
    filas = dv.estructura_rebanos(div, reb)
    r = Resultado("rebanos", "rebanos", "Estructura de la población por rebaños",
                  f"{len(filas)} rebaños según «{datos.roles['rebano']}»")
    tipos = Counter(f["tipo"] for f in filas)
    r.indicador("Rebaños", len(filas), decimales=0,
                descripcion=" · ".join(f"{t}: {c}" for t, c in tipos.most_common()))
    resumen = []
    for t in ("Núcleo", "Multiplicador", "Comercial", "Aislado", "Sin padres identificados"):
        fs = [f for f in filas if f["tipo"] == t]
        if not fs:
            continue
        propio = sum(f["padre_propio"] for f in fs)
        ajeno = sum(f["padre_ajeno"] for f in fs)
        resumen.append({"tipo": t, "usa_ajenos": "sí" if fs[0]["usa_ajenos"] else "no",
                        "usa_propios": "sí" if any(f["usa_propios"] for f in fs) else "no",
                        "vende": "sí" if fs[0]["vende"] else "no", "n": len(fs),
                        "pct_ajeno": 100 * ajeno / (propio + ajeno) if propio + ajeno else None,
                        "crias": sum(f["crias"] for f in fs)})
    r.tabla("Resumen por tipo de rebaño", [("tipo", "Tipo"), ("usa_ajenos", "Usa machos de otros rebaños"),
                                           ("usa_propios", "Usa machos propios"), ("vende", "Vende machos"),
                                           ("n", "Rebaños", "entero"), ("crias", "Crías", "entero"),
                                           ("pct_ajeno", "% de crías con padre de otro rebaño", "numero", 1)],
            resumen, id="tipos", nota="Clasificación de Vassallo et al. (1986): núcleo = solo usa machos propios y vende; "
                                      "multiplicador = compra y vende; comercial = compra y no vende; aislado = ni compra ni vende.")
    r.grafico("barras", "Rebaños por tipo", barras=[{"etiqueta": f["tipo"], "valor": f["n"]} for f in resumen], eje="Rebaños")
    r.tabla("Detalle por rebaño", [("rebano", "Rebaño"), ("tipo", "Tipo"), ("crias", "Crías", "entero"),
                                   ("padre_propio", "Con padre propio", "entero"), ("padre_ajeno", "Con padre de otro", "entero"),
                                   ("pct_propio", "% padre propio", "numero", 1), ("como_padre", "Crías de sus machos", "entero"),
                                   ("en_su_rebano", "…en el propio rebaño", "entero"), ("en_otro", "…en otros rebaños", "entero"),
                                   ("pct_otro", "% en otros rebaños", "numero", 1)],
            [dict(f, rebano=str(f["rebano"])) for f in filas], id="detalle",
            nota="Equivale a la tabla HerStr de ENDOG. El rebaño de un padre es su rebaño de nacimiento.")
    mascara = ref if ref is not None else np.ones(div.n, bool)
    rob = dv.robertson(div, reb, mascara)
    if rob:
        r.tabla("Rebaños que aportan padres (Robertson, 1953)", [("generacion", "Generación", "entero"),
                                                                 ("via", "Ancestro"), ("registros", "Registros", "entero"),
                                                                 ("rebanos", "Rebaños", "entero"),
                                                                 ("efectivos", "Rebaños efectivos", "numero", 2)],
                rob, id="robertson",
                nota="Rebaños efectivos = 1/Σ p_h², inverso de la probabilidad de que dos animales al azar tengan ese "
                     "ancestro nacido en el mismo rebaño.")
        r.indicador("Rebaños efectivos que aportan padres", rob[0]["efectivos"], decimales=1,
                    descripcion=f"De {rob[0]['rebanos']} rebaños que aportan padres")
    _avisos(r, datos)
    return r.json()
