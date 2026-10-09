"""Gráficos de los resultados con Altair (los mismos tipos que la versión de escritorio):
componentes, histograma, dispersión, medias por grupo, barras, líneas y matriz de calor.
Paleta categórica validada para daltonismo sobre el fondo oscuro."""
import math

import altair as alt
import pandas as pd

from .estilo import PALETA

AZUL, NARANJA, GRIS = PALETA[1], PALETA[2], "#6f6f6f"


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _componentes(g):
    partes = [p for p in g.get("partes", []) if _fin(p.get("valor")) and p["valor"] > 0]
    if not partes:
        return None
    total = sum(p["valor"] for p in partes)
    df = pd.DataFrame([{"Componente": p["etiqueta"], "Varianza": p["valor"], "Porcentaje": p["valor"] / total, "orden": i}
                       for i, p in enumerate(partes)])
    return alt.Chart(df).mark_bar(height=28, cornerRadius=4).encode(
        x=alt.X("Varianza:Q", stack="normalize", axis=alt.Axis(format="%", title="% de la varianza fenotípica")),
        color=alt.Color("Componente:N", scale=alt.Scale(domain=list(df["Componente"]), range=PALETA),
                        legend=alt.Legend(orient="bottom", title=None, columns=1, labelLimit=0)),
        order="orden:Q",
        tooltip=["Componente", alt.Tooltip("Porcentaje:Q", format=".1%"), alt.Tooltip("Varianza:Q", format=",.4f")],
    ).properties(height=36)


# alto del contenedor cuando el gráfico lleva la leyenda debajo (Streamlit solo cuenta el área de dibujo)
ALTO_CONTENEDOR = {"componentes": 200}


def _histograma(g):
    bordes, conteos = g.get("bordes") or [], g.get("conteos") or []
    if not conteos:
        return None
    df = pd.DataFrame({"desde": bordes[:-1], "hasta": bordes[1:], "Frecuencia": conteos})
    barras = alt.Chart(df).mark_bar(color=PALETA[0], cornerRadiusTopLeft=3, cornerRadiusTopRight=3, binSpacing=1).encode(
        x=alt.X("desde:Q", title=g.get("eje") or None), x2="hasta:Q", y=alt.Y("Frecuencia:Q", title="Frecuencia"),
        tooltip=[alt.Tooltip("desde:Q", format=",.3f", title="Desde"), alt.Tooltip("hasta:Q", format=",.3f", title="Hasta"), "Frecuencia"])
    capas = [barras]
    media, de, n = g.get("media"), g.get("de"), g.get("n") or sum(conteos)
    if _fin(media) and _fin(de) and de > 0 and len(bordes) > 1:
        ancho = (bordes[-1] - bordes[0]) / len(conteos)
        xs = [bordes[0] + (bordes[-1] - bordes[0]) * k / 80 for k in range(81)]
        curva = pd.DataFrame({"x": xs, "y": [n * ancho * math.exp(-0.5 * ((x - media) / de) ** 2) / (de * math.sqrt(2 * math.pi)) for x in xs]})
        capas.append(alt.Chart(curva).mark_line(color=AZUL, strokeWidth=2).encode(x="x:Q", y="y:Q"))
    return alt.layer(*capas).properties(height=260)


def _dispersion(g):
    pts = [p for p in g.get("puntos", []) if _fin(p[0]) and _fin(p[1])]
    if not pts:
        return None
    ex, ey = g.get("eje_x") or "x", g.get("eje_y") or "y"
    df = pd.DataFrame(pts, columns=["x", "y"])
    tooltip = [alt.Tooltip("x:Q", title=ex, format=",.3f"), alt.Tooltip("y:Q", title=ey, format=",.3f")]
    if g.get("etiquetas") and len(g["etiquetas"]) == len(g.get("puntos", [])):
        df["Etiqueta"] = [e for e, p in zip(g["etiquetas"], g["puntos"]) if _fin(p[0]) and _fin(p[1])]
        tooltip.insert(0, "Etiqueta")
    puntos = alt.Chart(df).mark_circle(size=34 if len(df) > 800 else 60, color=PALETA[0], opacity=.55 if len(df) > 800 else .8).encode(
        x=alt.X("x:Q", title=ex, scale=alt.Scale(zero=False)), y=alt.Y("y:Q", title=ey, scale=alt.Scale(zero=False)), tooltip=tooltip)
    capas = [puntos]
    recta = g.get("recta")
    if recta and _fin(recta.get("b")) and _fin(recta.get("a")):
        x0, x1 = float(df["x"].min()), float(df["x"].max())
        linea = pd.DataFrame({"x": [x0, x1], "y": [recta["a"] + recta["b"] * x0, recta["a"] + recta["b"] * x1]})
        capas.append(alt.Chart(linea).mark_line(color=AZUL, strokeWidth=2).encode(x="x:Q", y="y:Q"))
    return alt.layer(*capas).properties(height=320)


def _medias_grupo(g):
    gr = g.get("grupos") or []
    if not gr:
        return None
    et = g.get("etiqueta_grupo") or "Grupo"
    df = pd.DataFrame([{"orden": i, et: d["grupo"], "Promedio": d["media"], "n": d["n"],
                        "bajo": d["media"] - 1.96 * (d["ee"] or 0), "alto": d["media"] + 1.96 * (d["ee"] or 0)}
                       for i, d in enumerate(gr)])
    x = alt.X("orden:Q", title=f"{len(gr)} {et.lower()}s ordenados por promedio", axis=alt.Axis(labels=False, ticks=False))
    tooltip = [et, alt.Tooltip("Promedio:Q", format=",.3f"), "n", alt.Tooltip("bajo:Q", format=",.3f", title="IC 95 % desde"),
               alt.Tooltip("alto:Q", format=",.3f", title="IC 95 % hasta")]
    barras = alt.Chart(df).mark_rule(color=PALETA[0], opacity=.5, strokeWidth=2).encode(x=x, y=alt.Y("bajo:Q", title="Promedio", scale=alt.Scale(zero=False)), y2="alto:Q")
    puntos = alt.Chart(df).mark_circle(color=PALETA[0], size=40, opacity=1).encode(x=x, y="Promedio:Q", tooltip=tooltip)
    capas = [barras, puntos]
    if _fin(g.get("media_general")):
        capas.append(alt.Chart(pd.DataFrame({"y": [g["media_general"]]})).mark_rule(color=AZUL, strokeWidth=1.5).encode(y="y:Q"))
    return alt.layer(*capas).properties(height=280)


def _barras(g):
    b = [x for x in g.get("barras", []) if _fin(x.get("valor"))]
    if not b:
        return None
    df = pd.DataFrame([{"Etiqueta": str(x["etiqueta"]), "Valor": x["valor"], "signo": "positivo" if x["valor"] >= 0 else "negativo"} for x in b])
    return alt.Chart(df).mark_bar(cornerRadius=3, height=14).encode(
        y=alt.Y("Etiqueta:N", sort=None, title=None),
        x=alt.X("Valor:Q", title=g.get("eje") or "Valor"),
        color=alt.Color("signo:N", scale=alt.Scale(domain=["positivo", "negativo"], range=[AZUL, NARANJA]), legend=None),
        tooltip=["Etiqueta", alt.Tooltip("Valor:Q", format=",.4f")],
    ).properties(height=max(120, 24 * len(df)))


def _lineas(g):
    filas = []
    for se in g.get("series", []):
        for p in se.get("puntos", []):
            if _fin(p[0]) and _fin(p[1]):
                filas.append({"Serie": se["nombre"], "x": p[0], "y": p[1]})
    if not filas:
        return None
    df = pd.DataFrame(filas)
    nombres = list(dict.fromkeys(df["Serie"]))
    enc = dict(x=alt.X("x:Q", title=g.get("eje_x") or None, axis=alt.Axis(format="d" if all(float(v).is_integer() for v in df["x"]) else ",.2f")),
               y=alt.Y("y:Q", title=g.get("eje_y") or None),
               color=alt.Color("Serie:N", scale=alt.Scale(domain=nombres, range=PALETA),
                               legend=alt.Legend(orient="bottom", title=None) if len(nombres) > 1 else None),
               tooltip=["Serie", alt.Tooltip("x:Q", title=g.get("eje_x") or "x"), alt.Tooltip("y:Q", format=",.4f")])
    base = alt.Chart(df)
    capas = [base.mark_line(strokeWidth=2).encode(**enc)]
    if len(df) <= 120:
        capas.append(base.mark_circle(size=50, opacity=1).encode(**enc))
    return alt.layer(*capas).properties(height=260)


def _matriz_calor(g):
    filas, cols, V = g.get("nombres") or [], g.get("nombres_col") or g.get("nombres") or [], g.get("valores") or []
    datos = [{"Fila": str(f), "Columna": str(c), "Valor": V[i][j] if i < len(V) and j < len(V[i]) and _fin(V[i][j]) else None}
             for i, f in enumerate(filas) for j, c in enumerate(cols)]
    if not datos:
        return None
    df = pd.DataFrame(datos)
    vals = [v for v in df["Valor"] if v is not None]
    if g.get("rango"):
        escala = alt.Scale(domain=list(g["rango"]), range=["#1a1a1a", PALETA[0]])
    elif any(v < 0 for v in vals):
        escala = alt.Scale(domain=[-1, 0, 1], range=[AZUL, "#2a2a2a", NARANJA])
    else:
        escala = alt.Scale(domain=[min(0, min(vals, default=0)), max(vals, default=1) or 1], range=["#1a1a1a", PALETA[0]])
    orden_f, orden_c = [str(f) for f in filas], [str(c) for c in cols]
    base = alt.Chart(df).encode(x=alt.X("Columna:N", sort=orden_c, title=None, axis=alt.Axis(labelAngle=-45)),
                                y=alt.Y("Fila:N", sort=orden_f, title=None))
    capas = [base.mark_rect(cornerRadius=3).encode(color=alt.Color("Valor:Q", scale=escala, legend=alt.Legend(title=None)),
                                                   tooltip=["Fila", "Columna", alt.Tooltip("Valor:Q", format=".3f")])]
    if len(filas) * len(cols) <= 144:
        capas.append(base.mark_text(fontSize=11, color="#f3f3f3").encode(text=alt.Text("Valor:Q", format=".2f")))
    return alt.layer(*capas).properties(height=max(160, 34 * len(filas)))


TIPOS = {"componentes": _componentes, "histograma": _histograma, "dispersion": _dispersion, "medias_grupo": _medias_grupo,
         "barras": _barras, "lineas": _lineas, "matriz_calor": _matriz_calor}


def crear(g):
    fn = TIPOS.get(g.get("tipo"))
    if not fn:
        return None
    try:
        return fn(g)
    except Exception:  # un gráfico con datos raros no debe tumbar la página
        return None
