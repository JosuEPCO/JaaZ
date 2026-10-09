"""Muestra un resultado del motor (dict de genetica/resultados.Resultado) con componentes de
Streamlit: indicadores, tablero de correlación, gráficos, tablas, notas, fórmulas, columnas nuevas
para la hoja y exportación a Excel."""
import math
import re

import pandas as pd
import streamlit as st

from motor import archivos

from . import estado, graficos
from .estilo import esc


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def fmt(v, dec=3):
    if not _fin(v):
        return "—" if v is None or isinstance(v, (int, float)) else str(v)
    a = abs(v)
    if a != 0 and (a >= 1e9 or a < 10 ** -dec / 2):
        return f"{v:.{max(1, min(dec, 4))}e}"
    return f"{v:,.{dec}f}"


def clase_nivel(nivel):
    if not nivel:
        return None
    n = nivel.lower()
    if re.search(r"alta|fuerte|adecuad", n):
        return "alta"
    if re.search(r"media|moderada", n):
        return "media"
    if re.search(r"fuera|crític", n):
        return "error"
    return "baja"


def insignia(nivel):
    return f'<span class="jz-badge {clase_nivel(nivel)}">{esc(nivel)}</span>' if nivel else ""


# ================================================================== indicadores
def _escala(etiqueta):
    e = etiqueta.lower()
    if re.search(r"correlaci|^rg|rg ·", e):
        return (-1.0, 1.0)
    if re.search(r"heredabilidad|repetibilidad|h² |^h²|intraclase|exactitud", e):
        return (0.0, 1.0)
    return None


def _kpi(ind):
    dec = ind.get("decimales", 3) if ind.get("decimales") is not None else 3
    v = ind.get("valor")
    valor = (f"{v:,.0f}" if dec == 0 else fmt(v, dec)) if _fin(v) else esc(v if v is not None else "—")
    ee = f'<span class="ee">± {fmt(ind["ee"], dec)}</span>' if _fin(ind.get("ee")) else ""
    ic = ind.get("ic")
    ic_txt = f'<div class="ic">IC: {fmt(ic[0], dec)} a {fmt(ic[1], dec)}</div>' if ic and _fin(ic[0]) else ""
    medidor = ""
    escala = _escala(ind["etiqueta"]) if _fin(v) else None
    if escala:
        a, b = escala
        pos = lambda x: max(0.0, min(100.0, 100 * (x - a) / (b - a)))  # noqa: E731
        izq = pos(min(0, v)) if a < 0 else 0
        der = pos(max(0, v)) if a < 0 else pos(v)
        banda = (f'<span style="left:{pos(ic[0]):.2f}%;width:{max(0.5, pos(ic[1]) - pos(ic[0])):.2f}%"></span>'
                 if ic and _fin(ic[0]) and _fin(ic[1]) else "")
        medidor = (f'<div class="jz-medidor">{banda}<i style="left:{izq:.2f}%;width:{max(0.6, der - izq):.2f}%"></i></div>'
                   f'<div class="jz-escala"><span>{a:g}</span>{"<span>0</span>" if a < 0 else ""}<span>{b:g}</span></div>')
    unidad = f' <span class="ee">{esc(ind["unidad"])}</span>' if ind.get("unidad") else ""
    desc = f'<div class="desc">{esc(ind["descripcion"])}</div>' if ind.get("descripcion") else ""
    return (f'<div class="jz-kpi{" destacado" if ind.get("destacado") else ""}"><div class="et"><span>{esc(ind["etiqueta"])}</span>'
            f'{insignia(ind.get("nivel"))}</div><div class="val">{valor}{unidad}{ee}</div>{ic_txt}{medidor}{desc}</div>')


def indicadores(lista):
    if lista:
        st.html('<div class="jz-kpis">' + "".join(_kpi(i) for i in lista) + "</div>")


# ================================================================== tablas
def _formato(col):
    t, d = col.get("tipo"), col.get("decimales", 3)
    if t == "numero":
        return st.column_config.NumberColumn(col["titulo"], format=f"%.{d}f")
    if t == "entero":
        return st.column_config.NumberColumn(col["titulo"], format="%d")
    if t == "porcentaje":
        return st.column_config.NumberColumn(col["titulo"], format=f"%.{d if d is not None else 1}f %%")
    return st.column_config.TextColumn(col["titulo"])


def tabla(t):
    cols = t.get("columnas") or []
    filas = t.get("filas") or []
    st.markdown(f"**{t.get('titulo', '')}**" + (f"  ·  {len(filas):,} filas" if len(filas) > 10 else ""))
    if not filas:
        st.caption("Sin filas.")
        return
    datos = {}
    for j, c in enumerate(cols):
        clave = f"c{j}"
        vals = [f.get(c["clave"]) for f in filas]
        if c.get("tipo") == "p":
            vals = ["—" if not _fin(v) else ("< 0.0001" if v < 0.0001 else f"{v:.4f}") for v in vals]
        elif c.get("tipo") in ("numero", "entero", "porcentaje"):
            vals = [v if _fin(v) else None for v in vals]
        else:
            vals = ["" if v is None else str(v) for v in vals]
        datos[clave] = vals
    df = pd.DataFrame(datos)
    config = {f"c{j}": (_formato(c) if c.get("tipo") != "p" else st.column_config.TextColumn(c["titulo"])) for j, c in enumerate(cols)}
    st.dataframe(df, hide_index=True, column_config=config, height=min(36 + 35 * len(df), 460))
    if t.get("nota"):
        st.caption(t["nota"])


def notas(res):
    for n in sorted(res.get("notas") or [], key=lambda n: 0 if n.get("tipo") == "advertencia" else 1):
        (st.warning if n.get("tipo") in ("advertencia", "error") else st.info)(n["texto"])


def formulas(res):
    if res.get("formulas"):
        with st.expander("Fórmulas y método", icon=":material/function:"):
            st.html("".join(f'<div class="jz-formula"><b>{esc(f["titulo"])}</b><code>{esc(f["expresion"])}</code></div>'
                            for f in res["formulas"]))


def graficos_de(res, clave):
    figuras = [(g, graficos.crear(g)) for g in res.get("graficos") or []]
    figuras = [(g, f) for g, f in figuras if f is not None]
    for k in range(0, len(figuras), 2):
        cols = st.columns(2, gap="medium")
        for j, (col, (g, f)) in enumerate(zip(cols, figuras[k:k + 2])):
            with col.container(border=True):
                st.markdown(f"**{g.get('titulo', '')}**")
                st.altair_chart(f, width="stretch", height=graficos.ALTO_CONTENEDOR.get(g.get("tipo"), "content"),
                                key=f"{clave}:g{k + j}")


# ================================================================== tablero de correlación
def _barra01(v, ic):
    banda = ""
    if ic and _fin(ic[0]) and _fin(ic[1]):
        a, c = max(0, min(1, ic[0])), max(0, min(1, ic[1]))
        banda = f'<span style="left:{100 * a:.1f}%;width:{max(.5, 100 * (c - a)):.1f}%"></span>'
    relleno = f'<i style="width:{max(.8, 100 * max(0, min(1, v))):.1f}%"></i>' if _fin(v) else ""
    return f'<div class="jz-barra">{banda}{relleno}</div><div class="jz-escala" style="margin-top:6px"><span>0</span><span>0,5</span><span>1</span></div>'


def _metrica(etiqueta, v, ee, ic, nivel, conf, nota=None):
    if not _fin(v):
        return (f'<div class="jz-metrica"><div class="jz-cab" style="margin:0"><span class="jz-meta">{esc(etiqueta)}</span></div>'
                f'<div class="val" style="color:var(--faint)">—</div><div class="nota">{esc(nota or "No estimada.")}</div></div>')
    ee_txt = f'<span class="ee">± {fmt(ee)}</span>' if _fin(ee) else ""
    ic_txt = f'<div class="ic">IC {round(100 * conf)} %: {fmt(ic[0])} a {fmt(ic[1])}</div>' if ic and _fin(ic[0]) else ""
    return (f'<div class="jz-metrica"><div class="jz-cab" style="margin:0;align-items:center"><span class="jz-meta">{esc(etiqueta)}</span>'
            f'{insignia(nivel)}</div><div class="val">{fmt(v)}{ee_txt}</div>{ic_txt}{_barra01(v, ic)}</div>')


def _particion(ra):
    if not _fin(ra.get("h2")):
        return ""
    a = max(0.0, min(1.0, ra["h2"]))
    r = max(a, min(1.0, ra["r"])) if _fin(ra.get("r")) else None
    partes = ([("Genética aditiva", a, "var(--s1)"), ("Ambiente y otros efectos", 1 - a, "var(--s3)")] if r is None else
              [("Genética aditiva", a, "var(--s1)"), ("Ambiente permanente", r - a, "var(--s2)"), ("Ambiente temporal", 1 - r, "var(--s3)")])
    barra = "".join(f'<span title="{esc(e)}: {100 * v:.1f} %" style="flex-grow:{v:.4f};background:{c}"></span>' for e, v, c in partes if v > 0.004)
    leyenda = "".join(f'<span><i style="background:{c}"></i>{esc(e)} {100 * v:.0f} %</span>' for e, v, c in partes)
    return (f'<div class="jz-bloque"><div class="jz-meta">De dónde viene la variación</div><div class="jz-particion">{barra}</div>'
            f'<div class="jz-leyenda">{leyenda}</div></div>')


def _tarjeta_rasgo(ra, tb):
    cv = 100 * ra["de"] / ra["media"] if _fin(ra.get("de")) and _fin(ra.get("media")) and ra["media"] > 0 else None
    return (f'<div class="jz-panel"><div class="jz-cab"><div><div class="jz-meta">Rasgo · {ra["n"]:,} registros</div>'
            f'<h3>{esc(ra["nombre"])}</h3></div></div><div class="jz-metricas">'
            + _metrica("Heredabilidad (h²)", ra.get("h2"), ra.get("ee_h2"), ra.get("ic_h2"), ra.get("nivel_h2"), tb["conf"])
            + _metrica("Repetibilidad (r)", ra.get("r"), ra.get("ee_r"), ra.get("ic_r"), ra.get("nivel_r"), tb["conf"], tb.get("nota_rep"))
            + "</div>" + _particion(ra)
            + f'<dl class="jz-datos chico" style="margin-top:14px;max-width:none"><div><dt>Media</dt><dd>{fmt(ra.get("media"))}</dd></div>'
              f'<div><dt>DE</dt><dd>{fmt(ra.get("de"))}</dd></div><div><dt>CV</dt><dd>{"—" if cv is None else f"{cv:.1f} %"}</dd></div></dl></div>')


def tablero_correlacion(res, clave):
    tb = res["tablero"]
    pares = tb.get("pares") or []
    if not pares:
        return
    par = pares[0]
    if len(pares) > 1:
        etiquetas = [f'{p["x"]} × {p["y"]}' for p in pares]
        elegido = st.segmented_control("Par de rasgos", etiquetas, default=etiquetas[0], key=f"{clave}:par", required=True)
        par = pares[etiquetas.index(elegido)] if elegido in etiquetas else pares[0]
    ic = par.get("ic_rg") if par.get("ic_rg") and _fin(par["ic_rg"][0]) else None
    pill = ""
    if par.get("distinta"):
        si = par["distinta"] == "Sí"
        pill = f'<span class="jz-pill{" si" if si else ""}">{"Distinta de cero" if si else "No distinta de cero"}</span>'
    ee = f'<span class="ee">± {fmt(par["ee_rg"])}</span>' if _fin(par.get("ee_rg")) else ""
    st.html(f'<div class="jz-panel"><div class="jz-cab"><div><div class="jz-meta">Correlación genética · {esc(tb.get("resumen", ""))}</div>'
            f'<h3>{esc(par["x"])} × {esc(par["y"])}</h3></div>{insignia(par.get("nivel"))}</div>'
            f'<div class="jz-hero-cuerpo"><div><div class="jz-meta">Genética (rG)</div><div class="jz-valor-grande">{fmt(par.get("rg"))}{ee}</div>{pill}</div>'
            f'<dl class="jz-datos"><div><dt>IC {round(100 * tb["conf"])} % de rG</dt><dd>{f"{fmt(ic[0])} a {fmt(ic[1])}" if ic else "—"}</dd></div>'
            f'<div><dt>Fenotípica (rP)</dt><dd>{fmt(par.get("rp"))}</dd></div>'
            f'<div><dt>Ambiental (rE)</dt><dd>{fmt(par["re"]) if _fin(par.get("re")) else "No se estima"}</dd></div></dl></div></div>')
    rasgos = tb.get("rasgos") or []
    for k in range(0, len(rasgos), 2):
        cols = st.columns(2, gap="medium")
        for col, ra in zip(cols, rasgos[k:k + 2]):
            with col:
                st.html(_tarjeta_rasgo(ra, tb))
    graficos_de(res, clave)


# ================================================================== resultado completo
def _excel(item):
    """Función que arma el Excel recién al pulsar «Excel» (no en cada interacción con la página)."""
    resultado = item["resultado"]
    return lambda: archivos.escribir_resultados([resultado])


def _nombre_archivo(texto):
    return re.sub(r'[\\/:*?"<>|]', "_", texto)[:80] + ".xlsx"


def _agregar_columnas(res, item):
    libro = estado.libro_por_id(item.get("libro_id"))
    if not libro:
        return st.warning("El libro original ya no está abierto.")
    hoja = next((h for h in libro["hojas"] if h["nombre"] == item.get("hoja")), None)
    if hoja is None:
        return st.warning("La hoja original ya no está abierta.")
    roles = item.get("roles") or {}
    df = hoja["df"]
    total = 0
    for c in res["columnas_nuevas"]:
        clave = roles.get(c["clave"]) or (roles.get("animal") if c["clave"] == "id" else None)
        if not clave or clave not in df.columns:
            continue
        def norm(k):
            k = estado.valor_celda(k)
            if isinstance(k, float) and k.is_integer():
                k = int(k)
            return None if k is None else str(k).strip()
        nuevos = [c["valores"].get(norm(k)) for k in df[clave]]
        df[c["nombre"]] = pd.to_numeric(pd.Series(nuevos, index=df.index, dtype=object), errors="coerce")
        total += sum(v is not None for v in nuevos)
    st.toast(f"Columnas agregadas a «{hoja['nombre']}» ({total:,} valores).", icon=":material/check:")


def _crear_hoja(hn, item):
    libro = estado.libro_por_id(item.get("libro_id"))
    if not libro:
        return st.warning("El libro original ya no está abierto.")
    nombre, k = hn["nombre"][:31], 2
    while any(h["nombre"] == nombre for h in libro["hojas"]):
        nombre = f'{hn["nombre"][:27]} ({k})'
        k += 1
    libro["hojas"].append({"nombre": nombre, "df": estado.a_dataframe([hn["columnas"]] + hn["filas"], encabezado=True)})
    st.toast(f"Hoja «{nombre}» creada en «{libro['nombre']}».", icon=":material/check:")


def mostrar(item):
    res = item["resultado"]
    clave = f"res:{item['id']}"
    st.subheader(res.get("titulo", "Resultado"))
    st.caption(" · ".join(x for x in [res.get("subtitulo"), res.get("fecha")] if x))
    acciones = st.columns([1, 1, 1, 3])
    acciones[0].download_button("Excel", data=_excel(item), file_name=_nombre_archivo(res.get("titulo", "resultado")),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                icon=":material/download:", key=f"{clave}:xlsx", on_click="ignore")
    if res.get("columnas_nuevas") and estado.libro_por_id(item.get("libro_id")):
        nombres = ", ".join(c["nombre"] for c in res["columnas_nuevas"])
        if acciones[1].button("Agregar a la hoja", icon=":material/add_column_right:", key=f"{clave}:col", help=f"Agrega {nombres}"):
            _agregar_columnas(res, item)
    hn = res.get("hoja_nueva")
    if hn and hn.get("filas") and estado.libro_por_id(item.get("libro_id")):
        if acciones[2].button("Crear hoja", icon=":material/note_add:", key=f"{clave}:hoja", help=hn.get("descripcion")):
            _crear_hoja(hn, item)
    notas(res)
    if res.get("tablero") and res["tablero"].get("tipo") == "correlacion":
        t1, t2 = st.tabs(["Tablero", "Tablas y método"], key=f"{clave}:tabs")
        with t1:
            tablero_correlacion(res, clave)
        with t2:
            for t in res.get("tablas") or []:
                tabla(t)
            formulas(res)
        return
    indicadores(res.get("indicadores"))
    graficos_de(res, clave)
    for t in res.get("tablas") or []:
        tabla(t)
    formulas(res)
