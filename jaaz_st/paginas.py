"""Páginas que no son análisis: Inicio, Hoja de datos, Visor de pedigrí, Resultados y Ayuda."""
import io
import re
from datetime import datetime

import pandas as pd
import streamlit as st

from motor import archivos
from motor.genetica.datos import ErrorDatos, columna_ids
from motor.genetica.pedigri import Pedigri

from . import estado, vista
from .estilo import encabezado, esc

VERSION = "1.4"
TIPOS_ARCHIVO = ["xlsx", "xlsm", "xls", "csv", "txt", "ped", "dat", "tsv"]


def _ir(nombre):
    return st.session_state["_paginas"][nombre]


def _abrir_ejemplo(nombre):
    try:
        estado.abrir_ejemplo(nombre)
        st.toast(f"Ejemplo «{nombre}» abierto.", icon=":material/check:")
    except archivos.ErrorArchivo as e:
        st.error(str(e))


# ================================================================== INICIO
def inicio():
    st.html(f"""<div class="jz-hero"><div class="jz-pastilla-vivo"><i></i>En línea · versión {VERSION}</div>
      <div class="jz-display">Genética cuantitativa, <em>abierta como una hoja de cálculo.</em></div>
      <div class="jz-lead">JaaZ abre sus libros de Excel o CSV y estima heredabilidad, repetibilidad y correlaciones genéticas;
      revisa el pedigrí, la endogamia, los valores de cría y la diversidad genética de la población.</div></div>""")
    c1, c2, _ = st.columns([1, 1, 3])
    c1.page_link(_ir("datos"), label="Abrir archivo", icon=":material/upload_file:", width="stretch")
    c2.page_link(_ir("ayuda"), label="Ayuda y métodos", icon=":material/help:", width="stretch")

    st.space("medium")
    st.markdown("### Datos de ejemplo")
    st.caption("Ábralos para probar cualquier análisis; quedan en «Hoja de datos».")
    ejemplos = estado.ejemplos()
    for k in range(0, len(ejemplos), 2):
        cols = st.columns(2, gap="medium")
        for col, e in zip(cols, ejemplos[k:k + 2]):
            with col.container(border=True):
                st.markdown(f"**{e['nombre']}**")
                st.caption(e["descripcion"] or " ")
                st.button("Abrir", key=f"ej:{e['nombre']}", icon=":material/folder_open:", on_click=_abrir_ejemplo, args=(e["nombre"],))

    st.space("medium")
    st.markdown("### Herramientas")
    herramientas = [
        ("datos", "Hoja de datos", "Ver, editar y descargar sus libros."), ("descriptiva", "Estadística descriptiva", "Resúmenes, normalidad y correlaciones."),
        ("heredabilidad", "Heredabilidad", "Medios hermanos, hermanos completos, regresión y REML."),
        ("repetibilidad", "Repetibilidad", "ANOVA, REML y producción más probable."),
        ("correlacion", "Correlación genética", "ANCOVA, REML bivariado y covarianza cruzada, con tablero."),
        ("blup", "Evaluación BLUP", "Valores de cría y exactitud con el modelo animal."),
        ("visor", "Visor de pedigrí", "Árbol de ancestros y descendientes de un animal."),
        ("pedigri", "Calidad del pedigrí", "Errores, integridad (MacCluer) y parientes cercanos."),
        ("parentesco", "Parentesco", "Matriz A, coascendencia y posibles parejas."),
        ("apareamientos", "Apareamientos", "Endogamia esperada de la cría para cada pareja."),
        ("endogamia", "Endogamia y Ne", "F, AR, ΔFi y tamaño efectivo por varios métodos."),
        ("fundadores", "Fundadores y ancestros", "fe, fa, fg, GCI y endogamia parcial."),
        ("intervalos", "Intervalo generacional", "Cuatro vías, Ne por fecha y Ne de Hill."),
        ("subpoblaciones", "Subpoblaciones", "FIS, FST, FIT y distancias de Nei."),
        ("rebanos", "Estructura de rebaños", "Núcleos, multiplicadores y Robertson."),
    ]
    for k in range(0, len(herramientas), 3):
        cols = st.columns(3, gap="small")
        for col, (clave, titulo, texto) in zip(cols, herramientas[k:k + 3]):
            with col.container(border=True, height="stretch"):
                st.page_link(_ir(clave), label=titulo, icon=_ir(clave).icon)
                st.caption(texto)

    st.space("medium")
    st.markdown("### Cómo se trabaja")
    pasos = [("Abra sus datos", "Un registro por fila: animal, padre, madre, efectos (hato, año, sexo) y los rasgos medidos."),
             ("Revise la hoja", "Corrija valores, agregue o quite filas y elija la hoja que va a analizar."),
             ("Elija el análisis", "El programa propone qué columna es el padre, el animal y el rasgo; revíselo."),
             ("Interprete y exporte", "Cada resultado trae errores estándar, interpretación, gráficos y exportación a Excel.")]
    st.html('<div class="jz-pasos">' + "".join(f'<div class="jz-paso"><div class="n">{i:02d}</div><b>{esc(t)}</b><p>{esc(p)}</p></div>'
                                                for i, (t, p) in enumerate(pasos, 1)) + "</div>")
    st.html('<div class="jz-pie"><span><span class="marca">JaaZ</span> · Genética cuantitativa</span>'
            "<span>Métodos validados con Mrode (2014) y ENDOG 4.8</span><span>Sus datos no se guardan en el servidor</span></div>")


# ================================================================== HOJA DE DATOS
def _interpretar(v):
    """Texto escrito en la tabla → valor (número si lo parece; «007» queda como texto)."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    s = str(v).strip()
    if not s:
        return None
    if re.match(r"^[+-]?0\d", s):
        return s
    if re.match(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$", s):
        x = float(s)
        return int(x) if x.is_integer() and "." not in s and "e" not in s.lower() else x
    if re.match(r"^[+-]?\d+,\d+$", s):
        return float(s.replace(",", "."))
    return s


def _para_editor(df):
    """Copia apta para el editor: columnas con números y textos mezclados se muestran como texto."""
    vista_df = df.copy()
    for c in vista_df.columns:
        if vista_df[c].dtype == object:
            vista_df[c] = [None if estado.valor_celda(v) is None else str(v) for v in vista_df[c]]
    return vista_df


def _aplicar_cambios(hoja, clave):
    cambios = st.session_state.get(clave) or {}
    df = hoja["df"].copy()
    if df.dtypes.apply(lambda t: t != object).any():
        df = df.astype(object)
    for fila, valores in (cambios.get("edited_rows") or {}).items():
        for col, v in valores.items():
            df.at[df.index[int(fila)], col] = _interpretar(v)
    for nueva in cambios.get("added_rows") or []:
        df.loc[len(df)] = [_interpretar(nueva.get(c)) for c in df.columns]
    borrar = cambios.get("deleted_rows") or []
    if borrar:
        df = df.drop(df.index[borrar])
    df = df.reset_index(drop=True)
    for c in df.columns:  # recuperar columnas numéricas
        vals = [estado.valor_celda(v) for v in df[c]]
        if any(v is not None for v in vals) and all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)) for v in vals):
            df[c] = pd.to_numeric(pd.Series(vals, dtype=object))
    hoja["df"] = df
    hoja["version"] = hoja.get("version", 0) + 1


def datos():
    encabezado("Datos", "Hoja de datos", "Abra libros de Excel (.xlsx, .xls), CSV o texto de Pedigree Viewer (.ped, .txt). "
                                         "Se abren todas las hojas. Los datos viven solo en su sesión.")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        subidos = st.file_uploader("Abrir archivos", type=TIPOS_ARCHIVO, accept_multiple_files=True, key="subida")
        vistos = st.session_state.setdefault("_subidos", set())
        for f in subidos or []:
            if f.file_id in vistos:
                continue
            vistos.add(f.file_id)
            try:
                libro, sep = estado.abrir_bytes(f.getvalue(), f.name)
                filas = sum(len(h["df"]) for h in libro["hojas"])
                st.toast(f"Se abrió «{f.name}»: {len(libro['hojas'])} hoja(s), {filas:,} filas"
                         + (f" (separador: {sep})" if sep else "") + ".", icon=":material/check:")
            except archivos.ErrorArchivo as e:
                st.error(f"No se pudo abrir «{f.name}»: {e}")
    with c2:
        st.markdown("**Datos de ejemplo**")
        for e in estado.ejemplos():
            st.button(e["nombre"], key=f"dej:{e['nombre']}", icon=":material/description:", on_click=_abrir_ejemplo, args=(e["nombre"],),
                      help=e["descripcion"], type="tertiary")

    libros = estado.libros()
    if not libros:
        st.info("Todavía no hay libros abiertos.")
        return
    st.divider()
    activo = estado.libro_activo() or libros[-1]
    ids = [l["id"] for l in libros]
    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    nombres_libros = {l["id"]: l["nombre"] for l in libros}
    lid = c1.selectbox("Libro", ids, index=ids.index(activo["id"]), format_func=nombres_libros.get, key="dat:libro")
    st.session_state["activo"] = lid
    libro = estado.libro_por_id(lid)
    nombres = [h["nombre"] for h in libro["hojas"]]
    hsel = c2.selectbox("Hoja", list(range(len(nombres))), index=min(libro.get("hoja", 0), len(nombres) - 1),
                        format_func=lambda k: f"{nombres[k]} ({len(libro['hojas'][k]['df']):,} filas)", key=f"dat:hoja:{lid}")
    libro["hoja"] = hsel
    if c3.button("Cerrar libro", icon=":material/close:", key=f"dat:cerrar:{lid}", width="stretch"):
        estado.cerrar_libro(lid)
        st.rerun()
    hoja = libro["hojas"][hsel]
    if hoja.get("crudo") is not None:
        con_cab = st.toggle("La primera fila son nombres de columna", value=hoja.get("encabezado", True), key=f"dat:cab:{lid}:{hsel}")
        if con_cab != hoja.get("encabezado", True):
            hoja["encabezado"] = con_cab
            hoja["df"] = estado.a_dataframe(hoja["crudo"], encabezado=con_cab)
            hoja["version"] = hoja.get("version", 0) + 1
    df = hoja["df"]

    tipos = estado.tipos(df)
    det = estado.detectar_roles(df)
    roles = {k: det.get(k) for k in ("animal", "padre", "madre", "sexo", "fecha") if det.get(k)}
    st.caption(f"{len(df):,} filas × {len(df.columns)} columnas · numéricas: {sum(t == 'num' for t in tipos.values())}"
               + (" · detectadas: " + ", ".join(f"{k} = {v}" for k, v in roles.items()) if roles else ""))
    clave = f"editor:{lid}:{hsel}:{hoja.get('version', 0)}"
    if len(df) <= 60000:
        st.data_editor(_para_editor(df), key=clave, num_rows="dynamic", height=480, on_change=_aplicar_cambios, args=(hoja, clave))
        st.caption("Edite directamente en la tabla; puede agregar filas al final y borrar las seleccionadas. Pegue desde Excel con Ctrl+V.")
    else:
        st.dataframe(df, height=480)
        st.caption("Hoja grande: se muestra sin edición (más de 60 000 filas).")

    c1, c2, _ = st.columns([1, 1, 3])

    def libro_xlsx():  # se arma recién al pulsar el botón
        hojas = [{"nombre": h["nombre"], "columnas": [str(c) for c in h["df"].columns],
                  "filas": [[estado.valor_celda(v) for v in fila] for fila in h["df"].itertuples(index=False)]} for h in libro["hojas"]]
        return archivos.escribir_libro({"hojas": hojas})

    base = re.sub(r"\.(xlsx|xlsm|xls|csv|txt|ped|dat|tsv)$", "", libro["nombre"], flags=re.I)
    c1.download_button("Libro en Excel", data=libro_xlsx, file_name=f"{base}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       icon=":material/download:", on_click="ignore", key=f"dat:xlsx:{lid}", width="stretch")
    c2.download_button("Hoja en CSV", data=lambda: ("﻿" + df.to_csv(index=False)).encode("utf-8"), file_name=f"{base}_{hoja['nombre']}.csv",
                       mime="text/csv", icon=":material/download:", on_click="ignore", key=f"dat:csv:{lid}:{hsel}", width="stretch")


# ================================================================== VISOR DE PEDIGRÍ
MAX_CRIAS = 20  # crías dibujadas por animal (un toro con cientos de hijos haría ilegible el árbol)


def _dot(ped, foco, arriba, abajo, F):
    codigo = ped.codigo[foco]
    incluir = {codigo: 0}
    frente = [codigo]
    for g in range(arriba):
        sig = []
        for c in frente:
            for p in (int(ped.s[c]), int(ped.d[c])):
                if p and p not in incluir:
                    incluir[p] = -(g + 1)
                    sig.append(p)
        frente = sig
    hijos = ped.descendientes()
    frente = [codigo]
    for g in range(abajo):
        sig = []
        for c in frente:
            for h in hijos.get(c, [])[:MAX_CRIAS]:
                if h not in incluir:
                    incluir[h] = g + 1
                    sig.append(h)
        frente = sig
    sexo = ped.sexo_registrado
    # nombres de nodo únicos por árbol: la transición animada entre dos árboles deforma un nodo que
    # conserva su nombre pero cambia de forma (rectángulo ↔ óvalo)
    pre = f"a{abs(hash((foco, arriba, abajo))) % 10 ** 8}_"
    lineas = ['digraph G {', 'graph [rankdir=TB, bgcolor="transparent", nodesep=0.25, ranksep=0.5];',
              'node [fontname="Helvetica", fontsize=11, fontcolor="#f3f3f3", style="filled", fillcolor="#151515", color="#6f6f6f", penwidth=1.6];',
              'edge [arrowsize=0.5, penwidth=1.1];']
    usa_padre = {int(ped.s[c]) for c in range(1, ped.n + 1) if ped.s[c]}
    usa_madre = {int(ped.d[c]) for c in range(1, ped.n + 1) if ped.d[c]}
    for c in incluir:
        i = ped.ids[c - 1]
        sx = sexo.get(i) or ("M" if c in usa_padre else "H" if c in usa_madre else None)
        forma = "box" if sx == "M" else "ellipse" if sx == "H" else "box"
        color = "#3987e5" if sx == "M" else "#d95926" if sx == "H" else "#6f6f6f"
        f = F[c - 1]
        etiqueta = f"{i}" + (f"\\nF = {f:.3f}" if f > 1e-9 else "")
        extra = ', fillcolor="#c98500", fontcolor="#101010", color="#c98500"' if c == codigo else ""
        lineas.append(f'"{pre}{c}" [label="{etiqueta.replace(chr(34), "")}", shape={forma}, color="{color}"{extra}];')
    for c in incluir:
        for p, col in ((int(ped.s[c]), "#3987e5"), (int(ped.d[c]), "#d95926")):
            if p and p in incluir:
                lineas.append(f'"{pre}{p}" -> "{pre}{c}" [color="{col}"];')
    lineas.append("}")
    return "\n".join(lineas), len(incluir)


def visor():
    encabezado("Pedigrí", "Visor de pedigrí", "Árbol de un animal: sus ancestros y su descendencia, con el sexo (macho = rectángulo azul, "
                                              "hembra = óvalo naranja) y la endogamia de cada uno.")
    if not estado.libros():
        st.info("Abra una hoja con las columnas animal, padre y madre (p. ej. «pedigri_hato.ped» de los ejemplos).")
        return
    opciones = [(l["id"], k) for l in estado.libros() for k in range(len(l["hojas"]))]
    activo = estado.libro_activo()
    defecto = opciones[0]
    if activo:  # si el libro tiene una hoja «Pedigrí», se prefiere esa
        k = next((i for i, h in enumerate(activo["hojas"]) if re.search(r"pedig", h["nombre"], re.I)), activo.get("hoja", 0))
        defecto = (activo["id"], k)
    c1, c2, c3, c4 = st.columns(4)
    nombres = {(l["id"], k): f'{l["nombre"]} › {h["nombre"]}' for l in estado.libros() for k, h in enumerate(l["hojas"])}
    sel = c1.selectbox("Hoja", opciones, index=opciones.index(defecto) if defecto in opciones else 0, format_func=nombres.get, key="vis:hoja")
    df = estado.libro_por_id(sel[0])["hojas"][sel[1]]["df"]
    det = estado.detectar_roles(df)
    cols = list(df.columns)
    idx = lambda c: cols.index(c) if c in cols else None  # noqa: E731
    cid = c2.selectbox("Animal", cols, index=idx(det.get("animal")) or 0, key=f"vis:id:{sel}")
    cpa = c3.selectbox("Padre", [None] + cols, index=(idx(det.get("padre")) + 1) if det.get("padre") in cols else 0,
                       format_func=lambda c: c or "— ninguna —", key=f"vis:pa:{sel}")
    cma = c4.selectbox("Madre", [None] + cols, index=(idx(det.get("madre")) + 1) if det.get("madre") in cols else 0,
                       format_func=lambda c: c or "— ninguna —", key=f"vis:ma:{sel}")
    if not cpa and not cma:
        st.warning("Elija al menos la columna de padre o la de madre.")
        return
    sexos = estado.valores(df, det["sexo"]) if det.get("sexo") in cols else None
    try:
        ped = _pedigri_cache(tuple(columna_ids(estado.valores(df, cid))),
                             tuple(columna_ids(estado.valores(df, cpa))) if cpa else None,
                             tuple(columna_ids(estado.valores(df, cma))) if cma else None,
                             tuple(sexos) if sexos else None)
    except ErrorDatos as e:
        st.error(str(e))
        return
    F = ped.F()
    c1, c2, c3 = st.columns([2, 1, 1])
    hijos = ped.descendientes()
    # por defecto, un animal de la última generación (el árbol de ancestros más profundo)
    profundo = int(max(range(1, ped.n + 1), key=lambda c: (ped.generacion[c], c)))
    foco = c1.selectbox("Animal a mostrar", ped.ids, index=profundo - 1, key=f"vis:foco:{sel}", help="Escriba para buscar por ID.")
    arriba = c2.slider("Generaciones de ancestros", 0, 6, 3, key="vis:arriba")
    abajo = c3.slider("Generaciones de descendientes", 0, 3, 1, key="vis:abajo")
    dot, n = _dot(ped, foco, arriba, abajo, F)
    with st.container(border=True, horizontal_alignment="center"):
        st.graphviz_chart(dot)  # tamaño natural: un árbol chico no se estira a todo el ancho
    k = ped.codigo[foco]
    st.html(f'<dl class="jz-datos" style="max-width:none"><div><dt>Animal</dt><dd>{esc(foco)}</dd></div>'
            f'<div><dt>Endogamia (F)</dt><dd>{F[k - 1]:.4f}</dd></div><div><dt>Crías</dt><dd>{len(hijos.get(k, []))}</dd></div></dl>')
    st.caption(f"{n} animales en el diagrama · {ped.n:,} en el pedigrí · se dibujan hasta {MAX_CRIAS} crías por animal. "
               "Para el análisis completo use «Calidad y verificación».")
    for a in ped.avisos:
        if a["tipo"] == "error":
            st.warning(a["mensaje"])


@st.cache_resource(show_spinner="Armando el pedigrí…", max_entries=8)
def _pedigri_cache(ids, padres, madres, sexos):
    n = len(ids)
    return Pedigri(list(ids), list(padres) if padres else [None] * n, list(madres) if madres else [None] * n,
                   sexos=list(sexos) if sexos else None)


# ================================================================== RESULTADOS
def resultados():
    encabezado("Informe", "Resultados", "Los análisis de esta sesión. Elija uno para verlo o marque varios para exportarlos juntos a Excel.")
    items = st.session_state["resultados"]
    if not items:
        st.info("Aún no hay resultados. Ejecute un análisis desde el menú.")
        return
    nombre = lambda r: f'{r["resultado"].get("titulo")} · {r.get("libro", "")}'  # noqa: E731
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    ids = [r["id"] for r in items]
    elegido = c1.selectbox("Resultado", ids, format_func=lambda i: nombre(next(r for r in items if r["id"] == i)), key="res:sel")
    marcados = c2.multiselect("Exportar juntos", ids, default=ids[:1], format_func=lambda i: next(r for r in items if r["id"] == i)["resultado"].get("titulo"),
                              key="res:exp")
    c3, c4, _ = st.columns([1, 1, 3])
    if marcados:
        elegidos = [next(r for r in items if r["id"] == i)["resultado"] for i in marcados]
        c3.download_button("Exportar marcados", data=lambda: archivos.escribir_resultados(elegidos), file_name="JaaZ_resultados.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           icon=":material/download:", on_click="ignore", width="stretch")
    if c4.button("Borrar todos", icon=":material/delete:", width="stretch"):
        st.session_state["resultados"] = []
        st.rerun()
    st.divider()
    vista.mostrar(next(r for r in items if r["id"] == elegido))


# ================================================================== AYUDA
AYUDA = """
### Cómo empezar
1. **Abra sus datos** en *Hoja de datos* (Excel, CSV, texto o .ped) o use un ejemplo.
2. **Revise la hoja**: la primera fila se toma como nombres de columna (puede desactivarlo).
3. **Elija un análisis** en el menú. El panel propone qué columna es el padre, la madre, el animal y el rasgo según sus nombres; revíselo.
4. Pulse **Calcular**. El resultado trae el parámetro con su error estándar, intervalo de confianza, interpretación, tablas y gráficos, y se
   exporta a Excel.

Los datos viven solo en su sesión del navegador: **no se guardan en el servidor**. Descargue el libro o los resultados para conservarlos.

### Formato de los datos
Un registro por fila. Nombres de columna libres (se reconocen `padre`, `sire`, `toro`, `madre`, `dam`, `animal`, `id`, `arete`, `sexo`, `año`…).
Padre o madre desconocidos: celda vacía, `0`, `.`, `#`, `*` o `NA`. Datos faltantes: vacío, `.`, `NA` o un código como `-9999` (indíquelo en
*Opciones*). Los identificadores con ceros a la izquierda («007») se conservan.

### Heredabilidad
* **Medios hermanos paternos (ANOVA):** h² = 4σ²s / (σ²s + σ²w); EE de Swiger et al. (1964). Con registros repetidos usa el modelo
  anidado padre / animal / registro y da también la repetibilidad.
* **Hermanos completos:** separa los componentes de padre (h²S, el más confiable) y de madre (h²D, incluye efectos maternos).
* **Regresión progenie–progenitor:** h² = 2b con un progenitor, b con el promedio de ambos.
* **REML (modelo de padre o animal):** máxima verosimilitud restringida, exacta con datos desbalanceados y efectos fijos; errores estándar
  por la matriz de información y prueba de razón de verosimilitud (mezcla ½χ²₀ + ½χ²₁).

### Repetibilidad
r = (σ²G + σ²Ep) / σ²P, por ANOVA entre animales, anidado padre / animal o REML (con pedigrí separa la varianza aditiva del ambiente
permanente). Incluye la **producción más probable**: PMP = μ + [n·r / (1 + (n−1)·r)]·(ȳ − μ).

### Correlación genética
rG = σA(xy) / √(σ²A(x)·σ²A(y)), por ANCOVA de medios hermanos (EE de Robertson, 1959), REML bivariado (padre o animal) o covarianza
cruzada progenitor–progenie (EE por bootstrap). El **tablero** muestra rG con su intervalo, las correlaciones fenotípica y ambiental, y
para cada rasgo h², repetibilidad y de dónde viene la variación. En *Tablas y método* está la selección indirecta:
eficiencia = rG·hX/hY y respuesta correlacionada RC = i·rG·hX·hY·σP.

### Pedigrí y diversidad genética
Endogamia de Meuwissen y Luo (1992); A⁻¹ de Henderson con la corrección de Quaas. Los análisis de diversidad siguen los métodos de
**ENDOG 4.8** (Gutiérrez y Goyache, 2005) y se validaron contra los valores publicados en su manual: relación promedio, ΔFi, Ne por
regresión, por coascendencia y por fecha, fe, fa (Boichard), fg, GCI, endogamia parcial, estadísticos F de Wright e intervalo generacional.

### Interpretación
| h² | Lectura |
|---|---|
| < 0,10 | Muy baja: la selección por fenotipo propio avanza muy despacio. |
| 0,10–0,20 | Baja: conviene usar información de parientes (BLUP). |
| 0,20–0,40 | Media: la selección individual es efectiva. |
| > 0,40 | Alta: respuesta rápida a la selección. |

Con pocos padres (menos de 20–30) las estimaciones son imprecisas: revise siempre el error estándar y el intervalo de confianza.

### Referencias
Falconer y Mackay (1996) · Mrode (2014) *Linear Models for the Prediction of Animal Breeding Values* · Meyer (1989) · Graser et al. (1987) ·
Henderson (1976) · Meuwissen y Luo (1992) · Robertson (1959) · Swiger et al. (1964) · Gutiérrez y Goyache (2005) · Gutiérrez et al. (2008, 2009) ·
Boichard et al. (1997) · Cervantes et al. (2011) · Caballero y Toro (2002) · MacCluer et al. (1983).

### Acerca de
**JaaZ** usa el mismo motor de cálculo que la versión de escritorio (Python: NumPy, SciPy), con sus pruebas automáticas (ejemplos de Mrode,
valores del manual de ENDOG y REML = ANOVA en datos balanceados). El visor se inspira en *Pedigree Viewer* (B. y S. Kinghorn) y los
análisis de diversidad en *ENDOG*; JaaZ es una aplicación independiente.
"""


def ayuda():
    encabezado("Informe", "Ayuda y métodos", "Formato de los datos, métodos, fórmulas, interpretación y referencias.")
    st.markdown(AYUDA)
