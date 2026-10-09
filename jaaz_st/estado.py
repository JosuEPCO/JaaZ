"""Datos de la sesión: libros abiertos (hojas como DataFrame), libro y hoja activos, resultados,
lectura de archivos, tipos de columna y detección automática de columnas (padre, madre, animal…).

Cada visitante de la app tiene su propia sesión: los datos no se comparten ni se guardan en el
servidor (al cerrar la pestaña se pierden; para conservarlos, descargue el libro o los resultados).
"""
import math
import os
import re
import uuid

import numpy as np
import pandas as pd
import streamlit as st

from motor import archivos

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_EJEMPLOS = os.path.join(RAIZ, "ejemplos")
FALTANTE_TXT = {"", ".", "na", "n/a", "nan", "null", "none", "-", "?", "s/d"}
_NUM_COMA = re.compile(r"^[+-]?\d+,\d+$")


# ================================================================== sesión
def iniciar():
    st.session_state.setdefault("libros", [])        # [{id, nombre, hojas: [{nombre, df}], hoja}]
    st.session_state.setdefault("activo", None)      # id del libro activo
    st.session_state.setdefault("resultados", [])    # [{id, resultado, libro, hoja, roles, tipo}]


def libros():
    return st.session_state["libros"]


def libro_activo():
    return next((l for l in libros() if l["id"] == st.session_state.get("activo")), None)


def libro_por_id(id_):
    return next((l for l in libros() if l["id"] == id_), None)


def hoja_activa(libro=None):
    libro = libro or libro_activo()
    if not libro or not libro["hojas"]:
        return None
    return libro["hojas"][min(libro.get("hoja", 0), len(libro["hojas"]) - 1)]


def agregar_libro(nombre, hojas):
    libro = {"id": uuid.uuid4().hex[:10], "nombre": nombre, "hojas": hojas, "hoja": 0}
    st.session_state["libros"].append(libro)
    st.session_state["activo"] = libro["id"]
    return libro


def cerrar_libro(id_):
    st.session_state["libros"] = [l for l in libros() if l["id"] != id_]
    if st.session_state.get("activo") == id_:
        st.session_state["activo"] = libros()[-1]["id"] if libros() else None


# ================================================================== lectura
def _nombres_unicos(nombres):
    vistos, salida = set(), []
    for n in nombres:
        base = str(n).strip() or "Columna"
        k, i = base, 2
        while k.lower() in vistos:
            k = f"{base}_{i}"
            i += 1
        vistos.add(k.lower())
        salida.append(k)
    return salida


def _letra(j):
    s, j = "", j + 1
    while j > 0:
        m = (j - 1) % 26
        s = chr(65 + m) + s
        j = (j - 1) // 26
    return s


def _parece_encabezado(filas):
    """Primera fila = nombres de columna si es toda texto (como en la versión de escritorio)."""
    if not filas:
        return False
    f0 = [v for v in filas[0] if v is not None]
    if not f0 or any(not isinstance(v, str) for v in f0):
        return False
    if len(filas) == 1:
        return True
    if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in filas[1]):
        return True
    return len(set(f0)) == len(f0)


def a_dataframe(filas, encabezado=None):
    filas = filas or []
    ancho = max((len(f) for f in filas), default=0)
    if encabezado is None:
        encabezado = _parece_encabezado(filas)
    if encabezado and filas:
        cab = [str(v) if v not in (None, "") else f"Columna {_letra(j)}" for j, v in enumerate(filas[0])]
        filas = filas[1:]
    else:
        cab = []
    cab += [f"Columna {_letra(j)}" for j in range(len(cab), ancho)]
    datos = [list(f) + [None] * (ancho - len(f)) for f in filas]
    df = pd.DataFrame(datos, columns=_nombres_unicos(cab), dtype=object)
    # columnas totalmente numéricas como número (para editar y graficar); el resto queda como texto
    for c in df.columns:
        valores = [v for v in df[c] if v is not None]
        if valores and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in valores):
            df[c] = pd.to_numeric(df[c])
    return df


def abrir_bytes(datos, nombre):
    """Lee un archivo con el motor de JaaZ (Excel, CSV, texto, .ped). Devuelve (libro, separador)."""
    libro = archivos.leer(datos, nombre)
    hojas = [{"nombre": h["nombre"], "df": a_dataframe(h["filas"]), "crudo": h["filas"]} for h in libro["hojas"]]
    return agregar_libro(nombre, hojas), libro.get("separador")


def ejemplos():
    salida = []
    desc = {}
    leeme = os.path.join(CARPETA_EJEMPLOS, "LEEME.txt")
    if os.path.isfile(leeme):
        with open(leeme, encoding="utf-8") as f:
            for linea in f:
                if "|" in linea:
                    n, d = linea.split("|", 1)
                    desc[n.strip()] = d.strip()
    if os.path.isdir(CARPETA_EJEMPLOS):
        for n in sorted(os.listdir(CARPETA_EJEMPLOS)):
            if not n.upper().startswith("LEEME") and n.lower().endswith((".xlsx", ".xls", ".csv", ".txt", ".ped")):
                salida.append({"nombre": n, "descripcion": desc.get(n, "")})
    return salida


def abrir_ejemplo(nombre):
    with open(os.path.join(CARPETA_EJEMPLOS, nombre), "rb") as f:
        return abrir_bytes(f.read(), nombre)


# ================================================================== columnas
def valor_celda(v):
    """Valor de pandas → valor JSON que entiende el motor (None si falta)."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def valores(df, col):
    return [valor_celda(v) for v in df[col].tolist()]


@st.cache_data(show_spinner=False, max_entries=64)
def _tipos(columnas, muestras):
    tipos = []
    for vals in muestras:
        num = txt = 0
        for v in vals:
            if v is None:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                num += 1
            elif str(v).strip().lower() not in FALTANTE_TXT:
                if _NUM_COMA.match(str(v).strip()):
                    num += 1
                else:
                    txt += 1
        tipos.append("vacia" if not num and not txt else "num" if num and txt / (num + txt) <= 0.02 else "texto")
    return tipos


def tipos(df):
    """'num' | 'texto' | 'vacia' por columna (muestra de hasta 20 000 filas)."""
    paso = max(1, len(df) // 20000)
    muestras = tuple(tuple(valor_celda(v) for v in df[c].iloc[::paso].tolist()) for c in df.columns)
    return dict(zip(df.columns, _tipos(tuple(df.columns), muestras)))


def distintos(df, col, maximo=2000):
    return df[col].dropna().astype(str).nunique() if len(df) else 0


PATRONES = {
    "padre": re.compile(r"(^|[^a-z])(padre|sire|toro|semental|father|reproductor|pad|sireid)([^a-z]|$)|sire|padre|toro", re.I),
    "madre": re.compile(r"(^|[^a-z])(madre|dam|vaca|mother|mad|damid)([^a-z]|$)|madre|dam_|_dam|^dam", re.I),
    "sexo": re.compile(r"^(sexo|sex|genero|género|gender)$", re.I),
    "fecha": re.compile(r"fecha|(^|[^a-zñ])a[nñ]o([^a-zñ]|$)|anio|year|nacim|birth|nac$|_nac", re.I),
    "animal": re.compile(r"^(id|animal|individuo|arete|codigo|código|cod|ternero|cria|cría|calf|ind|registro|tatuaje)$"
                         r"|animal|individuo|arete|tatuaje|^id[_ ]|_id$|^id$", re.I),
    "contador": re.compile(r"registro_?num|^n_?reg|lactanc|parto|numero|número|^orden|^edad|^rep|medicion|medición"
                           r"|^dia|^día|semana|^mes", re.I),
    "referencia": re.compile(r"refer", re.I),
    "grupo": re.compile(r"reba[ñn]o|hato|herd|^area|^área|zona|regi[oó]n|subpob|l[ií]nea|granja|finca|fundo|criador"
                        r"|propietario|estrato", re.I),
}


def detectar_roles(df):
    """Propone columnas para cada rol a partir de los nombres y tipos (igual que la versión de escritorio)."""
    cols = list(df.columns)
    tp = tipos(df)
    usado, r = set(), {}

    def buscar(rol, filtro=None):
        for k, c in enumerate(cols):
            if k not in usado and PATRONES[rol].search(c) and (not filtro or filtro(k)):
                r[rol] = c
                usado.add(k)
                return

    buscar("padre", lambda k: not re.search(r"madre|dam", cols[k], re.I))
    buscar("madre")
    buscar("sexo")
    buscar("fecha", lambda k: not PATRONES["contador"].search(cols[k]))
    buscar("animal", lambda k: not PATRONES["padre"].search(cols[k]) and not PATRONES["madre"].search(cols[k]))
    if "animal" not in r and "padre" not in r and len(cols) >= 3 and tp[cols[0]] != "vacia":
        r.update(animal=cols[0], padre=cols[1], madre=cols[2])
        usado.update({0, 1, 2})
    if "animal" not in r and cols and 0 not in usado and tp[cols[0]] == "texto":
        r["animal"] = cols[0]
        usado.add(0)

    def libre(patron):
        return next((c for k, c in enumerate(cols) if k not in usado and patron.search(c) and tp[c] != "vacia"), None)

    r["referencia"] = libre(PATRONES["referencia"])
    r["subpoblacion"] = r["rebano"] = libre(PATRONES["grupo"])
    r["rasgos"] = [c for k, c in enumerate(cols) if k not in usado and tp[c] == "num" and not PATRONES["contador"].search(c)
                   and not PATRONES["fecha"].search(c) and c != r["referencia"] and not PATRONES["grupo"].search(c)]
    return r


# ================================================================== resultados
def agregar_resultado(resultado, meta):
    item = dict(meta, id=uuid.uuid4().hex[:10], resultado=resultado)
    st.session_state["resultados"].insert(0, item)
    del st.session_state["resultados"][100:]
    return item


def parametro(clave, rasgo=None):
    """Último valor de un parámetro (p. ej. h2) para un rasgo, de cualquier análisis de la sesión."""
    for item in st.session_state.get("resultados", []):
        p = item["resultado"].get("parametros") or {}
        if p.get(clave) is not None and (not rasgo or p.get("rasgo") == rasgo):
            return p[clave], item
    return None, None
