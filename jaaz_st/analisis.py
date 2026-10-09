"""Página genérica de análisis: arma el formulario desde definiciones.DEF, valida, envía al motor
solo las columnas necesarias, muestra el progreso del cálculo y el resultado.

Los valores del formulario se conservan al cambiar de página (persist_state="session")."""
import re
import time
import traceback

import streamlit as st

from motor.genetica import descriptiva, genealogia, parametros, poblacion
from motor.genetica.datos import ErrorDatos

from . import estado, vista
from .definiciones import DEF
from .estilo import encabezado

FUNCIONES = {
    "heredabilidad": parametros.heredabilidad, "repetibilidad": parametros.repetibilidad, "correlacion": parametros.correlacion,
    "descriptiva": descriptiva.descriptiva, "pedigri": genealogia.pedigri, "parentesco": genealogia.parentesco,
    "apareamientos": genealogia.apareamientos, "blup": genealogia.blup, "endogamia": poblacion.endogamia,
    "fundadores": poblacion.fundadores, "intervalos": poblacion.intervalo_generacional,
    "subpoblaciones": poblacion.subpoblaciones, "rebanos": poblacion.rebanos,
}
PERSISTIR = dict(persist_state="session")
NINGUNA = "__ninguna__"


class Contexto:
    """Lo que el motor recibe para informar el avance (la cancelación la hace el botón Stop de Streamlit)."""

    def __init__(self, caja, barra):
        self.caja, self.barra, self._t = caja, barra, 0.0

    def progreso(self, fraccion=None, mensaje=None):
        ahora = time.time()
        if ahora - self._t < 0.25 and fraccion not in (0, 1):
            return
        self._t = ahora
        if fraccion is not None:
            self.barra.progress(max(0.0, min(1.0, float(fraccion))))
        if mensaje:
            self.caja.update(label=mensaje)

    def cancelado(self):
        return False


def lista_ids(texto):
    return [x for x in re.split(r"[\s,;]+", str(texto or "")) if x]


def _sanear(clave, opciones, multiple=False):
    """Quita un valor recordado que ya no es válido (p. ej. columna de otra hoja)."""
    if clave not in st.session_state:
        return
    v = st.session_state[clave]
    if multiple:
        validos = [x for x in (v or []) if x in opciones]
        if validos != v:
            st.session_state[clave] = validos
    elif v not in opciones:
        del st.session_state[clave]


def _selector_hoja(clave, etiqueta="Libro y hoja"):
    opciones = [(l["id"], k) for l in estado.libros() for k in range(len(l["hojas"]))]
    if not opciones:
        return None, None
    _sanear(clave, opciones)
    activo = estado.libro_activo()
    defecto = (activo["id"], activo.get("hoja", 0)) if activo else opciones[0]
    nombres = {(l["id"], k): f'{l["nombre"]} › {h["nombre"]}' for l in estado.libros() for k, h in enumerate(l["hojas"])}
    elegido = st.selectbox(etiqueta, opciones, index=opciones.index(defecto) if defecto in opciones else 0,
                           format_func=nombres.get, key=clave, **PERSISTIR)
    libro = estado.libro_por_id(elegido[0])
    return libro, libro["hojas"][elegido[1]]


def _selector_rol(rol, df, tipos, clave, det):
    cols = [c for c in df.columns if rol["tipo"] != "num" or tipos.get(c) == "num"]
    opciones = ([] if rol["requerido"] else [NINGUNA]) + cols
    _sanear(clave, opciones)
    if rol.get("auto", True):
        sugerido = det.get("rasgos", [None])[0] if rol["id"] == "rasgo" and det.get("rasgos") else (
            det.get("animal") if rol["id"] == "id" else det.get(rol["id"]))
    else:
        sugerido = None
    indice = opciones.index(sugerido) if sugerido in opciones else (0 if opciones and opciones[0] == NINGUNA else None)
    v = st.selectbox(rol["etiqueta"] + (" *" if rol["requerido"] else ""), opciones, index=indice,
                     format_func=lambda c: "— ninguna —" if c == NINGUNA else f'{c}  · {"123" if tipos.get(c) == "num" else "abc"}',
                     key=clave, help=rol.get("ayuda"), placeholder="— elegir —", **PERSISTIR)
    return None if v in (None, NINGUNA) else v


def _numero(texto):
    s = str(texto or "").strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        raise ValueError


def formulario(tipo, d):
    """Dibuja el formulario. Devuelve (payload, errores, meta) o None si no hay datos."""
    libro, hoja = _selector_hoja(f"{tipo}:datos")
    if not libro:
        st.info("No hay datos abiertos. Abra un archivo o un ejemplo en «Hoja de datos».")
        st.page_link(st.session_state["_paginas"]["datos"], label="Ir a Hoja de datos", icon=":material/table_view:")
        return None
    df = hoja["df"]
    st.caption(f"{len(df):,} filas · {len(df.columns)} columnas")
    tipos = estado.tipos(df)
    det = estado.detectar_roles(df)

    metodos = d["metodos"]
    if len(metodos) > 1:
        ids = [m["id"] for m in metodos]
        _sanear(f"{tipo}:metodo", ids)
        mid = st.radio("Método", ids, format_func=lambda i: next(m["nombre"] + (f'  ·  {m["etiqueta"]}' if m.get("etiqueta") else "")
                                                               for m in metodos if m["id"] == i),
                       captions=[m.get("resumen", "") for m in metodos], key=f"{tipo}:metodo", **PERSISTIR)
    else:
        mid = metodos[0]["id"]
    m = next(x for x in metodos if x["id"] == mid)
    base = f"{tipo}:{mid}"

    st.markdown("**Columnas**")
    roles = {}
    for rol in m["roles"]:
        roles[rol["id"]] = _selector_rol(rol, df, tipos, f"{base}:rol:{rol['id']}", det)
    multi = []
    if m.get("multi"):
        mm = m["multi"]
        usadas = {v for v in roles.values() if v}
        candidatos = [c for c in df.columns if (mm["tipo"] != "num" or tipos.get(c) == "num") and c not in usadas]
        clave = f"{base}:multi"
        _sanear(clave, candidatos, multiple=True)
        defecto = [c for c in det.get("rasgos", []) if c in candidatos][:max(mm["min"], 2 if mm["clave"] == "rasgos" else 8)]
        multi = st.multiselect(f'{mm["etiqueta"]} * ({mm["min"]}' + (f' a {mm["max"]})' if mm["max"] != mm["min"] else ")"),
                               candidatos, default=defecto, max_selections=mm["max"], key=clave, **PERSISTIR)

    fijos, covariables = [], []
    if m.get("fijos"):
        with st.expander("Efectos fijos (opcional)", icon=":material/tune:"):
            usadas = {v for v in roles.values() if v} | set(multi) | {det.get(k) for k in ("animal", "padre", "madre") if det.get(k)}
            disponibles = [c for c in df.columns if c not in usadas]
            factores = [c for c in disponibles if (estado.distintos(df, c) <= 60 if tipos.get(c) == "num" else estado.distintos(df, c) <= 2000)]
            _sanear(f"{tipo}:fijos", factores, multiple=True)
            fijos = st.multiselect("Factores (hato, año, sexo, n.º de parto…)", factores, key=f"{tipo}:fijos", **PERSISTIR)
            nums = [c for c in disponibles if tipos.get(c) == "num" and c not in fijos]
            _sanear(f"{tipo}:cov", nums, multiple=True)
            covariables = st.multiselect("Covariables (regresión lineal: edad, peso…)", nums, key=f"{tipo}:cov", **PERSISTIR)

    pedigri = None
    if m.get("pedigri"):
        if st.toggle("Usar un pedigrí de otra hoja", key=f"{tipo}:ped", help="Si no, el pedigrí se arma con las columnas animal/padre/madre de esta hoja.",
                     **PERSISTIR):
            lp, hp = _selector_hoja(f"{tipo}:ped:hoja", "Hoja del pedigrí")
            dp = hp["df"]
            tp, detp = estado.tipos(dp), estado.detectar_roles(dp)
            c1, c2, c3 = st.columns(3)
            with c1:
                pid = _selector_rol(dict(id="animal", etiqueta="ID", tipo="id", requerido=True), dp, tp, f"{tipo}:ped:id", detp)
            with c2:
                ppa = _selector_rol(dict(id="padre", etiqueta="Padre", tipo="id", requerido=False), dp, tp, f"{tipo}:ped:padre", detp)
            with c3:
                pma = _selector_rol(dict(id="madre", etiqueta="Madre", tipo="id", requerido=False), dp, tp, f"{tipo}:ped:madre", detp)
            proles = {k: v for k, v in (("id", pid), ("padre", ppa), ("madre", pma)) if v}
            pedigri = {"tabla": {"n": len(dp), "valores": {c: estado.valores(dp, c) for c in set(proles.values())}}, "roles": proles}

    opciones, errores = {}, []
    if m.get("opciones") or m.get("confianza") or m.get("fijos") or m.get("faltantes"):
        st.markdown("**Opciones**")
    for o in m.get("opciones", []):
        clave = f"{base}:op:{o['clave']}"
        if o["tipo"] == "select":
            valores = [v for v, _ in o["opciones"]]
            _sanear(clave, valores)
            etiquetas = dict(o["opciones"])
            v = st.selectbox(o["etiqueta"], valores, index=valores.index(o.get("defecto", valores[0])), format_func=etiquetas.get,
                             key=clave, **PERSISTIR)
        elif o["tipo"] == "check":
            v = st.toggle(o["etiqueta"], value=bool(o.get("defecto")), key=clave, **PERSISTIR)
        elif o["tipo"] == "ids":
            texto = st.text_area(o["etiqueta"] + (" *" if o.get("requerido") else ""), key=clave, help=o.get("ayuda"),
                                 placeholder="A001, A002, A003…", height=90, **PERSISTIR)
            v = lista_ids(texto)
            st.caption(f"{len(v)} elegidos")
        else:  # número (admite coma decimal y vacío)
            if o.get("sugerir"):
                sugerido, origen = estado.parametro(o["sugerir"], roles.get("rasgo"))
                if sugerido is None:
                    sugerido, origen = estado.parametro(o["sugerir"])
                if isinstance(sugerido, (int, float)) and st.button(
                        f"Usar {sugerido:.3f} ({origen['resultado']['titulo'].split('·')[0].strip()})", key=f"{clave}:sug", type="tertiary"):
                    st.session_state[clave] = f"{sugerido:.4f}"
            if clave not in st.session_state and o.get("defecto") is not None:
                st.session_state[clave] = str(o["defecto"])
            texto = st.text_input(o["etiqueta"] + (" *" if o.get("requerido") else ""), key=clave, help=o.get("ayuda"), **PERSISTIR)
            try:
                v = _numero(texto)
            except ValueError:
                errores.append(f"«{o['etiqueta']}» debe ser un número.")
                v = None
        if o.get("requerido") and (v is None or v == []):
            errores.append(f"Complete «{o['etiqueta']}».")
        opciones[o["clave"]] = v
    if m.get("fijos") or m.get("faltantes"):
        falt = st.text_input("Código de dato faltante (además de celdas vacías)", key=f"{tipo}:faltantes", placeholder="p. ej. -9999",
                             help="Las celdas vacías, «.», «NA» y «-» ya se tratan como faltantes.", **PERSISTIR)
        codigos = []
        for x in re.split(r"[\s,;]+", falt or ""):
            if x:
                try:
                    codigos.append(float(x))
                except ValueError:
                    errores.append(f"El código de faltante «{x}» no es un número.")
        if codigos:
            opciones["faltantes"] = codigos
    if m.get("confianza"):
        opciones["confianza"] = st.selectbox("Nivel de confianza", [0.90, 0.95, 0.99], index=1, format_func=lambda c: f"{c:.0%}",
                                             key=f"{tipo}:confianza", **PERSISTIR)
    if m.get("fijos"):
        opciones["fijos"], opciones["covariables"] = list(fijos), list(covariables)
    if m.get("multi"):
        opciones[m["multi"]["clave"]] = list(multi)

    # validación
    for rol in m["roles"]:
        if rol["requerido"] and not roles.get(rol["id"]):
            errores.append(f"Elija la columna «{rol['etiqueta']}».")
    if m.get("uno_de") and not any(roles.get(k) for k in m["uno_de"]):
        errores.append("Elija al menos la columna de padre o la de madre.")
    if m.get("multi") and len(multi) < m["multi"]["min"]:
        errores.append(f"Elija al menos {m['multi']['min']} {m['multi']['etiqueta'].lower()}.")
    usados = {k: v for k, v in roles.items() if v}
    vistos = {}
    for k, c in usados.items():
        if c in vistos:
            errores.append(f"La columna «{c}» está asignada a «{vistos[c]}» y a «{k}».")
        vistos[c] = k
    columnas = set(usados.values()) | set(fijos) | set(covariables) | set(multi)
    payload = {"tipo": tipo, "metodo": mid, "tabla": {"n": len(df), "valores": {c: estado.valores(df, c) for c in columnas}},
               "roles": usados, "opciones": opciones}
    if pedigri:
        if "id" not in pedigri["roles"]:
            errores.append("Elija la columna de ID del pedigrí.")
        payload["pedigri"] = pedigri
    meta = {"tipo": tipo, "metodo": mid, "libro": libro["nombre"], "libro_id": libro["id"], "hoja": hoja["nombre"], "roles": usados}
    return payload, errores, meta


def calcular(payload, meta, zona):
    with zona:
        caja = st.status("Calculando…", expanded=True)
        with caja:
            barra = st.progress(0.0)
        t0 = time.time()
        try:
            res = FUNCIONES[payload["tipo"]](payload, Contexto(caja, barra))
        except ErrorDatos as e:
            caja.update(label="No se pudo completar el análisis", state="error", expanded=False)
            st.error(f"**No se pudo completar el análisis.** {e}")
            return None
        except Exception as e:  # error inesperado: se muestra el detalle para poder reportarlo
            caja.update(label="Error inesperado", state="error", expanded=False)
            st.error(f"Error inesperado en el cálculo ({type(e).__name__}: {e}).")
            with st.expander("Detalle técnico"):
                st.code(traceback.format_exc())
            return None
        caja.update(label=f"Listo en {time.time() - t0:.1f} s", state="complete", expanded=False)
    item = estado.agregar_resultado(res, meta)
    st.session_state[f"ultimo:{payload['tipo']}"] = item["id"]
    return item


def pagina(tipo):
    d = DEF[tipo]
    encabezado(d["eyebrow"], d["titulo"], d["sub"])
    ocultar = st.toggle("Ocultar configuración", key="cfg_oculta", help="Da todo el ancho al resultado.", **PERSISTIR)
    if ocultar:
        zona_res = st.container()
        armado = None
    else:
        col_cfg, zona_res = st.columns([0.34, 0.66], gap="large")
        with col_cfg.container(border=True):
            armado = formulario(tipo, d)
            if armado:
                payload, errores, meta = armado
                if st.button("Calcular", type="primary", icon=":material/play_arrow:", width="stretch", key=f"{tipo}:calcular"):
                    if errores:
                        st.error(errores[0])
                    else:
                        calcular(payload, meta, zona_res)
    ultimo = st.session_state.get(f"ultimo:{tipo}")
    item = next((r for r in st.session_state["resultados"] if r["id"] == ultimo), None)
    with zona_res:
        if item:
            vista.mostrar(item)
        elif not ocultar:
            with st.container(border=True):
                st.markdown(f"#### {d['metodos'][0]['nombre'] if len(d['metodos']) == 1 else 'Sin resultados todavía'}")
                st.caption("Asigne las columnas en el panel de configuración y pulse «Calcular». El resultado queda en «Resultados» "
                           "y se puede exportar a Excel.")
        else:
            st.info("Muestre la configuración para calcular.")
