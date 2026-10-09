"""JaaZ · Genética cuantitativa en Streamlit.

Archivo principal para Streamlit Community Cloud («Main file path»: app.py) o para usarlo localmente:

    streamlit run app.py
"""
import os

import streamlit as st

from jaaz_st import analisis, estado, estilo, paginas

AQUI = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="JaaZ · Genética cuantitativa", page_icon=os.path.join(AQUI, "recursos", "icono.svg"),
                   layout="wide", initial_sidebar_state="expanded")
estado.iniciar()
estilo.aplicar()
st.logo(os.path.join(AQUI, "recursos", "logo.svg"), icon_image=os.path.join(AQUI, "recursos", "icono.svg"), size="large")


def _analisis(tipo):
    def pagina():
        analisis.pagina(tipo)
    pagina.__name__ = f"pagina_{tipo}"
    return pagina


def _pagina(clave, fn, titulo, icono, **extra):
    return clave, st.Page(fn, title=titulo, icon=icono, url_path=clave, **extra)


SECCIONES = {
    "": [_pagina("inicio", paginas.inicio, "Inicio", ":material/home:", default=True),
         _pagina("datos", paginas.datos, "Hoja de datos", ":material/table_view:"),
         _pagina("descriptiva", _analisis("descriptiva"), "Estadística descriptiva", ":material/bar_chart:")],
    "Parámetros genéticos": [_pagina("heredabilidad", _analisis("heredabilidad"), "Heredabilidad", ":material/genetics:"),
                             _pagina("repetibilidad", _analisis("repetibilidad"), "Repetibilidad", ":material/repeat:"),
                             _pagina("correlacion", _analisis("correlacion"), "Correlación genética", ":material/scatter_plot:")],
    "Pedigrí": [_pagina("visor", paginas.visor, "Visor de pedigrí", ":material/account_tree:"),
                _pagina("pedigri", _analisis("pedigri"), "Calidad y verificación", ":material/verified:"),
                _pagina("parentesco", _analisis("parentesco"), "Parentesco", ":material/family_restroom:"),
                _pagina("apareamientos", _analisis("apareamientos"), "Apareamientos", ":material/join:"),
                _pagina("blup", _analisis("blup"), "Evaluación BLUP", ":material/trending_up:")],
    "Diversidad genética": [_pagina("endogamia", _analisis("endogamia"), "Endogamia y Ne", ":material/hub:"),
                            _pagina("fundadores", _analisis("fundadores"), "Fundadores y ancestros", ":material/park:"),
                            _pagina("intervalos", _analisis("intervalos"), "Intervalo generacional", ":material/schedule:"),
                            _pagina("subpoblaciones", _analisis("subpoblaciones"), "Subpoblaciones", ":material/groups:"),
                            _pagina("rebanos", _analisis("rebanos"), "Estructura de rebaños", ":material/agriculture:")],
    "Informe": [_pagina("resultados", paginas.resultados, "Resultados", ":material/description:"),
                _pagina("ayuda", paginas.ayuda, "Ayuda y métodos", ":material/help:")],
}
st.session_state["_paginas"] = {clave: pagina for grupo in SECCIONES.values() for clave, pagina in grupo}
navegacion = st.navigation({seccion: [p for _, p in grupo] for seccion, grupo in SECCIONES.items()}, expanded=True)

# libro activo y aviso de privacidad en la barra lateral
with st.sidebar:
    activo = estado.libro_activo()
    if activo:
        hoja = estado.hoja_activa(activo)
        st.caption(f"Datos activos: **{activo['nombre']}**" + (f" › {hoja['nombre']}" if hoja else ""))
    n = len(st.session_state["resultados"])
    if n:
        st.caption(f"{n} resultado(s) en esta sesión")
    st.caption(f"JaaZ {paginas.VERSION} · sus datos no se guardan en el servidor")

navegacion.run()
