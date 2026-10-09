"""Pruebas de la app de Streamlit: abre cada página, elige cada método, pulsa «Calcular» y verifica
que no haya excepciones ni errores y que aparezca el resultado.

    .venv\\Scripts\\python -m unittest discover -s pruebas -v
"""
import os
import unittest

from streamlit.testing.v1 import AppTest

os.environ["JAAZ_APP"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pagina(tipo, ejemplo, hoja, opciones):
    """Script de prueba: abre un ejemplo, deja activa la hoja pedida y dibuja la página."""
    import os
    import sys

    import streamlit as st

    if os.environ["JAAZ_APP"] not in sys.path:
        sys.path.insert(0, os.environ["JAAZ_APP"])
    from jaaz_st import analisis, estado, estilo, paginas

    estado.iniciar()
    estilo.aplicar()
    if not estado.libros():
        libro, _ = estado.abrir_ejemplo(ejemplo)
        libro["hoja"] = [h["nombre"] for h in libro["hojas"]].index(hoja)
    for clave, valor in opciones.items():
        st.session_state.setdefault(clave, valor)
    if tipo in ("inicio", "datos", "visor", "resultados", "ayuda"):
        getattr(paginas, tipo)()
    else:
        analisis.pagina(tipo)


CASOS = {
    # tipo: (ejemplo, hoja, métodos, valores iniciales de widgets)
    "descriptiva": ("hato_lechero_simulado.xlsx", "Lactancias", None, {}),
    # en el hato, los padres (toros) no tienen registros: la regresión y la covarianza cruzada usan la madre
    "heredabilidad": ("hato_lechero_simulado.xlsx", "Lactancias",
                      ["medios_hermanos", "hermanos_completos", "regresion", "reml_padre", "reml_animal"],
                      {"heredabilidad:regresion:op:tipo_regresion": "madre"}),
    "repetibilidad": ("hato_lechero_simulado.xlsx", "Lactancias", ["anova", "anova_anidado", "reml", "reml_animal"], {}),
    "correlacion": ("hato_lechero_simulado.xlsx", "Lactancias", ["ancova", "reml_padre", "cruzada"],
                    {"correlacion:cruzada:op:tipo_regresion": "madre"}),
    "blup": ("hato_lechero_simulado.xlsx", "Lactancias", None, {"blup:blup:op:h2": "0.3", "blup:blup:op:r": "0.45"}),
    "pedigri": ("endog_ejemplo.xls", "non ordered", None, {}),
    "endogamia": ("endog_ejemplo.xls", "non ordered", None, {}),
    "fundadores": ("endog_ejemplo.xls", "non ordered", None, {}),
    "intervalos": ("endog_ejemplo.xls", "non ordered", None, {}),
    "subpoblaciones": ("endog_ejemplo.xls", "non ordered", None, {"subpoblaciones:subpoblaciones:rol:subpoblacion": "AREA"}),
    "rebanos": ("endog_ejemplo.xls", "non ordered", None, {"rebanos:rebanos:rol:rebano": "AREA"}),
    "parentesco": ("hato_lechero_simulado.xlsx", "Pedigri", None, {"parentesco:parentesco:op:animales": "T002, F170001, V0001"}),
    "apareamientos": ("hato_lechero_simulado.xlsx", "Pedigri", None, {}),
}


class TestPaginas(unittest.TestCase):
    def _correr(self, at):
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def test_app_completa(self):
        # el archivo principal con el menú: abre en Inicio
        at = self._correr(AppTest.from_file(os.path.join(os.environ["JAAZ_APP"], "app.py"), default_timeout=60))
        self.assertIn("_paginas", at.session_state)

    def test_paginas_sin_analisis(self):
        for tipo in ("datos", "visor", "resultados", "ayuda"):
            with self.subTest(pagina=tipo):
                self._correr(AppTest.from_function(_pagina, args=(tipo, "pedigri_hato.ped", "pedigri_hato", {}), default_timeout=60))

    def test_analisis(self):
        for tipo, (ejemplo, hoja, metodos, opciones) in CASOS.items():
            for metodo in metodos or [None]:
                with self.subTest(analisis=tipo, metodo=metodo):
                    at = self._correr(AppTest.from_function(_pagina, args=(tipo, ejemplo, hoja, opciones), default_timeout=300))
                    if metodo:
                        at.radio(key=f"{tipo}:metodo").set_value(metodo)
                        self._correr(at)
                    at.button(key=f"{tipo}:calcular").click()
                    self._correr(at)
                    errores = [e.value for e in at.error]
                    self.assertFalse(errores, errores)
                    self.assertIn(f"ultimo:{tipo}", at.session_state, "no apareció el resultado")


if __name__ == "__main__":
    unittest.main()
