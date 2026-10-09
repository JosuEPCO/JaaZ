"""Estructura común de los resultados que se muestran en el navegador.

Cada análisis devuelve un dict serializable a JSON:
    titulo, subtitulo, indicadores[], tablas[], graficos[], notas[], formulas[],
    columnas_nuevas[] (para agregar a la hoja), parametros{} (valores crudos).
"""
import math
from datetime import datetime

import numpy as np


def limpio(v):
    """Convierte tipos de numpy y NaN/inf a valores JSON."""
    if isinstance(v, dict):
        return {str(k): limpio(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [limpio(x) for x in v]
    if isinstance(v, np.ndarray):
        return [limpio(x) for x in v.tolist()]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, np.bool_):
        return bool(v)
    return v


class Resultado:
    def __init__(self, tipo, metodo, titulo, subtitulo=""):
        self.d = {
            "tipo": tipo, "metodo": metodo, "titulo": titulo, "subtitulo": subtitulo,
            "fecha": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "indicadores": [], "tablas": [], "graficos": [], "notas": [], "formulas": [],
            "columnas_nuevas": [], "parametros": {},
        }

    def indicador(self, etiqueta, valor, ee=None, ic=None, decimales=3, nivel=None, descripcion=None,
                  destacado=False, unidad=None):
        self.d["indicadores"].append({
            "etiqueta": etiqueta, "valor": valor, "ee": ee, "ic": list(ic) if ic is not None else None,
            "decimales": decimales, "nivel": nivel, "descripcion": descripcion, "destacado": destacado,
            "unidad": unidad,
        })

    def tabla(self, titulo, columnas, filas, nota=None, id=None, exportar_completa=None):
        """columnas: lista de (clave, titulo, tipo, decimales) con tipo en
        texto | numero | entero | p | porcentaje."""
        cols = [{"clave": c[0], "titulo": c[1], "tipo": c[2] if len(c) > 2 else "texto",
                 "decimales": c[3] if len(c) > 3 else 3} for c in columnas]
        self.d["tablas"].append({"id": id or f"t{len(self.d['tablas']) + 1}", "titulo": titulo,
                                 "columnas": cols, "filas": filas, "nota": nota,
                                 "total_filas": exportar_completa if exportar_completa is not None else len(filas)})

    def grafico(self, tipo, titulo, **datos):
        self.d["graficos"].append({"tipo": tipo, "titulo": titulo, **datos})

    def nota(self, texto, tipo="info"):
        # sin repetidos: los análisis por pares (REML bivariado) generan la misma nota varias veces
        if texto and not any(n["texto"] == texto for n in self.d["notas"]):
            self.d["notas"].append({"tipo": tipo, "texto": texto})

    def formula(self, titulo, expresion):
        self.d["formulas"].append({"titulo": titulo, "expresion": expresion})

    def columna_nueva(self, nombre, clave_rol, valores, descripcion=""):
        """valores: dict id -> valor. clave_rol: columna de la hoja que identifica la fila."""
        self.d["columnas_nuevas"].append({"nombre": nombre, "clave": clave_rol, "valores": valores,
                                          "descripcion": descripcion})

    def parametro(self, clave, valor):
        self.d["parametros"][clave] = valor

    def json(self):
        return limpio(self.d)


# ---------------------------------------------------------------- interpretación
def nivel_h2(h2):
    if h2 is None or h2 != h2:
        return None
    if h2 < 0:
        return ("Fuera de rango", "error")
    if h2 < 0.10:
        return ("Muy baja", "baja")
    if h2 < 0.20:
        return ("Baja", "baja")
    if h2 <= 0.40:
        return ("Media", "media")
    if h2 <= 1.0:
        return ("Alta", "alta")
    return ("Fuera de rango", "error")


def nivel_r(r):
    if r is None or r != r:
        return None
    if r < 0 or r > 1:
        return ("Fuera de rango", "error")
    if r < 0.20:
        return ("Baja", "baja")
    if r <= 0.40:
        return ("Media", "media")
    return ("Alta", "alta")


def nivel_rg(rg):
    if rg is None or rg != rg:
        return None
    a = abs(rg)
    if a > 1:
        return ("Fuera de rango", "error")
    signo = "positiva" if rg >= 0 else "negativa"
    if a < 0.20:
        return (f"Débil {signo}", "baja")
    if a < 0.50:
        return (f"Moderada {signo}", "media")
    return (f"Fuerte {signo}", "alta")


def texto_h2(h2):
    n = nivel_h2(h2)
    if not n:
        return None
    if n[1] == "error":
        return ("La estimación cae fuera del intervalo [0, 1]: suele deberse a pocos grupos o pocos "
                "registros por grupo, o a efectos no considerados en el modelo.")
    return {
        "baja": "La mayor parte de la variación es ambiental: la selección por fenotipo propio avanza "
                "despacio; conviene usar información de parientes (pruebas de progenie, BLUP) y mejorar el manejo.",
        "media": "Parte importante de la variación es genética aditiva: la selección individual es efectiva, "
                 "y aún mejor si se combina con información de parientes.",
        "alta": "La variación es mayormente genética aditiva: la selección por fenotipo propio da una "
                "respuesta rápida y confiable.",
    }[n[1]]


def texto_r(r):
    n = nivel_r(r)
    if not n:
        return None
    if n[1] == "error":
        return "La estimación cae fuera del intervalo [0, 1]."
    return {
        "baja": "Un solo registro predice mal los siguientes: conviene promediar varios registros por animal "
                "antes de decidir descartes.",
        "media": "Repetir mediciones mejora notablemente la precisión; con 2 a 3 registros ya se gana mucho.",
        "alta": "El primer registro predice bien la producción futura: se puede seleccionar o descartar "
                "temprano con buena confianza.",
    }[n[1]]


def texto_rg(rg):
    n = nivel_rg(rg)
    if not n:
        return None
    if n[1] == "error":
        return "La estimación cae fuera de [−1, 1]: es poco confiable (pocos padres o varianzas muy pequeñas)."
    base = ("Seleccionar por un rasgo produce una respuesta correlacionada en el otro en la misma dirección."
            if rg >= 0 else
            "Seleccionar para aumentar un rasgo tiende a disminuir el otro (antagonismo genético).")
    if n[1] == "baja":
        base = "Los rasgos están casi controlados por genes distintos: la selección por uno apenas afecta al otro."
    return base


def resumen_numerico(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"n": 0}
    return {"n": len(x), "media": float(x.mean()), "de": float(x.std(ddof=1)) if len(x) > 1 else math.nan,
            "min": float(x.min()), "max": float(x.max())}


def histograma(x, bins=None):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"bordes": [], "conteos": []}
    if bins is None:
        q75, q25 = np.percentile(x, [75, 25])
        ancho = 2 * (q75 - q25) / len(x) ** (1 / 3)
        bins = int(np.clip(math.ceil((x.max() - x.min()) / ancho) if ancho > 0 else 10, 5, 40))
    conteos, bordes = np.histogram(x, bins=bins)
    return {"bordes": bordes.tolist(), "conteos": conteos.tolist(), "media": float(x.mean()),
            "de": float(x.std(ddof=1)) if len(x) > 1 else None, "n": int(len(x))}
