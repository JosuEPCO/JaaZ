"""Componentes de varianza y covarianza por análisis de varianza (método de
cuadrados medios esperados, Henderson I) para diseños desbalanceados.

Todas las funciones trabajan con una matriz Y (n x t): con t = 1 es un ANOVA,
con t >= 2 se obtienen a la vez las sumas de productos cruzados (ANCOVA/MANOVA),
de modo que varianzas y covarianzas salen de las mismas fórmulas.

Referencias: Becker (1984) Manual of Quantitative Genetics; Falconer y Mackay
(1996); Swiger et al. (1964) J. Anim. Sci. 23:1229; Robertson (1959) Biometrics 15:469.
"""
import math

import numpy as np
from scipy import stats

from .datos import ErrorDatos


def _sumas_por_grupo(codigos, Y, k):
    n = np.bincount(codigos, minlength=k).astype(float)
    S = np.column_stack([np.bincount(codigos, weights=Y[:, j], minlength=k) for j in range(Y.shape[1])])
    return n, S


def _como_matriz(Y):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return Y - Y.mean(axis=0)  # centrar evita la cancelación numérica (rasgos como 6000 kg)


def anova_una_via(grupos, Y):
    """Modelo y_ij = mu + g_i + e_ij con g aleatorio.

    Devuelve un dict con las fuentes de variación (SC/PC, GL, CM como matrices t x t),
    el coeficiente k y las matrices de componentes {"entre", "dentro"}.
    """
    Y = _como_matriz(Y)
    grupos = np.asarray(grupos)
    N, t = Y.shape
    g = int(grupos.max()) + 1
    ni, S = _sumas_por_grupo(grupos, Y, g)
    if g < 2:
        raise ErrorDatos("Se necesitan al menos 2 grupos (p. ej., 2 padres) para el análisis.")
    if N - g < 1:
        raise ErrorDatos("Cada grupo tiene un solo registro: no hay variación dentro de grupos.")
    sp_total = Y.T @ Y
    sp_entre = (S / ni[:, None]).T @ S
    sp_dentro = sp_total - sp_entre
    gl_entre, gl_dentro = g - 1, N - g
    cm_entre, cm_dentro = sp_entre / gl_entre, sp_dentro / gl_dentro
    k = (N - (ni ** 2).sum() / N) / gl_entre
    comp_entre = (cm_entre - cm_dentro) / k
    return {
        "N": N, "grupos": g, "k": k, "n_por_grupo": ni,
        "fuentes": [
            {"fuente": "entre", "gl": gl_entre, "sc": sp_entre, "cm": cm_entre},
            {"fuente": "dentro", "gl": gl_dentro, "sc": sp_dentro, "cm": cm_dentro},
        ],
        "total": {"gl": N - 1, "sc": sp_total},
        "componentes": {"entre": comp_entre, "dentro": cm_dentro},
        # componente_i = sum_j M[i, j] * CM_j  (para errores estándar por método delta)
        "coef": np.array([[1 / k, -1 / k], [0.0, 1.0]]),
        "esperanzas": ["σ²e + k·σ²g", "σ²e"],
    }


def anova_anidado(grupos, subgrupos, Y):
    """Modelo jerárquico y_ijk = mu + a_i + b_ij + e_ijk (b anidado en a), ambos aleatorios.

    `subgrupos` debe tener un código único por combinación (grupo, subgrupo).
    """
    Y = _como_matriz(Y)
    grupos = np.asarray(grupos)
    subgrupos = np.asarray(subgrupos)
    N, t = Y.shape
    a = int(grupos.max()) + 1
    b = int(subgrupos.max()) + 1
    # cada subgrupo pertenece a un solo grupo
    padre_de = np.full(b, -1)
    padre_de[subgrupos] = grupos
    if np.any(padre_de[subgrupos] != grupos):
        raise ErrorDatos("Un mismo subgrupo aparece en más de un grupo (revise la codificación anidada).")
    na, Sa = _sumas_por_grupo(grupos, Y, a)
    nb, Sb = _sumas_por_grupo(subgrupos, Y, b)
    gl_a, gl_b, gl_w = a - 1, b - a, N - b
    if a < 2:
        raise ErrorDatos("Se necesitan al menos 2 grupos (padres) para el análisis anidado.")
    if gl_b < 1:
        raise ErrorDatos("Cada grupo tiene un solo subgrupo: no se puede separar el componente anidado.")
    if gl_w < 1:
        raise ErrorDatos("Cada subgrupo tiene un solo registro: no hay variación dentro de subgrupos.")
    sp_total = Y.T @ Y
    t_a = (Sa / na[:, None]).T @ Sa
    t_b = (Sb / nb[:, None]).T @ Sb
    sp_a, sp_b, sp_w = t_a, t_b - t_a, sp_total - t_b
    cm_a, cm_b, cm_w = sp_a / gl_a, sp_b / gl_b, sp_w / gl_w
    # coeficientes de los cuadrados medios esperados (diseño desbalanceado)
    suma_nb2_por_a = np.bincount(padre_de, weights=nb ** 2, minlength=a)
    s1 = (suma_nb2_por_a / na).sum()
    k1 = (N - s1) / gl_b
    k2 = (s1 - (nb ** 2).sum() / N) / gl_a
    k3 = (N - (na ** 2).sum() / N) / gl_a
    comp_w = cm_w
    comp_b = (cm_b - cm_w) / k1
    comp_a = (cm_a - cm_w - k2 * comp_b) / k3
    coef = np.array([
        [1 / k3, -k2 / (k1 * k3), (k2 / k1 - 1) / k3],
        [0.0, 1 / k1, -1 / k1],
        [0.0, 0.0, 1.0],
    ])
    # prueba F aproximada para el grupo (denominador sintético de Satterthwaite)
    return {
        "N": N, "grupos": a, "subgrupos": b, "k1": k1, "k2": k2, "k3": k3,
        "fuentes": [
            {"fuente": "grupo", "gl": gl_a, "sc": sp_a, "cm": cm_a},
            {"fuente": "subgrupo", "gl": gl_b, "sc": sp_b, "cm": cm_b},
            {"fuente": "dentro", "gl": gl_w, "sc": sp_w, "cm": cm_w},
        ],
        "total": {"gl": N - 1, "sc": sp_total},
        "componentes": {"grupo": comp_a, "subgrupo": comp_b, "dentro": comp_w},
        "coef": coef,
        "esperanzas": ["σ²e + k2·σ²b + k3·σ²a", "σ²e + k1·σ²b", "σ²e"],
    }


def pruebas_f(res, j=0):
    """Estadísticos F y valores p para el rasgo j."""
    f = res["fuentes"]
    cm = [x["cm"][j, j] for x in f]
    gl = [x["gl"] for x in f]
    salida = []
    if len(f) == 2:
        F = cm[0] / cm[1] if cm[1] > 0 else math.nan
        salida.append((F, stats.f.sf(F, gl[0], gl[1]) if F == F else math.nan, gl[0], gl[1]))
        salida.append(None)
    else:
        k1, k2 = res["k1"], res["k2"]
        # subgrupo vs dentro: exacta
        Fb = cm[1] / cm[2] if cm[2] > 0 else math.nan
        # grupo: denominador sintético c·CMb + (1-c)·CMw con gl de Satterthwaite
        c = k2 / k1
        den = c * cm[1] + (1 - c) * cm[2]
        gl_den = den ** 2 / ((c * cm[1]) ** 2 / gl[1] + ((1 - c) * cm[2]) ** 2 / gl[2]) if den > 0 else math.nan
        Fa = cm[0] / den if den > 0 else math.nan
        salida.append((Fa, stats.f.sf(Fa, gl[0], gl_den) if Fa == Fa else math.nan, gl[0], gl_den))
        salida.append((Fb, stats.f.sf(Fb, gl[1], gl[2]) if Fb == Fb else math.nan, gl[1], gl[2]))
        salida.append(None)
    return salida


def varianza_cm(res, j=0, l=None):
    """Varianzas muestrales aproximadas de los CM (independientes): 2·CM²/(gl+2).
    Para productos cruzados: (CM_jl² + CM_jj·CM_ll)/(gl+2)."""
    if l is None:
        l = j
    out = []
    for x in res["fuentes"]:
        cm = x["cm"]
        out.append((cm[j, l] ** 2 + cm[j, j] * cm[l, l]) / (x["gl"] + 2))
    return np.array(out)


def delta_cm(res, funcion, j=0):
    """Error estándar de funcion(componentes del rasgo j) por el método delta sobre los CM.

    `funcion` recibe el vector de componentes (en el orden de res["componentes"]).
    """
    cm = np.array([x["cm"][j, j] for x in res["fuentes"]])
    var = varianza_cm(res, j)
    M = res["coef"]

    def g(cms):
        return funcion(M @ cms)

    valor = g(cm)
    grad = np.zeros_like(cm)
    for i in range(len(cm)):
        h = max(abs(cm[i]) * 1e-6, 1e-12)
        e = np.zeros_like(cm)
        e[i] = h
        grad[i] = (g(cm + e) - g(cm - e)) / (2 * h)
    return valor, math.sqrt(max(float(grad @ (var * grad)), 0.0))


def ee_swiger(t, N, s, k):
    """Error estándar de la correlación intraclase t (Swiger et al., 1964)."""
    if N - s <= 0 or s <= 1 or k <= 0:
        return math.nan
    v = 2 * (N - 1) * (1 - t) ** 2 * (1 + (k - 1) * t) ** 2 / (k ** 2 * (N - s) * (s - 1))
    return math.sqrt(v) if v >= 0 else math.nan


def ic_intraclase(res, j=0, nivel=0.95):
    """Intervalo de confianza de la correlación intraclase a partir de la F (Searle, 1971)."""
    f = res["fuentes"]
    cm_g, cm_w = f[0]["cm"][j, j], f[1]["cm"][j, j]
    if cm_w <= 0:
        return (math.nan, math.nan)
    F = cm_g / cm_w
    a = (1 - nivel) / 2
    k = res["k"]
    fu = stats.f.ppf(1 - a, f[0]["gl"], f[1]["gl"])
    fl = stats.f.ppf(a, f[0]["gl"], f[1]["gl"])
    FL, FU = F / fu, F / fl
    return ((FL - 1) / (FL + k - 1), (FU - 1) / (FU + k - 1))


def ee_correlacion_genetica(rg, h2x, ee_h2x, h2y, ee_h2y):
    """Error estándar aproximado de rG (Robertson, 1959; Falconer y Mackay, 1996)."""
    if not (h2x > 0 and h2y > 0) or any(map(lambda v: v != v, (rg, ee_h2x, ee_h2y))):
        return math.nan
    return (1 - rg ** 2) / math.sqrt(2) * math.sqrt(ee_h2x * ee_h2y / (h2x * h2y))
