"""Análisis de parámetros genéticos: heredabilidad, repetibilidad y correlación genética.

Cada función recibe el `payload` enviado por el navegador y un contexto con
`progreso(fraccion, mensaje)` y `cancelado()`, y devuelve un dict de Resultado.
"""
import math
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy import stats

from . import anova, reml
from .datos import ErrorDatos, Registros, ajustar_por_fijos, codificar, matriz_fija
from .pedigri import Pedigri
from .resultados import (Resultado, histograma, nivel_h2, nivel_r, nivel_rg, texto_h2, texto_r,
                         texto_rg)

Z95 = 1.959963984540054
LIMITE_FILAS = 200_000  # filas máximas por tabla de resultados (el navegador las pagina)


# ================================================================ utilidades
def _opciones(payload):
    return payload.get("opciones") or {}


def _faltantes(op):
    vals = []
    for v in op.get("faltantes") or []:
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            pass
    return tuple(vals)


def _registros(payload, rasgos, ids, ids_opc=(), con_fijos=True):
    op = _opciones(payload)
    reg = Registros(payload.get("tabla") or {}, payload.get("roles") or {}, rasgos, ids, ids_opc,
                    faltantes=_faltantes(op),
                    fijos=(op.get("fijos") or []) if con_fijos else [],
                    covariables=(op.get("covariables") or []) if con_fijos else [])
    for rol, col in reg.rasgos.items():
        if len(col) > 1 and float(np.ptp(col)) == 0.0:
            raise ErrorDatos(f"«{reg.nombres[rol]}» tiene el mismo valor en todos los registros usados: "
                             "no hay variación que analizar.")
    return reg


def _ids_completos(ids):
    """Un registro sin ID cuenta como un individuo distinto (no se agrupan todos los vacíos en uno)."""
    if ids is None:
        return None
    return [i if i is not None else f"␟sin-id-{k}" for k, i in enumerate(ids)]


def _confianza(op):
    try:
        c = float(op.get("confianza") or 0.95)
    except (TypeError, ValueError):
        c = 0.95
    return min(max(c, 0.5), 0.999)


def _ic(valor, ee, conf=0.95):
    if valor is None or ee is None or valor != valor or ee != ee:
        return None
    z = stats.norm.ppf(0.5 + conf / 2)
    return (valor - z * ee, valor + z * ee)


def _anidar(grupos, subgrupos, nombre_grupo, nombre_sub):
    """Códigos de subgrupo únicos dentro de grupo; avisa si un subgrupo cambia de grupo."""
    grupo_de = {}
    conflictos = set()
    for g, s in zip(grupos, subgrupos):
        previo = grupo_de.setdefault(s, g)
        if previo != g:
            conflictos.add(s)
    combinados = [f"{g}␟{s}" for g, s in zip(grupos, subgrupos)]
    codigos, niveles = codificar(combinados)
    return codigos, niveles, sorted(conflictos, key=str)


def _repetidos(ids):
    vistos = defaultdict(int)
    for i in ids:
        if i is not None:
            vistos[i] += 1
    return sum(1 for v in vistos.values() if v > 1), vistos


def _tabla_anova(res, nombres_fuentes, j=0, etiqueta_total="Total"):
    pf = anova.pruebas_f(res, j)
    filas = []
    for fuente, nombre, prueba, esperanza in zip(res["fuentes"], nombres_fuentes, pf, res["esperanzas"]):
        fila = {"fuente": nombre, "gl": fuente["gl"], "sc": fuente["sc"][j, j], "cm": fuente["cm"][j, j],
                "f": None, "p": None, "esperanza": esperanza}
        if prueba:
            fila["f"], fila["p"] = prueba[0], prueba[1]
        filas.append(fila)
    filas.append({"fuente": etiqueta_total, "gl": res["total"]["gl"], "sc": res["total"]["sc"][j, j],
                  "cm": None, "f": None, "p": None, "esperanza": ""})
    return filas


COLS_ANOVA = [("fuente", "Fuente de variación"), ("gl", "GL", "entero"), ("sc", "Suma de cuadrados", "numero", 4),
              ("cm", "Cuadrado medio", "numero", 4), ("f", "F", "numero", 3), ("p", "Valor p", "p"),
              ("esperanza", "E(CM)")]


def _familias(codigos, etiquetas, y, maximo=400):
    """Media, n y error estándar por grupo (para el gráfico de medias por padre)."""
    k = len(etiquetas)
    n = np.bincount(codigos, minlength=k)
    s = np.bincount(codigos, weights=y, minlength=k)
    s2 = np.bincount(codigos, weights=y * y, minlength=k)
    media = s / np.maximum(n, 1)
    var = np.where(n > 1, (s2 - n * media ** 2) / np.maximum(n - 1, 1), np.nan)
    ee = np.sqrt(np.maximum(var, 0) / np.maximum(n, 1))
    orden = np.argsort(media)
    if len(orden) > maximo:
        paso = len(orden) / maximo
        orden = orden[(np.arange(maximo) * paso).astype(int)]
    return [{"grupo": str(etiquetas[i]), "media": float(media[i]), "ee": float(ee[i]) if ee[i] == ee[i] else None,
             "n": int(n[i])} for i in orden]


def _estructura(nombre, codigos, k):
    n = np.bincount(codigos, minlength=k)
    return {"elemento": nombre, "cantidad": int(k), "min": int(n.min()), "media": float(n.mean()),
            "max": int(n.max())}


COLS_ESTRUCTURA = [("elemento", "Nivel"), ("cantidad", "Cantidad", "entero"), ("min", "Registros mín.", "entero"),
                   ("media", "Registros prom.", "numero", 2), ("max", "Registros máx.", "entero")]

COLS_COMPONENTES = [("componente", "Componente"), ("estimacion", "Estimación", "numero", 4),
                    ("ee", "Error estándar", "numero", 4), ("porcentaje", "% de σ²P", "porcentaje", 1),
                    ("significado", "Interpretación genética")]


def _describir_rasgo(res, y, nombre):
    media = float(np.mean(y))
    de = float(np.std(y, ddof=1)) if len(y) > 1 else math.nan
    desc = f"{nombre} · DE = {de:.4g}"
    if media > 0 and de == de:
        desc += f", CV = {100 * de / media:.1f} %"  # el CV solo tiene sentido en escala de razón
    res.indicador("Media del rasgo", media, decimales=3, descripcion=desc)


# ================================================================ HEREDABILIDAD
def heredabilidad(payload, ctx):
    metodo = payload.get("metodo") or "medios_hermanos"
    funciones = {
        "medios_hermanos": _h2_medios_hermanos,
        "hermanos_completos": _h2_hermanos_completos,
        "regresion": _h2_regresion,
        "reml_padre": _h2_reml_padre,
        "reml_animal": _h2_reml_animal,
    }
    if metodo not in funciones:
        raise ErrorDatos(f"Método de heredabilidad desconocido: {metodo}")
    return funciones[metodo](payload, ctx)


def _h2_medios_hermanos(payload, ctx):
    op = _opciones(payload)
    conf = _confianza(op)
    reg = _registros(payload, ["rasgo"], ["padre"], ["animal"])
    nombre = reg.nombres["rasgo"]
    y, nota_aj = ajustar_por_fijos(reg.rasgos["rasgo"], reg)
    padres = reg.ids["padre"]
    sc, sl = codificar(padres)
    animales = _ids_completos(reg.ids.get("animal"))
    hay_rep = False
    if animales is not None:
        n_rep, _ = _repetidos(animales)
        hay_rep = n_rep > 0

    r = Resultado("heredabilidad", "medios_hermanos",
                  "Heredabilidad por medios hermanos paternos",
                  f"{nombre} · {reg.n} registros · {len(sl)} padres")
    r.nota(reg.nota_descartados(), "advertencia")
    r.nota(nota_aj)

    if not hay_rep:
        ctx.progreso(0.3, "ANOVA de una vía entre padres…")
        res = anova.anova_una_via(sc, y)
        s2s = float(res["componentes"]["entre"][0, 0])
        s2w = float(res["componentes"]["dentro"][0, 0])
        s2p = s2s + s2w
        t = s2s / s2p if s2p > 0 else math.nan
        h2 = 4 * t
        ee_t = anova.ee_swiger(t, res["N"], res["grupos"], res["k"])
        ee_h2 = 4 * ee_t
        ic_t = anova.ic_intraclase(res, 0, conf)
        ic = (4 * ic_t[0], 4 * ic_t[1])
        n_h2 = nivel_h2(h2)
        r.indicador("Heredabilidad (h²)", h2, ee_h2, ic, nivel=n_h2[0] if n_h2 else None,
                    descripcion=texto_h2(h2), destacado=True)
        r.indicador("Correlación intraclase (t)", t, ee_t, ic_t, descripcion="Semejanza entre medios hermanos = ¼ h²")
        r.indicador("Varianza aditiva (σ²A)", 4 * s2s, decimales=4, descripcion="σ²A = 4 σ²padre")
        r.indicador("Varianza fenotípica (σ²P)", s2p, decimales=4)
        _describir_rasgo(r, reg.rasgos["rasgo"], nombre)
        r.tabla("Análisis de varianza", COLS_ANOVA,
                _tabla_anova(res, ["Entre padres", "Dentro de padres (entre medios hermanos)"]),
                nota=f"Coeficiente k (tamaño efectivo de familia, diseño desbalanceado) = {res['k']:.4f}", id="anova")
        _, ee_s = anova.delta_cm(res, lambda c: c[0])
        _, ee_w = anova.delta_cm(res, lambda c: c[1])
        r.tabla("Componentes de varianza", COLS_COMPONENTES, [
            {"componente": "σ²padre (σ²s)", "estimacion": s2s, "ee": ee_s, "porcentaje": 100 * s2s / s2p,
             "significado": "¼ σ²A (covarianza entre medios hermanos)"},
            {"componente": "σ²dentro (σ²w)", "estimacion": s2w, "ee": ee_w, "porcentaje": 100 * s2w / s2p,
             "significado": "¾ σ²A + σ²D + σ²E"},
            {"componente": "σ²fenotípica (σ²P)", "estimacion": s2p, "ee": None, "porcentaje": 100.0,
             "significado": "σ²s + σ²w"},
            {"componente": "σ²aditiva (σ²A)", "estimacion": 4 * s2s, "ee": 4 * ee_s, "porcentaje": 100 * 4 * s2s / s2p,
             "significado": "4 σ²s"},
        ], id="componentes")
        r.tabla("Estructura de los datos", COLS_ESTRUCTURA, [_estructura("Padres", sc, len(sl))])
        r.grafico("componentes", "Partición de la varianza fenotípica",
                  partes=[{"etiqueta": "Padre (¼ σ²A)", "valor": max(s2s, 0)},
                          {"etiqueta": "Dentro de padres", "valor": max(s2w, 0)}])
        r.formula("Componente de padre", "σ²s = (CMentre − CMdentro) / k")
        r.formula("Heredabilidad", "h² = 4 σ²s / (σ²s + σ²w)")
        r.formula("Error estándar (Swiger et al., 1964)",
                  "EE(t) = √[ 2(N−1)(1−t)²(1+(k−1)t)² / (k²(N−s)(s−1)) ];  EE(h²) = 4·EE(t)")
        f_entre = anova.pruebas_f(res)[0]
        if f_entre[1] == f_entre[1] and f_entre[1] > 0.05:
            r.nota(f"La diferencia entre padres no es significativa (F = {f_entre[0]:.3f}, p = {f_entre[1]:.3f}): "
                   "la heredabilidad estimada no difiere de cero con estos datos.", "advertencia")
        params = {"h2": h2, "ee_h2": ee_h2, "var_p": s2p, "var_a": 4 * s2s, "var_padre": s2s, "var_e": s2w}
    else:
        ctx.progreso(0.3, "ANOVA anidado: padres / animales / registros…")
        ac, al, conflictos = _anidar(padres, animales, "padre", "animal")
        if conflictos:
            raise ErrorDatos(f"{len(conflictos)} animales aparecen con más de un padre (p. ej. "
                             f"{', '.join(map(str, conflictos[:5]))}). Corrija la columna de padre.")
        res = anova.anova_anidado(sc, ac, y)
        c = {k: float(v[0, 0]) for k, v in res["componentes"].items()}
        s2s, s2a, s2w = c["grupo"], c["subgrupo"], c["dentro"]
        s2p = s2s + s2a + s2w
        h2, ee_h2 = anova.delta_cm(res, lambda v: 4 * v[0] / (v[0] + v[1] + v[2]))
        rep, ee_r = anova.delta_cm(res, lambda v: (v[0] + v[1]) / (v[0] + v[1] + v[2]))
        n_h2 = nivel_h2(h2)
        r.indicador("Heredabilidad (h²)", h2, ee_h2, _ic(h2, ee_h2, conf), nivel=n_h2[0] if n_h2 else None,
                    descripcion=texto_h2(h2), destacado=True)
        n_r = nivel_r(rep)
        r.indicador("Repetibilidad (r)", rep, ee_r, _ic(rep, ee_r, conf), nivel=n_r[0] if n_r else None,
                    descripcion="Calculada a la vez porque hay registros repetidos por animal.")
        r.indicador("Varianza aditiva (σ²A)", 4 * s2s, decimales=4, descripcion="σ²A = 4 σ²padre")
        r.indicador("Varianza fenotípica (σ²P)", s2p, decimales=4)
        _describir_rasgo(r, reg.rasgos["rasgo"], nombre)
        r.tabla("Análisis de varianza (anidado)", COLS_ANOVA,
                _tabla_anova(res, ["Entre padres", "Animales dentro de padres", "Registros dentro de animales"]),
                nota=f"Coeficientes: k1 = {res['k1']:.4f}, k2 = {res['k2']:.4f}, k3 = {res['k3']:.4f}. "
                     "La F de padres usa un denominador sintético (Satterthwaite).", id="anova")
        ees = [anova.delta_cm(res, (lambda i: (lambda v: v[i]))(i))[1] for i in range(3)]
        r.tabla("Componentes de varianza", COLS_COMPONENTES, [
            {"componente": "σ²padre (σ²s)", "estimacion": s2s, "ee": ees[0], "porcentaje": 100 * s2s / s2p,
             "significado": "¼ σ²A"},
            {"componente": "σ²animal/padre", "estimacion": s2a, "ee": ees[1], "porcentaje": 100 * s2a / s2p,
             "significado": "¾ σ²A + σ²D + σ²Ep (ambiente permanente)"},
            {"componente": "σ²residual (σ²e)", "estimacion": s2w, "ee": ees[2], "porcentaje": 100 * s2w / s2p,
             "significado": "σ²Et (ambiente temporal entre registros)"},
            {"componente": "σ²fenotípica (σ²P)", "estimacion": s2p, "ee": None, "porcentaje": 100.0,
             "significado": "suma de los tres"},
        ], id="componentes")
        r.tabla("Estructura de los datos", COLS_ESTRUCTURA,
                [_estructura("Padres", sc, len(sl)), _estructura("Animales", ac, len(al))])
        r.grafico("componentes", "Partición de la varianza fenotípica",
                  partes=[{"etiqueta": "Padre (¼ σ²A)", "valor": max(s2s, 0)},
                          {"etiqueta": "Animal dentro de padre", "valor": max(s2a, 0)},
                          {"etiqueta": "Residual", "valor": max(s2w, 0)}])
        r.formula("Heredabilidad", "h² = 4 σ²s / (σ²s + σ²a/s + σ²e)")
        r.formula("Repetibilidad", "r = (σ²s + σ²a/s) / (σ²s + σ²a/s + σ²e)")
        r.formula("Errores estándar", "Método delta con Var(CM) = 2·CM² / (gl + 2)")
        params = {"h2": h2, "ee_h2": ee_h2, "r": rep, "ee_r": ee_r, "var_p": s2p, "var_a": 4 * s2s,
                  "var_padre": s2s, "var_pe": s2a, "var_e": s2w}
    r.grafico("medias_grupo", f"Promedio de la progenie por padre · {nombre}",
              grupos=_familias(sc, sl, reg.rasgos["rasgo"]), etiqueta_grupo="Padre",
              media_general=float(np.mean(reg.rasgos["rasgo"])))
    _avisos_tamano(r, len(sl), reg.n)
    if h2 == h2 and (h2 < 0 or h2 > 1):
        r.nota("La heredabilidad estimada está fuera del rango [0, 1]. Con ANOVA esto ocurre cuando hay pocos "
               "padres, familias pequeñas o efectos no modelados; pruebe REML y revise los datos.", "advertencia")
    for k, v in params.items():
        r.parametro(k, v)
    r.parametro("rasgo", nombre)
    return r.json()


def _avisos_tamano(r, padres, n):
    if padres < 20:
        r.nota(f"Solo hay {padres} padres: la estimación es poco precisa. Se recomiendan al menos 20–30 padres "
               "con varias crías cada uno.", "advertencia")


def _h2_hermanos_completos(payload, ctx):
    op = _opciones(payload)
    conf = _confianza(op)
    reg = _registros(payload, ["rasgo"], ["padre", "madre"], ["animal"])
    nombre = reg.nombres["rasgo"]
    y, nota_aj = ajustar_por_fijos(reg.rasgos["rasgo"], reg)
    padres, madres = reg.ids["padre"], reg.ids["madre"]
    r = Resultado("heredabilidad", "hermanos_completos", "Heredabilidad por hermanos completos (padre / madre)",
                  "")
    r.nota(reg.nota_descartados(), "advertencia")
    r.nota(nota_aj)
    animales = _ids_completos(reg.ids.get("animal"))
    if animales is not None and _repetidos(animales)[0] > 0:
        # promediar registros por animal
        cod, niv = codificar(animales)
        n = np.bincount(cod)
        y = np.bincount(cod, weights=y) / n
        primero = {}
        for i, c in enumerate(cod):
            primero.setdefault(c, i)
        padres = [padres[primero[c]] for c in range(len(niv))]
        madres = [madres[primero[c]] for c in range(len(niv))]
        r.nota("Hay registros repetidos por animal: se analizó el promedio de cada animal.")
    sc, sl = codificar(padres)
    dc, dl, _ = _anidar(padres, madres, "padre", "madre")
    r.d["subtitulo"] = f"{nombre} · {len(y)} individuos · {len(sl)} padres · {len(dl)} familias de hermanos completos"
    ctx.progreso(0.4, "ANOVA anidado padre / madre…")
    res = anova.anova_anidado(sc, dc, y)
    c = {k: float(v[0, 0]) for k, v in res["componentes"].items()}
    s2s, s2d, s2w = c["grupo"], c["subgrupo"], c["dentro"]
    s2p = s2s + s2d + s2w
    h2s, ee_s = anova.delta_cm(res, lambda v: 4 * v[0] / v.sum())
    h2d, ee_d = anova.delta_cm(res, lambda v: 4 * v[1] / v.sum())
    h2sd, ee_sd = anova.delta_cm(res, lambda v: 2 * (v[0] + v[1]) / v.sum())
    for et, val, ee, desc, dest in [
        ("h² componente de padre (h²S)", h2s, ee_s, "Estimador más confiable: libre de efectos maternos y de dominancia.", True),
        ("h² componente de madre (h²D)", h2d, ee_d, "Incluye efectos maternos y de dominancia: tiende a sobrestimar.", False),
        ("h² padre + madre (h²S+D)", h2sd, ee_sd, "Promedio de ambos componentes (más preciso, posible sesgo).", False),
    ]:
        nv = nivel_h2(val)
        r.indicador(et, val, ee, _ic(val, ee, conf), nivel=nv[0] if nv else None, descripcion=desc, destacado=dest)
    r.indicador("Varianza fenotípica (σ²P)", s2p, decimales=4)
    r.tabla("Análisis de varianza (anidado)", COLS_ANOVA,
            _tabla_anova(res, ["Entre padres", "Madres dentro de padres", "Progenie dentro de madres"]),
            nota=f"k1 = {res['k1']:.4f}, k2 = {res['k2']:.4f}, k3 = {res['k3']:.4f}", id="anova")
    ees = [anova.delta_cm(res, (lambda i: (lambda v: v[i]))(i))[1] for i in range(3)]
    r.tabla("Componentes de varianza", COLS_COMPONENTES, [
        {"componente": "σ²padre (σ²S)", "estimacion": s2s, "ee": ees[0], "porcentaje": 100 * s2s / s2p,
         "significado": "¼ σ²A"},
        {"componente": "σ²madre/padre (σ²D)", "estimacion": s2d, "ee": ees[1], "porcentaje": 100 * s2d / s2p,
         "significado": "¼ σ²A + ¼ σ²D + σ²Ec (efecto materno)"},
        {"componente": "σ²dentro (σ²W)", "estimacion": s2w, "ee": ees[2], "porcentaje": 100 * s2w / s2p,
         "significado": "½ σ²A + ¾ σ²D + σ²Ew"},
        {"componente": "σ²fenotípica (σ²P)", "estimacion": s2p, "ee": None, "porcentaje": 100.0, "significado": ""},
    ], id="componentes")
    r.tabla("Estructura de los datos", COLS_ESTRUCTURA,
            [_estructura("Padres", sc, len(sl)), _estructura("Familias (padre × madre)", dc, len(dl))])
    r.grafico("componentes", "Partición de la varianza fenotípica",
              partes=[{"etiqueta": "Padre", "valor": max(s2s, 0)}, {"etiqueta": "Madre dentro de padre", "valor": max(s2d, 0)},
                      {"etiqueta": "Dentro de familia", "valor": max(s2w, 0)}])
    if s2d > 1.5 * s2s and s2s > 0:
        r.nota("σ²madre es bastante mayor que σ²padre: sugiere efectos maternos o de dominancia; use h²S.", "advertencia")
    r.formula("h² de padre", "h²S = 4 σ²S / σ²P")
    r.formula("h² de madre", "h²D = 4 σ²D / σ²P")
    r.formula("h² combinada", "h²S+D = 2 (σ²S + σ²D) / σ²P")
    _avisos_tamano(r, len(sl), len(y))
    for k, v in {"h2": h2s, "ee_h2": ee_s, "h2_madre": h2d, "h2_combinada": h2sd, "var_p": s2p}.items():
        r.parametro(k, v)
    return r.json()


def _h2_regresion(payload, ctx):
    op = _opciones(payload)
    conf = _confianza(op)
    tipo = op.get("tipo_regresion") or "padre"
    roles = payload.get("roles") or {}
    r = Resultado("heredabilidad", "regresion", "Heredabilidad por regresión progenie–progenitor", "")
    if roles.get("valor_progenitor") and roles.get("valor_progenie"):
        reg = _registros(payload, ["valor_progenitor", "valor_progenie"], [], con_fijos=False)
        x, y = reg.rasgos["valor_progenitor"], reg.rasgos["valor_progenie"]
        grupos = None
        uno = tipo != "promedio"
        etiqueta_x = reg.nombres["valor_progenitor"]
        nombre = reg.nombres["valor_progenie"]
        r.nota(reg.nota_descartados(), "advertencia")
    else:
        reg = _registros(payload, ["rasgo"], ["animal"], ["padre", "madre"], con_fijos=False)
        nombre = reg.nombres["rasgo"]
        r.nota(reg.nota_descartados(), "advertencia")
        cod, niv = codificar(reg.ids["animal"])
        n = np.bincount(cod)
        fen = dict(zip(niv, np.bincount(cod, weights=reg.rasgos["rasgo"]) / n))
        if np.any(n > 1):
            r.nota("Hay registros repetidos: se usó el promedio de cada animal como su fenotipo.")
        # padres de cada animal: de la misma tabla o del pedigrí aparte
        padre_de, madre_de = {}, {}
        pa = reg.ids.get("padre") or [None] * reg.n
        ma = reg.ids.get("madre") or [None] * reg.n
        for a, p, m in zip(reg.ids["animal"], pa, ma):
            padre_de.setdefault(a, p)
            madre_de.setdefault(a, m)
        ped = payload.get("pedigri")
        if ped:
            ids_p, pad_p, mad_p, _ = _leer_pedigri(ped)
            for a, p, m in zip(ids_p, pad_p, mad_p):
                if a is not None:
                    padre_de[a] = p
                    madre_de[a] = m
        xs, ys, gs = [], [], []
        for a, va in fen.items():
            p, m = padre_de.get(a), madre_de.get(a)
            vp = fen.get(p) if p is not None else None
            vm = fen.get(m) if m is not None else None
            if tipo == "padre" and vp is not None:
                xs.append(vp); ys.append(va); gs.append(p)
            elif tipo == "madre" and vm is not None:
                xs.append(vm); ys.append(va); gs.append(m)
            elif tipo == "promedio" and vp is not None and vm is not None:
                xs.append((vp + vm) / 2); ys.append(va); gs.append(f"{p}×{m}")
            elif tipo == "madre_intra_padre" and vm is not None and p is not None:
                xs.append(vm); ys.append(va); gs.append(p)
        if len(xs) < 3:
            raise ErrorDatos("No hay suficientes pares progenitor–progenie con fenotipo. La regresión requiere que "
                             "los progenitores también tengan registro del rasgo en la hoja (o en el pedigrí).")
        x, y = np.array(xs), np.array(ys)
        grupos = gs
        uno = tipo != "promedio"
        etiqueta_x = {"padre": "Fenotipo del padre", "madre": "Fenotipo de la madre",
                      "promedio": "Promedio de los padres", "madre_intra_padre": "Fenotipo de la madre"}[tipo]
    if tipo == "madre_intra_padre" and grupos is None:
        raise ErrorDatos("La regresión sobre la madre dentro de padre necesita las columnas animal, padre y madre.")
    n = len(x)
    if tipo == "madre_intra_padre":
        cod, niv = codificar(grupos)
        k = len(niv)
        cnt = np.bincount(cod, minlength=k)
        mx = np.bincount(cod, weights=x, minlength=k) / cnt
        my = np.bincount(cod, weights=y, minlength=k) / cnt
        dx, dy = x - mx[cod], y - my[cod]
        sxx, sxy = float(dx @ dx), float(dx @ dy)
        if sxx <= 0:
            raise ErrorDatos("No hay variación de las madres dentro de padres.")
        b = sxy / sxx
        gl = n - k - 1
        resid = dy - b * dx
        a = float(y.mean() - b * x.mean())
        r.nota(f"Regresión intra-padre: se eliminaron las diferencias entre {k} padres (gl = {gl}).")
    else:
        xm, ym = x.mean(), y.mean()
        sxx = float(((x - xm) ** 2).sum())
        sxy = float(((x - xm) * (y - ym)).sum())
        if sxx <= 0:
            raise ErrorDatos("Los valores del progenitor no varían.")
        b = sxy / sxx
        a = float(ym - b * xm)
        gl = n - 2
        resid = (y - ym) - b * (x - xm)
    if gl < 1:
        raise ErrorDatos("Muy pocos pares para la regresión.")
    s2 = float(resid @ resid) / gl
    ee_b = math.sqrt(s2 / sxx)
    t_b = b / ee_b if ee_b > 0 else math.nan
    p_b = 2 * stats.t.sf(abs(t_b), gl) if t_b == t_b else math.nan
    corr = float(np.corrcoef(x, y)[0, 1]) if n > 2 else math.nan
    factor = 2.0 if uno else 1.0
    h2, ee_h2 = factor * b, factor * ee_b
    tcrit = stats.t.ppf(0.5 + conf / 2, gl)
    nv = nivel_h2(h2)
    r.d["subtitulo"] = f"{nombre} · {n} pares progenitor–progenie"
    r.indicador("Heredabilidad (h²)", h2, ee_h2, (h2 - tcrit * ee_h2, h2 + tcrit * ee_h2),
                nivel=nv[0] if nv else None, descripcion=texto_h2(h2), destacado=True)
    r.indicador("Coeficiente de regresión (b)", b, ee_b, (b - tcrit * ee_b, b + tcrit * ee_b),
                descripcion="h² = 2b con un progenitor; h² = b con el promedio de ambos padres")
    r.indicador("Pares analizados", n, decimales=0)
    r.indicador("Correlación progenitor–progenie", corr, decimales=3)
    r.tabla("Regresión lineal", [("parametro", "Parámetro"), ("valor", "Valor", "numero", 5),
                                 ("ee", "Error estándar", "numero", 5), ("t", "t", "numero", 3), ("p", "Valor p", "p")],
            [{"parametro": "Intercepto (a)", "valor": a, "ee": None, "t": None, "p": None},
             {"parametro": "Pendiente (b)", "valor": b, "ee": ee_b, "t": t_b, "p": p_b},
             {"parametro": "Varianza residual", "valor": s2, "ee": None, "t": None, "p": None},
             {"parametro": "Grados de libertad", "valor": gl, "ee": None, "t": None, "p": None}], id="regresion")
    puntos = list(zip(x.tolist(), y.tolist()))
    if len(puntos) > 3000:
        idx = np.random.default_rng(1).choice(len(puntos), 3000, replace=False)
        puntos = [puntos[i] for i in sorted(idx)]
    r.grafico("dispersion", "Progenie vs. progenitor", puntos=puntos, recta={"a": a, "b": b},
              eje_x=etiqueta_x, eje_y=f"Fenotipo de la progenie ({nombre})")
    r.formula("Pendiente", "b = Cov(progenitor, progenie) / Var(progenitor)")
    r.formula("Heredabilidad", "un progenitor: h² = 2b · promedio de padres: h² = b")
    if n < 30:
        r.nota("Menos de 30 pares: la estimación es muy imprecisa.", "advertencia")
    if tipo == "madre":
        r.nota("La regresión sobre la madre puede inflarse por efectos maternos; compare con la del padre.")
    r.parametro("h2", h2)
    r.parametro("ee_h2", ee_h2)
    return r.json()


# ---------------------------------------------------------------- REML
def _leer_pedigri(ped):
    from .datos import columna_ids
    tabla = ped.get("tabla") or {}
    roles = ped.get("roles") or {}
    vals = tabla.get("valores") or {}
    if not roles.get("id") or roles["id"] not in vals:
        raise ErrorDatos("En el pedigrí aparte falta la columna de ID.")
    ids = columna_ids(vals[roles["id"]])
    n = len(ids)
    pad = columna_ids(vals[roles["padre"]]) if roles.get("padre") in vals else [None] * n
    mad = columna_ids(vals[roles["madre"]]) if roles.get("madre") in vals else [None] * n
    sexo = vals.get(roles.get("sexo")) if roles.get("sexo") else None
    return ids, pad, mad, sexo


def _pedigri_de_registros(reg, payload, ctx):
    """Arma el pedigrí con (animal, padre, madre) de la hoja de datos + pedigrí aparte opcional."""
    ids = list(reg.ids["animal"])
    pad = list(reg.ids.get("padre") or [None] * reg.n)
    mad = list(reg.ids.get("madre") or [None] * reg.n)
    ped = payload.get("pedigri")
    if ped:
        ids_p, pad_p, mad_p, _ = _leer_pedigri(ped)
        # el pedigrí aparte manda: va primero (se usa la primera aparición)
        ids = ids_p + ids
        pad = pad_p + pad
        mad = mad_p + mad
    return Pedigri(ids, pad, mad, progreso=ctx.progreso, cancelado=ctx.cancelado)


def _inicio_anova_padre(y, sc, *anidados):
    """Valores iniciales razonables para REML a partir del ANOVA: un G por el padre y uno por
    cada efecto anidado en el padre (madre, animal), en ese orden."""
    Y = y.reshape(len(y), -1)
    var = np.atleast_2d(np.cov(Y, rowvar=False))
    try:
        if anidados:
            G, R = [None], None
            for codigos in anidados:
                res = anova.anova_anidado(sc, codigos, Y)
                if G[0] is None:
                    G[0] = res["componentes"]["grupo"]
                G.append(res["componentes"]["subgrupo"] / len(anidados))
                R = res["componentes"]["dentro"]
        else:
            res = anova.anova_una_via(sc, Y)
            G = [res["componentes"]["entre"]]
            R = res["componentes"]["dentro"]
    except ErrorDatos:
        return None
    G = [_reparar(g, 0.02 * np.diag(var)) for g in G]
    R = _reparar(R, 0.2 * np.diag(var))
    return {"R": R, "G": G}


def _reparar(S, minimos):
    S = np.atleast_2d(np.array(S, dtype=float))
    d = np.maximum(np.diag(S), minimos)
    out = S.copy()
    np.fill_diagonal(out, d)
    # acotar correlaciones a |r| < 0.95
    t = S.shape[0]
    for i in range(t):
        for j in range(t):
            if i != j:
                lim = 0.95 * math.sqrt(d[i] * d[j])
                out[i, j] = max(-lim, min(lim, S[i, j]))
    return out


def _modelo_padre(payload, ctx, rasgos):
    """Arma el modelo de padre: padre + (madre) + (animal si hay repetidos)."""
    reg = _registros(payload, rasgos, ["padre"], ["animal", "madre"])
    X, nombres_x, notas_x = matriz_fija(reg)
    Y = reg.matriz_y(rasgos)
    sc, sl = codificar(reg.ids["padre"])
    aleatorios = [reml.Aleatorio("padre", sc, len(sl), etiquetas=sl)]
    etiquetas = ["padre"]
    anidados = []
    if reg.ids.get("madre") is not None:
        # una cría sin madre registrada cuenta como hija de una madre distinta (no se agrupan)
        dc, dl, _ = _anidar(reg.ids["padre"], _ids_completos(reg.ids["madre"]), "padre", "madre")
        aleatorios.append(reml.Aleatorio("madre", dc, len(dl), etiquetas=dl))
        etiquetas.append("madre")
        anidados.append(dc)
    animales = _ids_completos(reg.ids.get("animal"))
    if animales is not None and _repetidos(animales)[0] > 0:
        ac, al, conflictos = _anidar(reg.ids["padre"], animales, "padre", "animal")
        if conflictos:
            raise ErrorDatos(f"{len(conflictos)} animales aparecen con más de un padre (p. ej. "
                             f"{', '.join(map(str, conflictos[:5]))}).")
        aleatorios.append(reml.Aleatorio("animal", ac, len(al), etiquetas=[x.split("␟", 1)[1] for x in al]))
        etiquetas.append("pe")
        anidados.append(ac)
    inicio = _inicio_anova_padre(Y, sc, *anidados)
    return reg, X, nombres_x, notas_x, Y, aleatorios, etiquetas, inicio, (sc, sl)


def _modelo_animal(payload, ctx, rasgos):
    reg = _registros(payload, rasgos, ["animal"], ["padre", "madre"])
    X, nombres_x, notas_x = matriz_fija(reg)
    Y = reg.matriz_y(rasgos)
    ctx.progreso(None, "Armando el pedigrí y la matriz A⁻¹…")
    ped = _pedigri_de_registros(reg, payload, ctx)
    Ainv = ped.a_inversa()
    codigos = np.array([ped.codigo[a] - 1 for a in reg.ids["animal"]])
    aleatorios = [reml.Aleatorio("animal", codigos, ped.n, Ainv=Ainv, etiquetas=ped.ids, logdet_a=ped.log_det_a())]
    etiquetas = ["animal"]
    if _repetidos(reg.ids["animal"])[0] > 0:
        pc, pl = codificar(reg.ids["animal"])
        aleatorios.append(reml.Aleatorio("pe", pc, len(pl), etiquetas=pl))
        etiquetas.append("pe")
    var = np.atleast_2d(np.cov(Y, rowvar=False))
    if "pe" in etiquetas:
        inicio = {"R": var * 0.45, "G": [var * 0.3, var * 0.25]}
    else:
        inicio = {"R": var * 0.65, "G": [var * 0.35]}
    for g in inicio["G"] + [inicio["R"]]:
        np.fill_diagonal(g, np.diag(g))
        g[~np.eye(len(g), dtype=bool)] *= 0.3
    return reg, X, nombres_x, notas_x, Y, aleatorios, etiquetas, inicio, ped


def _ajustar(Y, X, aleatorios, inicio, ctx):
    modelo = reml.ModeloMixto(Y, X, aleatorios)
    reml.verificar_tamano(modelo)
    ctx.progreso(None, "REML: buscando el máximo de la verosimilitud…")
    res = reml.ajustar(modelo, inicio=inicio, progreso=ctx.progreso, cancelado=ctx.cancelado)
    return modelo, res


def _funcion_componente(t, K, k, i=0, j=0):
    """Función del vector de parámetros que devuelve un elemento de G_k (k=-1 → R)."""
    def f(v):
        R, G = reml.matrices_de_vector(v, t, K)
        S = R if k < 0 else G[k]
        return S[i, j]
    return f


def _fenotipica(v, t, K, i=0, j=0):
    R, G = reml.matrices_de_vector(v, t, K)
    return R[i, j] + sum(g[i, j] for g in G)


def _info_modelo(r, modelo, res, nombres_x, notas_x):
    r.tabla("Información del ajuste", [("dato", "Dato"), ("valor", "Valor")], [
        {"dato": "−2 log L restringida", "valor": f"{res.m2logl:.4f}"},
        {"dato": "Registros", "valor": f"{modelo.n}"},
        {"dato": "Efectos fijos (columnas)", "valor": f"{modelo.p}"},
        {"dato": "Efectos aleatorios", "valor": ", ".join(f"{a.nombre} ({a.q} niveles)" for a in modelo.aleatorios)},
        {"dato": "Evaluaciones de la verosimilitud", "valor": f"{res.evaluaciones}"},
        {"dato": "Convergencia", "valor": "Sí" if res.convergio else "Revisar (se alcanzó el límite de iteraciones)"},
        {"dato": "Tiempo", "valor": f"{res.segundos:.1f} s"},
    ], id="ajuste")
    for n in notas_x:
        r.nota(n, "advertencia")
    if res.cov_param is None:
        r.nota("No se pudieron calcular algunos errores estándar: algún componente está en el límite del "
               "espacio paramétrico (≈ 0) o la superficie de verosimilitud es muy plana.", "advertencia")
    if any(res.limite):
        r.nota("Algún componente de varianza se estimó en el límite (≈ 0).", "advertencia")


def _fijos_tabla(r, sols, nombres_x, j=0):
    filas = [{"efecto": nombres_x[i], "solucion": float(sols["fijos"][i, j])} for i in range(min(len(nombres_x), 300))]
    r.tabla("Soluciones de efectos fijos (BLUE)", [("efecto", "Efecto"), ("solucion", "Solución", "numero", 4)], filas,
            nota="Con codificación de referencia: cada nivel es la diferencia respecto del primer nivel del factor.",
            id="fijos")


def _lrt(Y, X, aleatorios, quitar, res_completo, ctx):
    """Prueba de razón de verosimilitud para la varianza del efecto `quitar` (mezcla ½χ²₀ + ½χ²₁)."""
    resto = [a for k, a in enumerate(aleatorios) if k != quitar]
    try:
        if resto:
            m0 = reml.ModeloMixto(Y, X, resto)
            r0 = reml.ajustar(m0, progreso=None, cancelado=ctx.cancelado, max_eval=2000)
            m2 = r0.m2logl
        else:
            return None
    except Exception:
        return None
    lrt = max(m2 - res_completo.m2logl, 0.0)
    return lrt, 0.5 * stats.chi2.sf(lrt, 1)


def _h2_reml_padre(payload, ctx):
    op = _opciones(payload)
    conf = _confianza(op)
    reg, X, nombres_x, notas_x, Y, aleatorios, etiquetas, inicio, (sc, sl) = _modelo_padre(payload, ctx, ["rasgo"])
    nombre = reg.nombres["rasgo"]
    modelo, res = _ajustar(Y, X, aleatorios, inicio, ctx)
    K = len(aleatorios)
    v = reml.vector_parametros(res)
    cov = res.cov_param
    comp = {et: float(res.G[k][0, 0]) for k, et in enumerate(etiquetas)}
    s2e = float(res.R[0, 0])
    s2p = s2e + sum(comp.values())
    k_padre = etiquetas.index("padre")
    h2, ee_h2 = reml.delta(lambda w: 4 * reml.matrices_de_vector(w, 1, K)[1][k_padre][0, 0] / _fenotipica(w, 1, K), v, cov)
    r = Resultado("heredabilidad", "reml_padre", "Heredabilidad por REML · modelo de padre",
                  f"{nombre} · {reg.n} registros · {len(sl)} padres")
    r.nota(reg.nota_descartados(), "advertencia")
    nv = nivel_h2(h2)
    r.indicador("Heredabilidad (h²)", h2, ee_h2, _ic(h2, ee_h2, conf), nivel=nv[0] if nv else None,
                descripcion=texto_h2(h2), destacado=True)
    params = {"h2": h2, "ee_h2": ee_h2, "var_p": s2p, "var_a": 4 * comp["padre"], "var_padre": comp["padre"],
              "var_e": s2e}
    if "pe" in etiquetas:
        k_pe = etiquetas.index("pe")
        rep, ee_r = reml.delta(lambda w: (reml.matrices_de_vector(w, 1, K)[1][k_padre][0, 0]
                                          + reml.matrices_de_vector(w, 1, K)[1][k_pe][0, 0]) / _fenotipica(w, 1, K), v, cov)
        nr = nivel_r(rep)
        r.indicador("Repetibilidad (r)", rep, ee_r, _ic(rep, ee_r, conf), nivel=nr[0] if nr else None,
                    descripcion="Incluida porque hay registros repetidos por animal.")
        params.update({"r": rep, "ee_r": ee_r, "var_pe": comp["pe"]})
    r.indicador("Varianza aditiva (σ²A)", 4 * comp["padre"], decimales=4, descripcion="σ²A = 4 σ²padre")
    r.indicador("Varianza fenotípica (σ²P)", s2p, decimales=4)
    significado = {"padre": "¼ σ²A", "madre": "¼ σ²A + ¼ σ²D + σ²Ec", "pe": "¾ σ²A + σ²D + σ²Ep"}
    nombres_comp = {"padre": "σ²padre (σ²s)", "madre": "σ²madre/padre", "pe": "σ²animal/padre"}
    filas = []
    for k, et in enumerate(etiquetas):
        val, ee = reml.delta(_funcion_componente(1, K, k), v, cov)
        filas.append({"componente": nombres_comp[et], "estimacion": val, "ee": ee, "porcentaje": 100 * val / s2p,
                      "significado": significado[et]})
    val, ee = reml.delta(_funcion_componente(1, K, -1), v, cov)
    filas.append({"componente": "σ²residual (σ²e)", "estimacion": val, "ee": ee, "porcentaje": 100 * val / s2p,
                  "significado": "σ²E (y lo no explicado por los demás)"})
    filas.append({"componente": "σ²fenotípica (σ²P)", "estimacion": s2p, "ee": None, "porcentaje": 100.0,
                  "significado": "suma de componentes"})
    r.tabla("Componentes de varianza (REML)", COLS_COMPONENTES, filas, id="componentes")
    r.grafico("componentes", "Partición de la varianza fenotípica",
              partes=[{"etiqueta": nombres_comp[et], "valor": max(comp[et], 0)} for et in etiquetas]
              + [{"etiqueta": "Residual", "valor": max(s2e, 0)}])
    _info_modelo(r, modelo, res, nombres_x, notas_x)
    ctx.progreso(None, "Calculando valores de cría de los padres…")
    sols = modelo.soluciones(*res.std, varianza_error=True)
    u = sols["aleatorios"][k_padre][:, 0]
    pev = sols["pev"].get(k_padre)
    filas = []
    n_hijos = np.bincount(sc, minlength=len(sl))
    for i, et in enumerate(sl):
        exact = math.sqrt(max(0.0, 1 - pev[i, 0] / comp["padre"])) if pev is not None and comp["padre"] > 0 else None
        filas.append({"padre": et, "progenie": int(n_hijos[i]), "solucion": float(u[i]), "vc": 2 * float(u[i]),
                      "exactitud": exact})
    filas.sort(key=lambda f: -f["vc"])
    for i, f in enumerate(filas, 1):
        f["rango"] = i
    r.tabla("Valores de cría estimados de los padres (BLUP)",
            [("rango", "Rango", "entero"), ("padre", "Padre"), ("progenie", "Registros de progenie", "entero"),
             ("solucion", "Efecto del padre (ŝ)", "numero", 4), ("vc", "Valor de cría (VC = 2ŝ)", "numero", 4),
             ("exactitud", "Exactitud", "numero", 3)], filas, id="vc_padres",
            nota="La habilidad de transmisión predicha (HTP) es ŝ; el valor de cría es 2ŝ. Exactitud = √(1 − PEV/σ²s).")
    r.grafico("barras", "Valor de cría de los padres (10 mejores y 10 peores)",
              barras=[{"etiqueta": f["padre"], "valor": f["vc"]} for f in (filas[:10] + filas[-10:] if len(filas) > 20 else filas)],
              eje="Valor de cría")
    r.columna_nueva(f"VC_padre_{nombre}", "padre", {f["padre"]: f["vc"] for f in filas},
                    "Valor de cría del padre (REML/BLUP, modelo de padre)")
    _fijos_tabla(r, sols, nombres_x)
    if op.get("lrt", True):
        ctx.progreso(None, "Prueba de razón de verosimilitud (sin efecto del padre)…")
        prueba = _lrt(Y, X, aleatorios, k_padre, res, ctx)
        if prueba:
            r.tabla("Prueba de razón de verosimilitud", [("hipotesis", "Hipótesis nula"), ("lrt", "LRT", "numero", 3),
                                                         ("p", "Valor p", "p")],
                    [{"hipotesis": "σ²padre = 0 (no hay variación genética)", "lrt": prueba[0], "p": prueba[1]}],
                    nota="Distribución de referencia: mezcla ½χ²₀ + ½χ²₁ (el parámetro está en el límite).")
    r.formula("Modelo", "y = Xb + Z_s s + " + ("Z_d d + " if "madre" in etiquetas else "")
              + ("Z_p pe + " if "pe" in etiquetas else "") + "e")
    r.formula("Heredabilidad", "h² = 4 σ²s / σ²P")
    r.formula("Errores estándar", "Inversa de la matriz de información observada y método delta")
    _avisos_tamano(r, len(sl), reg.n)
    for k2, val in params.items():
        r.parametro(k2, val)
    r.parametro("rasgo", nombre)
    return r.json()


def _h2_reml_animal(payload, ctx):
    op = _opciones(payload)
    conf = _confianza(op)
    reg, X, nombres_x, notas_x, Y, aleatorios, etiquetas, inicio, ped = _modelo_animal(payload, ctx, ["rasgo"])
    nombre = reg.nombres["rasgo"]
    modelo, res = _ajustar(Y, X, aleatorios, inicio, ctx)
    return _resultado_animal(payload, ctx, reg, X, nombres_x, notas_x, Y, aleatorios, etiquetas, ped, modelo, res,
                             nombre, conf, tipo="heredabilidad")


def _resultado_animal(payload, ctx, reg, X, nombres_x, notas_x, Y, aleatorios, etiquetas, ped, modelo, res,
                      nombre, conf, tipo):
    op = _opciones(payload)
    K = len(aleatorios)
    v = reml.vector_parametros(res)
    cov = res.cov_param
    s2a = float(res.G[0][0, 0])
    s2pe = float(res.G[1][0, 0]) if "pe" in etiquetas else 0.0
    s2e = float(res.R[0, 0])
    s2p = s2a + s2pe + s2e
    h2, ee_h2 = reml.delta(lambda w: reml.matrices_de_vector(w, 1, K)[1][0][0, 0] / _fenotipica(w, 1, K), v, cov)
    titulo = ("Heredabilidad por REML · modelo animal" if tipo == "heredabilidad"
              else "Repetibilidad por REML · modelo animal con ambiente permanente")
    r = Resultado(tipo, "reml_animal", titulo,
                  f"{nombre} · {reg.n} registros · {ped.n} animales en el pedigrí")
    r.nota(reg.nota_descartados(), "advertencia")
    nv = nivel_h2(h2)
    params = {"h2": h2, "ee_h2": ee_h2, "var_a": s2a, "var_e": s2e, "var_p": s2p}
    rep = ee_r = None
    if "pe" in etiquetas:
        rep, ee_r = reml.delta(lambda w: (reml.matrices_de_vector(w, 1, K)[1][0][0, 0]
                                          + reml.matrices_de_vector(w, 1, K)[1][1][0, 0]) / _fenotipica(w, 1, K), v, cov)
        params.update({"r": rep, "ee_r": ee_r, "var_pe": s2pe})
    if tipo == "heredabilidad":
        r.indicador("Heredabilidad (h²)", h2, ee_h2, _ic(h2, ee_h2, conf), nivel=nv[0] if nv else None,
                    descripcion=texto_h2(h2), destacado=True)
        if rep is not None:
            nr = nivel_r(rep)
            r.indicador("Repetibilidad (r)", rep, ee_r, _ic(rep, ee_r, conf), nivel=nr[0] if nr else None,
                        descripcion="Incluida porque hay registros repetidos por animal.")
    else:
        nr = nivel_r(rep)
        r.indicador("Repetibilidad (r)", rep, ee_r, _ic(rep, ee_r, conf), nivel=nr[0] if nr else None,
                    descripcion=texto_r(rep), destacado=True)
        r.indicador("Heredabilidad (h²)", h2, ee_h2, _ic(h2, ee_h2, conf), nivel=nv[0] if nv else None)
    r.indicador("Varianza aditiva (σ²A)", s2a, decimales=4)
    r.indicador("Varianza fenotípica (σ²P)", s2p, decimales=4)
    filas = []
    nombres_comp = {"animal": ("σ²A (aditiva)", "Varianza genética aditiva"),
                    "pe": ("σ²pe (ambiente permanente)", "Ambiente permanente + genética no aditiva")}
    for k, et in enumerate(etiquetas):
        val, ee = reml.delta(_funcion_componente(1, K, k), v, cov)
        filas.append({"componente": nombres_comp[et][0], "estimacion": val, "ee": ee,
                      "porcentaje": 100 * val / s2p, "significado": nombres_comp[et][1]})
    val, ee = reml.delta(_funcion_componente(1, K, -1), v, cov)
    filas.append({"componente": "σ²e (residual)", "estimacion": val, "ee": ee, "porcentaje": 100 * val / s2p,
                  "significado": "Ambiente temporal"})
    filas.append({"componente": "σ²P (fenotípica)", "estimacion": s2p, "ee": None, "porcentaje": 100.0,
                  "significado": ""})
    r.tabla("Componentes de varianza (REML)", COLS_COMPONENTES, filas, id="componentes")
    r.grafico("componentes", "Partición de la varianza fenotípica",
              partes=[{"etiqueta": "Aditiva", "valor": max(s2a, 0)}]
              + ([{"etiqueta": "Ambiente permanente", "valor": max(s2pe, 0)}] if "pe" in etiquetas else [])
              + [{"etiqueta": "Residual", "valor": max(s2e, 0)}])
    _info_modelo(r, modelo, res, nombres_x, notas_x)
    ctx.progreso(None, "Calculando valores de cría (BLUP) y exactitudes…")
    calcular_pev = ped.n <= 20000
    sols = modelo.soluciones(*res.std, varianza_error=calcular_pev)
    _tabla_vc(r, ped, reg, sols, 0, s2a, nombre, "animal")
    _fijos_tabla(r, sols, nombres_x)
    if not calcular_pev:
        r.nota("Pedigrí grande: no se calcularon exactitudes (requieren invertir la matriz del modelo).")
    if op.get("lrt", True) and tipo == "heredabilidad":
        ctx.progreso(None, "Prueba de razón de verosimilitud (sin efecto genético)…")
        prueba = _lrt(Y, X, aleatorios, 0, res, ctx)
        if prueba:
            r.tabla("Prueba de razón de verosimilitud", [("hipotesis", "Hipótesis nula"), ("lrt", "LRT", "numero", 3),
                                                         ("p", "Valor p", "p")],
                    [{"hipotesis": "σ²A = 0 (no hay variación genética aditiva)", "lrt": prueba[0], "p": prueba[1]}],
                    nota="Distribución de referencia: mezcla ½χ²₀ + ½χ²₁.")
    r.formula("Modelo animal", "y = Xb + Za + " + ("Wpe + " if "pe" in etiquetas else "") + "e;  var(a) = A σ²A")
    r.formula("Heredabilidad", "h² = σ²A / σ²P")
    if "pe" in etiquetas:
        r.formula("Repetibilidad", "r = (σ²A + σ²pe) / σ²P")
    for k2, val in params.items():
        r.parametro(k2, val)
    r.parametro("rasgo", nombre)
    return r.json()


def _tabla_vc(r, ped, reg, sols, k, s2a, nombre, clave, j=0, titulo=None):
    u = sols["aleatorios"][k][:, j]
    pev = sols["pev"].get(k)
    n_reg = defaultdict(int)
    suma = defaultdict(float)
    for a, yv in zip(reg.ids["animal"], reg.rasgos[list(reg.rasgos)[j]]):
        n_reg[a] += 1
        suma[a] += yv
    F = ped.F()
    filas = []
    for i, a in enumerate(ped.ids):
        pa, ma = ped.s[i + 1], ped.d[i + 1]
        ex = None
        if pev is not None and s2a > 0:
            ex = math.sqrt(max(0.0, 1 - pev[i, j] / s2a))
        filas.append({"animal": a, "padre": ped.ids[pa - 1] if pa else "", "madre": ped.ids[ma - 1] if ma else "",
                      "registros": n_reg.get(a, 0),
                      "promedio": suma[a] / n_reg[a] if n_reg.get(a) else None,
                      "vc": float(u[i]), "exactitud": ex, "f": float(F[i])})
    filas.sort(key=lambda f: -f["vc"])
    for i, f in enumerate(filas, 1):
        f["rango"] = i
    columnas = [("rango", "Rango", "entero"), ("animal", "Animal"), ("padre", "Padre"), ("madre", "Madre"),
                ("registros", "Registros", "entero"), ("promedio", "Promedio fenotípico", "numero", 3),
                ("vc", "Valor de cría (VC)", "numero", 4), ("exactitud", "Exactitud", "numero", 3),
                ("f", "Endogamia (F)", "numero", 4)]
    r.tabla(titulo or f"Valores de cría (BLUP) · {nombre}", columnas, filas[:LIMITE_FILAS], id=f"vc_{j}",
            nota="Ordenados por valor de cría. Exactitud = √(1 − PEV/σ²A). "
                 "Los animales sin registros reciben VC por la información de sus parientes.")
    vals = [f["vc"] for f in filas]
    r.grafico("histograma", f"Distribución de los valores de cría · {nombre}", **histograma(vals), eje="Valor de cría")
    r.columna_nueva(f"VC_{nombre}", clave, {f["animal"]: f["vc"] for f in filas},
                    "Valor de cría (BLUP, modelo animal)")
    if any(f["exactitud"] is not None for f in filas):
        r.columna_nueva(f"Exactitud_{nombre}", clave, {f["animal"]: f["exactitud"] for f in filas},
                        "Exactitud del valor de cría")
    return filas


# ================================================================ REPETIBILIDAD
def repetibilidad(payload, ctx):
    metodo = payload.get("metodo") or "anova"
    funciones = {"anova": _r_anova, "anova_anidado": _r_anidado, "reml": _r_reml, "reml_animal": _r_reml_animal}
    if metodo not in funciones:
        raise ErrorDatos(f"Método de repetibilidad desconocido: {metodo}")
    return funciones[metodo](payload, ctx)


def _ppmp(r, reg, rep, ajustado, nombre, clave="animal", desvios=None):
    """Producción más probable (PMP / MPPA) y tabla de precisión por número de registros."""
    animales = reg.ids["animal"]
    cod, niv = codificar(animales)
    n = np.bincount(cod)
    medias = np.bincount(cod, weights=ajustado) / n
    mu = float(np.mean(ajustado))
    if desvios is None:
        factor = n * rep / (1 + (n - 1) * rep)
        pmp = mu + factor * (medias - mu)
    else:
        factor = None
        pmp = mu + desvios
    filas = []
    for i, a in enumerate(niv):
        filas.append({"animal": a, "registros": int(n[i]), "promedio": float(medias[i]),
                      "factor": float(factor[i]) if factor is not None else None, "pmp": float(pmp[i])})
    filas.sort(key=lambda f: -f["pmp"])
    for i, f in enumerate(filas, 1):
        f["rango"] = i
    r.tabla(f"Producción más probable (PMP) · {nombre}",
            [("rango", "Rango", "entero"), ("animal", "Animal"), ("registros", "Registros", "entero"),
             ("promedio", "Promedio propio", "numero", 3), ("factor", "Factor n·r/[1+(n−1)r]", "numero", 3),
             ("pmp", "PMP", "numero", 3)], filas[:LIMITE_FILAS], id="pmp",
            nota="PMP = μ + [n·r / (1 + (n−1)·r)]·(ȳ − μ). Ordena a los animales por su producción futura esperada.")
    r.columna_nueva(f"PMP_{nombre}", clave, {f["animal"]: f["pmp"] for f in filas}, "Producción más probable")
    if rep == rep and 0 <= rep <= 1:
        tabla = []
        for k in range(1, 11):
            tabla.append({"n": k, "rep_promedio": k * rep / (1 + (k - 1) * rep),
                          "var_relativa": (1 + (k - 1) * rep) / k,
                          "ganancia": 100 * (math.sqrt(k / (1 + (k - 1) * rep)) - 1)})
        r.tabla("Beneficio de repetir mediciones", [
            ("n", "N.º de registros", "entero"), ("rep_promedio", "Repetibilidad del promedio", "numero", 3),
            ("var_relativa", "Varianza del promedio / σ²P", "numero", 3),
            ("ganancia", "Ganancia en precisión (%)", "numero", 1)], tabla, id="beneficio",
            nota="Ganancia en precisión de selección respecto de un solo registro: √(n / [1 + (n−1)r]) − 1.")
        r.grafico("lineas", "Precisión ganada al promediar n registros",
                  series=[{"nombre": "Ganancia (%)", "puntos": [[f["n"], f["ganancia"]] for f in tabla]}],
                  eje_x="Número de registros por animal", eje_y="Ganancia en precisión (%)")
    return filas


def _r_anova(payload, ctx):
    op = _opciones(payload)
    conf = _confianza(op)
    reg = _registros(payload, ["rasgo"], ["animal"])
    nombre = reg.nombres["rasgo"]
    y, nota_aj = ajustar_por_fijos(reg.rasgos["rasgo"], reg)
    ac, al = codificar(reg.ids["animal"])
    if _repetidos(reg.ids["animal"])[0] == 0:
        raise ErrorDatos("Ningún animal tiene más de un registro: la repetibilidad necesita mediciones repetidas.")
    ctx.progreso(0.4, "ANOVA de una vía entre animales…")
    res = anova.anova_una_via(ac, y)
    s2a = float(res["componentes"]["entre"][0, 0])
    s2w = float(res["componentes"]["dentro"][0, 0])
    s2p = s2a + s2w
    rep = s2a / s2p if s2p > 0 else math.nan
    ee = anova.ee_swiger(rep, res["N"], res["grupos"], res["k"])
    ic = anova.ic_intraclase(res, 0, conf)
    r = Resultado("repetibilidad", "anova", "Repetibilidad por análisis de varianza",
                  f"{nombre} · {reg.n} registros · {len(al)} animales")
    r.nota(reg.nota_descartados(), "advertencia")
    r.nota(nota_aj)
    nr = nivel_r(rep)
    r.indicador("Repetibilidad (r)", rep, ee, ic, nivel=nr[0] if nr else None, descripcion=texto_r(rep), destacado=True)
    r.indicador("σ²entre animales", s2a, decimales=4, descripcion="σ²G + σ²Ep (genética + ambiente permanente)")
    r.indicador("σ²dentro de animales", s2w, decimales=4, descripcion="σ²Et (ambiente temporal)")
    r.indicador("Registros por animal (k₀)", res["k"], decimales=2)
    _describir_rasgo(r, reg.rasgos["rasgo"], nombre)
    r.tabla("Análisis de varianza", COLS_ANOVA,
            _tabla_anova(res, ["Entre animales", "Dentro de animales (entre registros)"]),
            nota=f"k₀ = {res['k']:.4f}", id="anova")
    _, ee_a = anova.delta_cm(res, lambda c: c[0])
    _, ee_w = anova.delta_cm(res, lambda c: c[1])
    r.tabla("Componentes de varianza", COLS_COMPONENTES, [
        {"componente": "σ²entre animales", "estimacion": s2a, "ee": ee_a, "porcentaje": 100 * s2a / s2p,
         "significado": "σ²G + σ²Ep"},
        {"componente": "σ²dentro (residual)", "estimacion": s2w, "ee": ee_w, "porcentaje": 100 * s2w / s2p,
         "significado": "σ²Et"},
        {"componente": "σ²fenotípica (σ²P)", "estimacion": s2p, "ee": None, "porcentaje": 100.0, "significado": ""},
    ], id="componentes")
    r.grafico("componentes", "Partición de la varianza fenotípica",
              partes=[{"etiqueta": "Entre animales", "valor": max(s2a, 0)},
                      {"etiqueta": "Dentro de animales", "valor": max(s2w, 0)}])
    _ppmp(r, reg, rep, y, nombre)
    r.formula("Repetibilidad", "r = σ²entre / (σ²entre + σ²dentro),  σ²entre = (CMentre − CMdentro) / k₀")
    r.formula("Error estándar (Swiger et al., 1964)", "EE(r) = √[ 2(N−1)(1−r)²(1+(k−1)r)² / (k²(N−a)(a−1)) ]")
    r.formula("Producción más probable", "PMP = μ + [n r / (1 + (n−1) r)] (ȳ − μ)")
    r.parametro("r", rep)
    r.parametro("ee_r", ee)
    r.parametro("rasgo", nombre)
    return r.json()


def _r_anidado(payload, ctx):
    payload = dict(payload)
    resultado = _h2_medios_hermanos(payload, ctx)
    # reorganizar: la repetibilidad es la protagonista
    ind = resultado["indicadores"]
    rep = next((i for i in ind if i["etiqueta"].startswith("Repetibilidad")), None)
    if rep is None:
        raise ErrorDatos("Para el ANOVA anidado se necesitan registros repetidos por animal (columna «animal»).")
    rep["destacado"] = True
    rep["descripcion"] = texto_r(rep["valor"])
    for i in ind:
        if i["etiqueta"].startswith("Heredabilidad"):
            i["destacado"] = False
    resultado["indicadores"] = [rep] + [i for i in ind if i is not rep]
    resultado["tipo"] = "repetibilidad"
    resultado["metodo"] = "anova_anidado"
    resultado["titulo"] = "Repetibilidad y heredabilidad · ANOVA anidado (padre / animal / registro)"
    # PMP con la repetibilidad estimada
    reg = _registros(payload, ["rasgo"], ["animal"])
    y, _ = ajustar_por_fijos(reg.rasgos["rasgo"], reg)
    extra = Resultado("repetibilidad", "anova_anidado", "")
    _ppmp(extra, reg, rep["valor"], y, reg.nombres["rasgo"])
    resultado["tablas"] += extra.json()["tablas"]
    resultado["graficos"] += extra.json()["graficos"]
    resultado["columnas_nuevas"] += extra.json()["columnas_nuevas"]
    resultado["formulas"].append({"titulo": "Producción más probable", "expresion": "PMP = μ + [n r / (1 + (n−1) r)] (ȳ − μ)"})
    return resultado


def _r_reml(payload, ctx):
    """Modelo de repetibilidad: animal (iid) [+ padre]; r = (σ²s + σ²an)/σ²P."""
    op = _opciones(payload)
    conf = _confianza(op)
    roles = payload.get("roles") or {}
    # con padre, los registros sin padre se excluyen (no pueden ubicarse en el modelo anidado)
    reg = _registros(payload, ["rasgo"], ["animal", "padre"] if roles.get("padre") else ["animal"])
    nombre = reg.nombres["rasgo"]
    if _repetidos(reg.ids["animal"])[0] == 0:
        raise ErrorDatos("Ningún animal tiene más de un registro: la repetibilidad necesita mediciones repetidas.")
    X, nombres_x, notas_x = matriz_fija(reg)
    Y = reg.matriz_y(["rasgo"])
    aleatorios, etiquetas = [], []
    if roles.get("padre") and reg.ids.get("padre") is not None:
        sc, sl = codificar(reg.ids["padre"])
        ac, al, conflictos = _anidar(reg.ids["padre"], reg.ids["animal"], "padre", "animal")
        if conflictos:
            raise ErrorDatos(f"{len(conflictos)} animales aparecen con más de un padre.")
        aleatorios = [reml.Aleatorio("padre", sc, len(sl)), reml.Aleatorio("animal", ac, len(al))]
        etiquetas = ["padre", "animal"]
        inicio = _inicio_anova_padre(Y, sc, ac)
    else:
        ac, al = codificar(reg.ids["animal"])
        aleatorios = [reml.Aleatorio("animal", ac, len(al))]
        etiquetas = ["animal"]
        try:
            res_a = anova.anova_una_via(ac, Y)
            v = np.diag(np.atleast_2d(np.cov(Y, rowvar=False)))
            inicio = {"R": _reparar(res_a["componentes"]["dentro"], 0.2 * v),
                      "G": [_reparar(res_a["componentes"]["entre"], 0.02 * v)]}
        except ErrorDatos:
            inicio = None
    modelo, res = _ajustar(Y, X, aleatorios, inicio, ctx)
    K = len(aleatorios)
    v = reml.vector_parametros(res)
    cov = res.cov_param
    comp = {et: float(res.G[k][0, 0]) for k, et in enumerate(etiquetas)}
    s2e = float(res.R[0, 0])
    s2p = s2e + sum(comp.values())
    rep, ee_r = reml.delta(lambda w: sum(g[0, 0] for g in reml.matrices_de_vector(w, 1, K)[1]) / _fenotipica(w, 1, K),
                           v, cov)
    r = Resultado("repetibilidad", "reml", "Repetibilidad por REML",
                  f"{nombre} · {reg.n} registros · {len(set(reg.ids['animal']))} animales")
    r.nota(reg.nota_descartados(), "advertencia")
    nr = nivel_r(rep)
    r.indicador("Repetibilidad (r)", rep, ee_r, _ic(rep, ee_r, conf), nivel=nr[0] if nr else None,
                descripcion=texto_r(rep), destacado=True)
    if "padre" in etiquetas:
        h2, ee_h2 = reml.delta(lambda w: 4 * reml.matrices_de_vector(w, 1, K)[1][0][0, 0] / _fenotipica(w, 1, K), v, cov)
        nv = nivel_h2(h2)
        r.indicador("Heredabilidad (h²)", h2, ee_h2, _ic(h2, ee_h2, conf), nivel=nv[0] if nv else None,
                    descripcion="Del componente de padre (4σ²s/σ²P).")
        r.parametro("h2", h2)
        r.parametro("ee_h2", ee_h2)
    r.indicador("Varianza fenotípica (σ²P)", s2p, decimales=4)
    nombres = {"padre": ("σ²padre", "¼ σ²A"), "animal": ("σ²animal" + ("/padre" if "padre" in etiquetas else ""),
                                                        "σ²G + σ²Ep" if "padre" not in etiquetas else "¾ σ²A + σ²D + σ²Ep")}
    filas = []
    for k, et in enumerate(etiquetas):
        val, ee = reml.delta(_funcion_componente(1, K, k), v, cov)
        filas.append({"componente": nombres[et][0], "estimacion": val, "ee": ee, "porcentaje": 100 * val / s2p,
                      "significado": nombres[et][1]})
    val, ee = reml.delta(_funcion_componente(1, K, -1), v, cov)
    filas.append({"componente": "σ²residual", "estimacion": val, "ee": ee, "porcentaje": 100 * val / s2p,
                  "significado": "σ²Et (ambiente temporal)"})
    filas.append({"componente": "σ²fenotípica (σ²P)", "estimacion": s2p, "ee": None, "porcentaje": 100.0,
                  "significado": ""})
    r.tabla("Componentes de varianza (REML)", COLS_COMPONENTES, filas, id="componentes")
    r.grafico("componentes", "Partición de la varianza fenotípica",
              partes=[{"etiqueta": nombres[et][0], "valor": max(comp[et], 0)} for et in etiquetas]
              + [{"etiqueta": "Residual", "valor": max(s2e, 0)}])
    _info_modelo(r, modelo, res, nombres_x, notas_x)
    sols = modelo.soluciones(*res.std)
    # PMP = μ + (ŝ + â): predicción BLUP de la capacidad productiva del animal
    cod, niv = codificar(reg.ids["animal"])
    desv = np.zeros(len(niv))
    k_an = etiquetas.index("animal")
    ua = sols["aleatorios"][k_an][:, 0]
    if "padre" in etiquetas:
        us = sols["aleatorios"][0][:, 0]
        animal_a_padre = {}
        for i in range(reg.n):
            animal_a_padre[cod[i]] = (sc[i], ac[i])
        for c in range(len(niv)):
            s_i, a_i = animal_a_padre[c]
            desv[c] = us[s_i] + ua[a_i]
    else:
        primero = {}
        for i in range(reg.n):
            primero.setdefault(cod[i], ac[i])
        for c in range(len(niv)):
            desv[c] = ua[primero[c]]
    y_aj = Y[:, 0]
    _ppmp(r, reg, rep, y_aj, nombre, desvios=desv)
    r.formula("Repetibilidad", "r = (σ²G + σ²Ep) / σ²P")
    r.formula("PMP (BLUP)", "PMP = μ + predicción BLUP del efecto del animal")
    r.parametro("r", rep)
    r.parametro("ee_r", ee_r)
    r.parametro("rasgo", nombre)
    return r.json()


def _r_reml_animal(payload, ctx):
    op = _opciones(payload)
    conf = _confianza(op)
    reg, X, nombres_x, notas_x, Y, aleatorios, etiquetas, inicio, ped = _modelo_animal(payload, ctx, ["rasgo"])
    if "pe" not in etiquetas:
        raise ErrorDatos("Ningún animal tiene más de un registro: la repetibilidad necesita mediciones repetidas.")
    modelo, res = _ajustar(Y, X, aleatorios, inicio, ctx)
    return _resultado_animal(payload, ctx, reg, X, nombres_x, notas_x, Y, aleatorios, etiquetas, ped, modelo, res,
                             reg.nombres["rasgo"], conf, tipo="repetibilidad")


# ================================================================ CORRELACIÓN GENÉTICA
def correlacion(payload, ctx):
    metodo = payload.get("metodo") or "ancova"
    op = _opciones(payload)
    rasgos = [x for x in (op.get("rasgos") or []) if x]
    roles = dict(payload.get("roles") or {})
    if not rasgos:
        rasgos = [roles.get("rasgo_x"), roles.get("rasgo_y")]
        rasgos = [x for x in rasgos if x]
    rasgos = list(dict.fromkeys(rasgos))
    if len(rasgos) < 2:
        raise ErrorDatos("Elija al menos dos rasgos distintos.")
    for i, nombre in enumerate(rasgos):
        roles[f"rasgo_{i}"] = nombre
    payload = dict(payload, roles=roles)
    claves = [f"rasgo_{i}" for i in range(len(rasgos))]
    funciones = {"ancova": _rg_ancova, "reml_padre": _rg_reml, "reml_animal": _rg_reml, "cruzada": _rg_cruzada}
    if metodo not in funciones:
        raise ErrorDatos(f"Método de correlación desconocido: {metodo}")
    return funciones[metodo](payload, ctx, claves, rasgos, metodo)


def _motivo_sin_repetibilidad(animales):
    if animales is None:
        return ("Repetibilidad: asigne la columna «Animal» para estimarla; se necesitan varios registros "
                "(lactancias, esquilas, pesajes…) por animal.")
    return ("Repetibilidad: cada animal tiene un solo registro de estos rasgos, así que no se puede estimar "
            "(necesita mediciones repetidas del mismo animal).")


def _rep_una_via(animales, Y):
    """Repetibilidad de cada rasgo por ANOVA de una vía entre animales (EE de Swiger)."""
    t = Y.shape[1]
    rep, ee = np.full(t, np.nan), np.full(t, np.nan)
    ac, _ = codificar(animales)
    try:
        res = anova.anova_una_via(ac, Y)
    except ErrorDatos:
        return rep, ee
    for j in range(t):
        a, w = float(res["componentes"]["entre"][j, j]), float(res["componentes"]["dentro"][j, j])
        if a + w > 0:
            rep[j] = a / (a + w)
            ee[j] = anova.ee_swiger(rep[j], res["N"], res["grupos"], res["k"])
    return rep, ee


def _texto_ic(ic):
    if not ic or ic[0] is None or ic[0] != ic[0]:
        return "—"
    return f"{ic[0]:.3f} a {ic[1]:.3f}"


def _num(v):
    return None if v is None or v != v else float(v)


def _presentar_correlacion(r, nombres, d, conf, resumen):
    """Indicadores, tabla por rasgo (h², r), tabla por par (rG, rP, rE), selección indirecta y datos del
    tablero. Reemplaza la antigua «matriz de parámetros», que mezclaba h², rG y rP en una sola tabla.

    d: h2, ee_h2, rep, ee_rep, var_p (vectores de t) · rg, ee_rg, rp, re (matrices t × t) ·
       valores (registros usados de cada rasgo) · nota_rep."""
    t = len(nombres)
    h2, ee_h2, rep, ee_rep, var_p = (np.asarray(d[k], dtype=float) for k in ("h2", "ee_h2", "rep", "ee_rep", "var_p"))
    rg, ee_rg, rp, re = d["rg"], d["ee_rg"], d["rp"], d["re"]
    hay_rep = bool(np.any(~np.isnan(rep)))
    pct = int(round(100 * conf))

    # --- indicadores (resumen, exportación y «Copiar resumen»)
    for i in range(t):
        for j in range(i + 1, t):
            nv = nivel_rg(rg[i, j])
            et = "Correlación genética (rG)" if t == 2 else f"rG · {nombres[i]} × {nombres[j]}"
            r.indicador(et, rg[i, j], ee_rg[i, j], _ic(rg[i, j], ee_rg[i, j], conf), nivel=nv[0] if nv else None,
                        descripcion=texto_rg(rg[i, j]) if t == 2 else None, destacado=(i == 0 and j == 1))
    if t == 2:
        r.indicador("Correlación fenotípica (rP)", rp[0, 1])
        r.indicador("Correlación ambiental (rE)", re[0, 1])
    for j, nombre in enumerate(nombres):
        nh = nivel_h2(h2[j])
        r.indicador(f"h² · {nombre}", h2[j], ee_h2[j], _ic(h2[j], ee_h2[j], conf), nivel=nh[0] if nh else None)
        if rep[j] == rep[j]:
            nr = nivel_r(rep[j])
            r.indicador(f"Repetibilidad · {nombre}", rep[j], ee_rep[j], _ic(rep[j], ee_rep[j], conf),
                        nivel=nr[0] if nr else None)

    # --- parámetros de cada rasgo
    filas, rasgos = [], []
    for j, nombre in enumerate(nombres):
        y = np.asarray(d["valores"][j], dtype=float)
        media = float(np.mean(y))
        de = float(np.std(y, ddof=1)) if len(y) > 1 else math.nan
        ic_h, ic_r = _ic(h2[j], ee_h2[j], conf), _ic(rep[j], ee_rep[j], conf)
        nh, nr = nivel_h2(h2[j]), nivel_r(rep[j])
        filas.append({"rasgo": nombre, "registros": int(len(y)), "media": media, "de": de,
                      "h2": h2[j], "ee_h2": ee_h2[j], "ic_h2": _texto_ic(ic_h), "nivel_h2": nh[0] if nh else "—",
                      "r": rep[j], "ee_r": ee_rep[j], "ic_r": _texto_ic(ic_r), "nivel_r": nr[0] if nr else "—"})
        rasgos.append({"nombre": nombre, "n": int(len(y)), "media": media, "de": de, "var_p": _num(var_p[j]),
                       "h2": _num(h2[j]), "ee_h2": _num(ee_h2[j]), "ic_h2": list(ic_h) if ic_h else None,
                       "nivel_h2": nh[0] if nh else None,
                       "r": _num(rep[j]), "ee_r": _num(ee_rep[j]), "ic_r": list(ic_r) if ic_r else None,
                       "nivel_r": nr[0] if nr else None})
    cols = [("rasgo", "Rasgo"), ("registros", "Registros", "entero"), ("media", "Media", "numero", 3),
            ("de", "DE", "numero", 3), ("h2", "Heredabilidad (h²)", "numero", 3), ("ee_h2", "EE", "numero", 3),
            ("ic_h2", f"IC {pct} % de h²")]
    if hay_rep:
        cols += [("r", "Repetibilidad (r)", "numero", 3), ("ee_r", "EE", "numero", 3), ("ic_r", f"IC {pct} % de r")]
    r.tabla("Parámetros de cada rasgo", cols, filas, id="por_rasgo",
            nota=None if hay_rep else d.get("nota_rep"))

    # --- correlaciones por par de rasgos
    filas, pares = [], []
    for i in range(t):
        for j in range(i + 1, t):
            ic = _ic(rg[i, j], ee_rg[i, j], conf)
            distinta = None if not ic else ("Sí" if ic[0] > 0 or ic[1] < 0 else "No")
            nv = nivel_rg(rg[i, j])
            filas.append({"par": f"{nombres[i]} × {nombres[j]}", "rg": rg[i, j], "ee_rg": ee_rg[i, j],
                          "ic": _texto_ic(ic), "distinta": distinta or "—", "nivel": nv[0] if nv else "—",
                          "rp": rp[i, j], "re": re[i, j]})
            pares.append({"i": i, "j": j, "x": nombres[i], "y": nombres[j], "rg": _num(rg[i, j]),
                          "ee_rg": _num(ee_rg[i, j]), "ic_rg": list(ic) if ic else None, "distinta": distinta,
                          "nivel": nv[0] if nv else None, "rp": _num(rp[i, j]), "re": _num(re[i, j])})
    r.tabla("Correlaciones entre rasgos", [
        ("par", "Par de rasgos"), ("rg", "Genética (rG)", "numero", 3), ("ee_rg", "EE", "numero", 3),
        ("ic", f"IC {pct} % de rG"), ("distinta", "¿Distinta de 0?"), ("nivel", "Magnitud"),
        ("rp", "Fenotípica (rP)", "numero", 3), ("re", "Ambiental (rE)", "numero", 3)], filas, id="pares",
        nota="«¿Distinta de 0?»: Sí cuando el intervalo de confianza de rG no incluye el cero.")

    # --- selección indirecta: ¿conviene seleccionar por X para mejorar Y?
    filas = []
    for i in range(t):
        for j in range(t):
            if i == j or not (rg[i, j] == rg[i, j] and h2[i] > 0 and h2[j] > 0):
                continue
            ef = rg[i, j] * math.sqrt(h2[i]) / math.sqrt(h2[j])
            rc = rg[i, j] * math.sqrt(h2[i] * h2[j]) * math.sqrt(var_p[j]) if var_p[j] > 0 else math.nan
            if abs(ef) >= 1:
                texto = f"Seleccionar por {nombres[i]} mejora {nombres[j]} más que seleccionar por {nombres[j]} mismo"
            else:
                texto = (f"Seleccionar por {nombres[i]} logra el {100 * abs(ef):.0f} % de la respuesta de la selección "
                         f"directa por {nombres[j]}")
            if ef < 0:
                texto += " (en sentido contrario)"
            filas.append({"i": i, "j": j, "sobre": nombres[i], "respuesta": nombres[j], "eficiencia": 100 * ef,
                          "rc": rc, "interpretacion": texto + "."})
    if filas:
        r.tabla("Selección indirecta (respuesta correlacionada)", [
            ("sobre", "Se selecciona por"), ("respuesta", "Respuesta en"), ("eficiencia", "Eficiencia (%)", "numero", 1),
            ("rc", "Respuesta correlacionada (i = 1)", "numero", 4), ("interpretacion", "Interpretación")], filas,
            id="indirecta",
            nota="Eficiencia = rG·hX/hY respecto de seleccionar directamente por el otro rasgo. Respuesta correlacionada "
                 "en unidades del rasgo por cada unidad de intensidad de selección: RC = i·rG·hX·hY·σP; multiplíquela "
                 "por i (1,40 si se selecciona el 20 %; 1,76 el 10 %).")
        r.formula("Selección indirecta", "Eficiencia = rG · hX / hY;   RC_Y = i · rG · hX · hY · σP(Y)")
    if hay_rep:
        r.formula("Repetibilidad", "r = (σ²G + σ²Ep) / σ²P  (varianza entre animales / fenotípica)")
    elif d.get("nota_rep"):
        r.nota(d["nota_rep"])

    r.d["tablero"] = {"tipo": "correlacion", "conf": conf, "resumen": resumen, "rasgos": rasgos, "pares": pares,
                      "nota_rep": None if hay_rep else d.get("nota_rep")}
    if hay_rep:
        r.parametro("repetibilidades", {n: _num(v) for n, v in zip(nombres, rep)})


def _rg_ancova(payload, ctx, claves, nombres, metodo):
    op = _opciones(payload)
    conf = _confianza(op)
    reg = _registros(payload, claves, ["padre"], ["animal"])
    Y = reg.matriz_y(claves)
    Yaj, nota_aj = ajustar_por_fijos(Y, reg)
    sc, sl = codificar(reg.ids["padre"])
    animales = _ids_completos(reg.ids.get("animal"))
    anidado = animales is not None and _repetidos(animales)[0] > 0
    ctx.progreso(0.4, "Análisis de covarianza entre padres…")
    if anidado:
        ac, al, conflictos = _anidar(reg.ids["padre"], animales, "padre", "animal")
        if conflictos:
            raise ErrorDatos(f"{len(conflictos)} animales aparecen con más de un padre.")
        res = anova.anova_anidado(sc, ac, Yaj)
        G = res["componentes"]["grupo"]
        P = res["componentes"]["grupo"] + res["componentes"]["subgrupo"] + res["componentes"]["dentro"]
    else:
        res = anova.anova_una_via(sc, Yaj)
        G = res["componentes"]["entre"]
        P = G + res["componentes"]["dentro"]
    t = len(claves)
    A = 4 * G
    E = P - A
    h2 = np.diag(A) / np.diag(P)
    ee_h2 = []
    for j in range(t):
        if anidado:
            ee_h2.append(anova.delta_cm(res, lambda v: 4 * v[0] / v.sum(), j)[1])
        else:
            tt = G[j, j] / P[j, j]
            ee_h2.append(4 * anova.ee_swiger(tt, res["N"], res["grupos"], res["k"]))
    rg = np.full((t, t), np.nan)
    rp = np.full((t, t), np.nan)
    re = np.full((t, t), np.nan)
    ee_rg = np.full((t, t), np.nan)
    for i in range(t):
        for j in range(t):
            if G[i, i] > 0 and G[j, j] > 0:
                rg[i, j] = G[i, j] / math.sqrt(G[i, i] * G[j, j])
            if P[i, i] > 0 and P[j, j] > 0:
                rp[i, j] = P[i, j] / math.sqrt(P[i, i] * P[j, j])
            if E[i, i] > 0 and E[j, j] > 0:
                re[i, j] = E[i, j] / math.sqrt(E[i, i] * E[j, j])
            if i != j:
                ee_rg[i, j] = anova.ee_correlacion_genetica(rg[i, j], h2[i], ee_h2[i], h2[j], ee_h2[j])
    # repetibilidad de cada rasgo: del mismo modelo anidado, r = (σ²s + σ²a/s) / σ²P
    rep, ee_rep = np.full(t, np.nan), np.full(t, np.nan)
    if anidado:
        for j in range(t):
            rep[j], ee_rep[j] = anova.delta_cm(res, lambda v: (v[0] + v[1]) / v.sum(), j)
    r = Resultado("correlacion", "ancova",
                  "Correlación genética por análisis de covarianza de medios hermanos paternos",
                  f"{' · '.join(nombres)} · {reg.n} registros · {len(sl)} padres")
    r.nota(reg.nota_descartados(), "advertencia")
    r.nota(nota_aj)
    if anidado:
        r.nota("Hay registros repetidos por animal: se usó el modelo anidado padre / animal / registro; "
               "la correlación genética proviene del componente de padre y la repetibilidad, de padre + animal.")
    _presentar_correlacion(r, nombres, {
        "h2": h2, "ee_h2": ee_h2, "rep": rep, "ee_rep": ee_rep, "var_p": np.diag(P), "rg": rg, "ee_rg": ee_rg,
        "rp": rp, "re": re, "valores": [reg.rasgos[c] for c in claves],
        "nota_rep": None if anidado else _motivo_sin_repetibilidad(reg.ids.get("animal"))},
        conf, f"{len(sl)} padres · {reg.n} registros")
    # tabla de componentes de covarianza por par
    filas = []
    for i in range(t):
        for j in range(i, t):
            nombre = nombres[i] if i == j else f"{nombres[i]} × {nombres[j]}"
            filas.append({"par": nombre, "tipo": "Varianza" if i == j else "Covarianza",
                          "padre": G[i, j], "aditiva": A[i, j], "ambiental": E[i, j], "fenotipica": P[i, j]})
    r.tabla("Componentes de (co)varianza", [
        ("par", "Rasgo / par"), ("tipo", "Tipo"), ("padre", "Padre (σs)", "numero", 5),
        ("aditiva", "Aditiva (σA = 4σs)", "numero", 5), ("ambiental", "Ambiental (σE = σP − σA)", "numero", 5),
        ("fenotipica", "Fenotípica (σP)", "numero", 5)], filas, id="componentes")
    # análisis de covarianza (productos cruzados) del primer par
    nombres_f = (["Entre padres", "Animales dentro de padres", "Registros dentro de animales"] if anidado
                 else ["Entre padres", "Dentro de padres"])
    filas = []
    for fuente, nf in zip(res["fuentes"], nombres_f):
        fila = {"fuente": nf, "gl": fuente["gl"]}
        for i in range(t):
            fila[f"cm{i}"] = fuente["cm"][i, i]
        fila["mcp"] = fuente["cm"][0, 1]
        filas.append(fila)
    r.tabla("Cuadrados medios y productos cruzados medios",
            [("fuente", "Fuente"), ("gl", "GL", "entero")]
            + [(f"cm{i}", f"CM {nombres[i]}", "numero", 4) for i in range(t)]
            + [("mcp", f"PCM {nombres[0]} × {nombres[1]}", "numero", 4)], filas, id="ancova")
    # gráfico: medias de progenie por padre, rasgo 1 vs rasgo 2
    k = len(sl)
    n = np.bincount(sc, minlength=k)
    mx = np.bincount(sc, weights=reg.rasgos[claves[0]], minlength=k) / n
    my = np.bincount(sc, weights=reg.rasgos[claves[1]], minlength=k) / n
    b = np.polyfit(mx, my, 1) if k > 2 else [0, 0]
    r.grafico("dispersion", f"Promedio de la progenie por padre: {nombres[0]} vs. {nombres[1]}",
              puntos=list(zip(mx.tolist(), my.tolist())), etiquetas=[str(x) for x in sl],
              recta={"a": float(b[1]), "b": float(b[0])}, eje_x=nombres[0], eje_y=nombres[1])
    if t > 2:
        r.grafico("matriz_calor", "Correlaciones genéticas", nombres=nombres, valores=rg.tolist())
    r.formula("Covarianza de padre", "σs(xy) = (PCMentre − PCMdentro) / k")
    r.formula("Correlación genética", "rG = σs(xy) / √(σ²s(x) · σ²s(y))")
    r.formula("Error estándar (Robertson, 1959)", "EE(rG) = (1 − rG²)/√2 · √[ EE(h²x)·EE(h²y) / (h²x·h²y) ]")
    r.formula("Correlación ambiental", "rE = (σP(xy) − σA(xy)) / √(σ²E(x) σ²E(y))")
    _notas_rg(r, rg, h2)
    _avisos_tamano(r, len(sl), reg.n)
    r.parametro("rg", rg[0, 1])
    r.parametro("ee_rg", ee_rg[0, 1])
    r.parametro("rp", rp[0, 1])
    r.parametro("re", re[0, 1])
    return r.json()


def _notas_rg(r, rg, h2):
    if np.any(np.abs(rg[~np.isnan(rg)]) > 1):
        r.nota("Alguna correlación genética estimada está fuera de [−1, 1]: suele ocurrir con pocos padres o "
               "heredabilidades bajas. Interprétela con mucha cautela.", "advertencia")
    if np.any(h2 < 0.05):
        r.nota("Algún rasgo tiene heredabilidad muy baja: la correlación genética correspondiente es muy imprecisa.",
               "advertencia")


def _rg_reml(payload, ctx, claves, nombres, metodo):
    op = _opciones(payload)
    conf = _confianza(op)
    t = len(claves)
    if t > 4:
        raise ErrorDatos("Con REML elija como máximo 4 rasgos (se ajusta un modelo bivariado por cada par).")
    pares = [(i, j) for i in range(t) for j in range(i + 1, t)]
    rg = np.full((t, t), np.nan)
    rp = np.full((t, t), np.nan)
    re = np.full((t, t), np.nan)
    ee_rg = np.full((t, t), np.nan)
    h2s = [[] for _ in range(t)]
    ee_h2s = [[] for _ in range(t)]
    reps, ee_reps, vps = [[] for _ in range(t)], [[] for _ in range(t)], [[] for _ in range(t)]
    valores = [None] * t
    motivo_rep = None
    r = Resultado("correlacion", metodo,
                  "Correlación genética por REML bivariado · " + ("modelo animal" if metodo == "reml_animal"
                                                                  else "modelo de padre"), "")
    filas_comp = []
    filas_ajuste = []
    ultimo = None
    for n_par, (i, j) in enumerate(pares):
        ctx.progreso(n_par / len(pares), f"Par {n_par + 1} de {len(pares)}: {nombres[i]} × {nombres[j]}")
        sub = [claves[i], claves[j]]
        if metodo == "reml_animal":
            reg, X, nombres_x, notas_x, Y, aleatorios, etiquetas, inicio, ped = _modelo_animal(payload, ctx, sub)
            k_gen, factor = 0, 1.0
        else:
            reg, X, nombres_x, notas_x, Y, aleatorios, etiquetas, inicio, extra = _modelo_padre(payload, ctx, sub)
            k_gen, factor = etiquetas.index("padre"), 4.0
        modelo, res = _ajustar(Y, X, aleatorios, inicio, ctx)
        ultimo = (reg, modelo, res, etiquetas, nombres_x, notas_x)
        K = len(aleatorios)
        v = reml.vector_parametros(res)
        cov = res.cov_param

        def rg_f(w, k=k_gen):
            G = reml.matrices_de_vector(w, 2, K)[1][k]
            return G[0, 1] / math.sqrt(G[0, 0] * G[1, 1]) if G[0, 0] > 0 and G[1, 1] > 0 else math.nan

        val, ee = reml.delta(rg_f, v, cov)
        rg[i, j] = rg[j, i] = val
        ee_rg[i, j] = ee_rg[j, i] = ee
        P = res.R + sum(res.G)
        A = factor * res.G[k_gen]
        E = P - A
        rp[i, j] = rp[j, i] = P[0, 1] / math.sqrt(P[0, 0] * P[1, 1])
        if E[0, 0] > 0 and E[1, 1] > 0:
            re[i, j] = re[j, i] = E[0, 1] / math.sqrt(E[0, 0] * E[1, 1])
        for a, idx in ((0, i), (1, j)):
            hv, he = reml.delta(lambda w, a=a: factor * reml.matrices_de_vector(w, 2, K)[1][k_gen][a, a]
                                / _fenotipica(w, 2, K, a, a), v, cov)
            h2s[idx].append(hv)
            ee_h2s[idx].append(he)
            vps[idx].append(float(P[a, a]))
            if valores[idx] is None:
                valores[idx] = reg.rasgos[sub[a]]
        # repetibilidad con el efecto de ambiente permanente del mismo ajuste:
        # padre: (σ²s + σ²animal/padre) / σ²P · animal: (σ²A + σ²pe) / σ²P
        if "pe" in etiquetas:
            k_pe = etiquetas.index("pe")
            for a, idx in ((0, i), (1, j)):
                rv, rerr = reml.delta(lambda w, a=a: (reml.matrices_de_vector(w, 2, K)[1][k_gen][a, a]
                                                      + reml.matrices_de_vector(w, 2, K)[1][k_pe][a, a])
                                      / _fenotipica(w, 2, K, a, a), v, cov)
                reps[idx].append(rv)
                ee_reps[idx].append(rerr)
        else:
            motivo_rep = _motivo_sin_repetibilidad(reg.ids.get("animal"))
        nombres_ef = {"padre": "Padre", "madre": "Madre", "pe": "Ambiente permanente", "animal": "Aditivo (animal)"}
        for k, et in enumerate(etiquetas):
            Gk = res.G[k]
            for a, b in ((0, 0), (1, 1), (0, 1)):
                _, ee_c = reml.delta(_funcion_componente(2, K, k, a, b), v, cov)
                filas_comp.append({"par": f"{nombres[i]} × {nombres[j]}", "efecto": nombres_ef.get(et, et),
                                   "elemento": _elem(nombres[i], nombres[j], a, b), "valor": Gk[a, b], "ee": ee_c})
        for a, b in ((0, 0), (1, 1), (0, 1)):
            _, ee_c = reml.delta(_funcion_componente(2, K, -1, a, b), v, cov)
            filas_comp.append({"par": f"{nombres[i]} × {nombres[j]}", "efecto": "Residual",
                               "elemento": _elem(nombres[i], nombres[j], a, b), "valor": res.R[a, b], "ee": ee_c})
        filas_ajuste.append({"par": f"{nombres[i]} × {nombres[j]}", "registros": modelo.n, "m2logl": res.m2logl,
                             "evaluaciones": res.evaluaciones, "segundos": res.segundos,
                             "convergencia": "Sí" if res.convergio else "Revisar"})
        r.nota(reg.nota_descartados(), "advertencia")
        for nx in notas_x:
            r.nota(nx, "advertencia")
        if res.cov_param is None:
            r.nota(f"Par {nombres[i]} × {nombres[j]}: no se pudieron calcular todos los errores estándar "
                   "(algún parámetro en el límite).", "advertencia")
    media = lambda xs: float(np.mean(xs)) if xs else math.nan  # noqa: E731
    h2 = np.array([media(x) for x in h2s])
    ee_h2 = np.array([media(x) for x in ee_h2s])
    reg = ultimo[0]
    r.d["subtitulo"] = f"{' · '.join(nombres)} · {reg.n} registros completos"
    n_padres = (f"{len(set(reg.ids['padre']))} padres · " if metodo == "reml_padre" and reg.ids.get("padre")
                else f"{ultimo[1].aleatorios[0].q} animales en el pedigrí · " if metodo == "reml_animal" else "")
    _presentar_correlacion(r, nombres, {
        "h2": h2, "ee_h2": ee_h2, "rep": np.array([media(x) for x in reps]),
        "ee_rep": np.array([media(x) for x in ee_reps]), "var_p": np.array([media(x) for x in vps]),
        "rg": rg, "ee_rg": ee_rg, "rp": rp, "re": re, "valores": valores, "nota_rep": motivo_rep},
        conf, f"{n_padres}{reg.n} registros")
    if t > 2:
        r.nota("Con más de dos rasgos, h² y r de cada rasgo son el promedio de sus ajustes bivariados.")
    else:
        # valores de cría de ambos rasgos del ajuste bivariado: muestra la correlación genética «en los animales»
        _, modelo_u, res_u, etiquetas_u = ultimo[:4]
        k_u = 0 if metodo == "reml_animal" else etiquetas_u.index("padre")
        sols = modelo_u.soluciones(*res_u.std)
        u = sols["aleatorios"][k_u] * (1.0 if metodo == "reml_animal" else 2.0)
        if len(u) > 2 and np.ptp(u[:, 0]) > 0:
            etiquetas_u = [str(x) for x in (modelo_u.aleatorios[k_u].etiquetas or range(len(u)))]
            idx = np.arange(len(u))
            if len(idx) > 3000:
                idx = np.sort(np.random.default_rng(1).choice(len(u), 3000, replace=False))
            b = np.polyfit(u[:, 0], u[:, 1], 1)
            quien = "animales" if metodo == "reml_animal" else "padres"
            r.grafico("dispersion", f"Valores de cría de los {quien}: {nombres[0]} vs. {nombres[1]}",
                      puntos=[(float(u[k, 0]), float(u[k, 1])) for k in idx], etiquetas=[etiquetas_u[k] for k in idx],
                      recta={"a": float(b[1]), "b": float(b[0])},
                      eje_x=f"VC {nombres[0]}", eje_y=f"VC {nombres[1]}")
    r.tabla("Componentes de (co)varianza (REML)", [
        ("par", "Par de rasgos"), ("efecto", "Efecto"), ("elemento", "Elemento"),
        ("valor", "Estimación", "numero", 5), ("ee", "Error estándar", "numero", 5)], filas_comp, id="componentes")
    r.tabla("Información de los ajustes", [("par", "Par"), ("registros", "Registros", "entero"),
                                           ("m2logl", "−2 log L", "numero", 3), ("evaluaciones", "Evaluaciones", "entero"),
                                           ("segundos", "Segundos", "numero", 1), ("convergencia", "Convergencia")],
            filas_ajuste, id="ajuste")
    if t > 2:
        r.grafico("matriz_calor", "Correlaciones genéticas", nombres=nombres, valores=rg.tolist())
    r.formula("Modelo bivariado", "y = Xb + Zu + e;  var(u) = G0 ⊗ " + ("A" if metodo == "reml_animal" else "I")
              + ",  var(e) = R0 ⊗ I")
    r.formula("Correlación genética", "rG = σu(xy) / √(σ²u(x) σ²u(y))")
    r.formula("Errores estándar", "Matriz de información observada (hessiano numérico) y método delta")
    _notas_rg(r, rg, h2)
    r.nota("Solo se usan los registros con ambos rasgos medidos.")
    r.parametro("rg", rg[0, 1])
    r.parametro("ee_rg", ee_rg[0, 1])
    r.parametro("rp", rp[0, 1])
    r.parametro("re", re[0, 1])
    return r.json()


def _elem(a, b, i, j):
    if i == j:
        return f"σ² {a if i == 0 else b}"
    return "covarianza"


def _rg_cruzada(payload, ctx, claves, nombres, metodo):
    """rA = Cov(XpYo)·½+Cov(YpXo)·½ / √(Cov(XpXo)·Cov(YpYo))  (Falconer y Mackay, cap. 19)."""
    op = _opciones(payload)
    if len(claves) != 2:
        raise ErrorDatos("La covarianza cruzada progenitor–progenie se calcula para un par de rasgos.")
    tipo = op.get("tipo_regresion") or "padre"
    reg = _registros(payload, claves, ["animal"], ["padre", "madre"], con_fijos=False)
    cod, niv = codificar(reg.ids["animal"])
    n = np.bincount(cod)
    fx = dict(zip(niv, np.bincount(cod, weights=reg.rasgos[claves[0]]) / n))
    fy = dict(zip(niv, np.bincount(cod, weights=reg.rasgos[claves[1]]) / n))
    pa = reg.ids.get("padre") or [None] * reg.n
    ma = reg.ids.get("madre") or [None] * reg.n
    padre_de, madre_de = {}, {}
    for a, p, m in zip(reg.ids["animal"], pa, ma):
        padre_de.setdefault(a, p)
        madre_de.setdefault(a, m)
    ped = payload.get("pedigri")
    if ped:
        ids_p, pad_p, mad_p, _ = _leer_pedigri(ped)
        for a, p, m in zip(ids_p, pad_p, mad_p):
            if a is not None:
                padre_de[a], madre_de[a] = p, m
    pares = []
    for a in niv:
        progs = []
        if tipo in ("padre", "ambos"):
            progs.append(padre_de.get(a))
        if tipo in ("madre", "ambos"):
            progs.append(madre_de.get(a))
        for p in progs:
            if p is not None and p in fx:
                pares.append((fx[p], fy[p], fx[a], fy[a]))
    if len(pares) < 10:
        raise ErrorDatos("Hay menos de 10 pares progenitor–progenie con ambos rasgos medidos en los dos.")
    M = np.array(pares)
    C = np.cov(M, rowvar=False)
    cxx, cyy = C[0, 2], C[1, 3]
    cxy = (C[0, 3] + C[1, 2]) / 2
    rg = cxy / math.sqrt(cxx * cyy) if cxx > 0 and cyy > 0 else math.nan
    h2x = 2 * cxx / C[0, 0]
    h2y = 2 * cyy / C[1, 1]
    # error estándar por remuestreo (bootstrap de pares)
    rng = np.random.default_rng(7)
    boots = []
    for _ in range(500):
        mm = M[rng.integers(0, len(M), len(M))]
        cc = np.cov(mm, rowvar=False)
        if cc[0, 2] > 0 and cc[1, 3] > 0:
            boots.append((((cc[0, 3] + cc[1, 2]) / 2) / math.sqrt(cc[0, 2] * cc[1, 3]),
                          2 * cc[0, 2] / cc[0, 0], 2 * cc[1, 3] / cc[1, 1]))
    if len(boots) > 30:
        ee, ee_hx, ee_hy = (float(x) for x in np.std(np.array(boots), axis=0, ddof=1))
    else:
        ee = ee_hx = ee_hy = math.nan
    rp = float(np.corrcoef(M[:, 2], M[:, 3])[0, 1])
    r = Resultado("correlacion", "cruzada", "Correlación genética por covarianza cruzada progenitor–progenie",
                  f"{nombres[0]} × {nombres[1]} · {len(pares)} pares")
    # repetibilidad de cada rasgo entre animales (si hay registros repetidos)
    if _repetidos(reg.ids["animal"])[0] > 0:
        rep, ee_rep = _rep_una_via(reg.ids["animal"], reg.matriz_y(claves))
        motivo = None
    else:
        rep, ee_rep = np.full(2, np.nan), np.full(2, np.nan)
        motivo = _motivo_sin_repetibilidad(reg.ids["animal"])
    mat = lambda v: np.array([[1.0, v], [v, 1.0]])  # noqa: E731
    _presentar_correlacion(r, nombres, {
        "h2": [h2x, h2y], "ee_h2": [ee_hx, ee_hy], "rep": rep, "ee_rep": ee_rep, "var_p": [C[2, 2], C[3, 3]],
        "rg": mat(rg), "ee_rg": mat(ee), "rp": mat(rp), "re": mat(math.nan),
        "valores": [reg.rasgos[c] for c in claves], "nota_rep": motivo}, _confianza(op), f"{len(pares)} pares progenitor–progenie")
    r.grafico("dispersion", f"Progenitor vs. progenie: {nombres[0]} del progenitor y {nombres[1]} de la cría",
              puntos=[(float(a), float(b)) for a, b in M[:3000, [0, 3]]], eje_x=f"{nombres[0]} (progenitor)",
              eje_y=f"{nombres[1]} (progenie)")
    r.tabla("Covarianzas progenitor–progenie", [("cov", "Covarianza"), ("valor", "Valor", "numero", 5)], [
        {"cov": f"Cov({nombres[0]} progenitor, {nombres[0]} progenie)", "valor": cxx},
        {"cov": f"Cov({nombres[1]} progenitor, {nombres[1]} progenie)", "valor": cyy},
        {"cov": f"Cov({nombres[0]} progenitor, {nombres[1]} progenie)", "valor": C[0, 3]},
        {"cov": f"Cov({nombres[1]} progenitor, {nombres[0]} progenie)", "valor": C[1, 2]},
    ], id="covarianzas")
    r.formula("Correlación genética", "rG = ½[Cov(Xp,Yo) + Cov(Yp,Xo)] / √[Cov(Xp,Xo)·Cov(Yp,Yo)]")
    r.nota("Los errores estándar de rG y h² se obtuvieron por bootstrap (500 remuestreos de pares); h² = 2b "
           "(regresión de la progenie sobre un progenitor). La correlación ambiental no se estima con este método.")
    r.parametro("rg", rg)
    r.parametro("ee_rg", ee)
    return r.json()
