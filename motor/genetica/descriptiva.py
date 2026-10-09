"""Estadística descriptiva de las columnas numéricas y correlaciones fenotípicas."""
import math

import numpy as np
from scipy import stats

from .datos import ErrorDatos, a_id, columna_numerica
from .parametros import _faltantes, _opciones
from .resultados import Resultado, histograma


def descriptiva(payload, ctx):
    op = _opciones(payload)
    tabla = payload.get("tabla") or {}
    vals = tabla.get("valores") or {}
    columnas = [c for c in (op.get("columnas") or []) if c in vals]
    if not columnas:
        raise ErrorDatos("Elija al menos una columna numérica.")
    falt = _faltantes(op)
    datos = {c: columna_numerica(vals[c], falt) for c in columnas}
    grupo = op.get("grupo")
    r = Resultado("descriptiva", "descriptiva", "Estadística descriptiva",
                  f"{len(columnas)} variables · {len(next(iter(datos.values())))} filas")
    filas = []
    for c in columnas:
        x = datos[c]
        v = x[~np.isnan(x)]
        n = len(v)
        if n == 0:
            filas.append({"variable": c, "n": 0, "faltantes": len(x)})
            continue
        media = float(v.mean())
        de = float(v.std(ddof=1)) if n > 1 else math.nan
        if abs(media) < 1e-9 * max(1.0, de if de == de else 1.0):
            media = 0.0  # ruido de redondeo (p. ej. valores de cría centrados en cero)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        normal_p = math.nan
        prueba = ""
        if 3 <= n <= 5000:
            normal_p = float(stats.shapiro(v).pvalue)
            prueba = "Shapiro–Wilk"
        elif n > 5000:
            normal_p = float(stats.normaltest(v).pvalue)
            prueba = "D'Agostino"
        z = (v - media) / de if de and de > 0 else np.zeros(n)
        filas.append({
            "variable": c, "n": n, "faltantes": int(len(x) - n), "media": media, "de": de,
            "ee": de / math.sqrt(n) if n > 1 else None,
            "cv": 100 * de / media if media > 0 else None,  # el CV solo tiene sentido con media positiva
            "min": float(v.min()), "q1": float(q1), "mediana": float(med), "q3": float(q3), "max": float(v.max()),
            "asimetria": float(stats.skew(v)) if n > 2 else None,
            "curtosis": float(stats.kurtosis(v)) if n > 3 else None,
            "normalidad": normal_p, "prueba": prueba, "atipicos": int(np.sum(np.abs(z) > 3)),
        })
        r.grafico("histograma", f"Distribución · {c}", **histograma(v), eje=c, curva_normal=True)
    r.tabla("Resumen por variable", [
        ("variable", "Variable"), ("n", "n", "entero"), ("faltantes", "Faltantes", "entero"),
        ("media", "Media", "numero", 4), ("de", "DE", "numero", 4), ("ee", "EE", "numero", 4),
        ("cv", "CV %", "numero", 2), ("min", "Mín.", "numero", 4), ("q1", "Q1", "numero", 4),
        ("mediana", "Mediana", "numero", 4), ("q3", "Q3", "numero", 4), ("max", "Máx.", "numero", 4),
        ("asimetria", "Asimetría", "numero", 3), ("curtosis", "Curtosis", "numero", 3),
        ("normalidad", "p normalidad", "p"), ("atipicos", "Atípicos (|z|>3)", "entero")], filas, id="resumen",
        nota="Normalidad: Shapiro–Wilk (n ≤ 5000) o D'Agostino (n > 5000); p < 0,05 indica desvío de la normal.")
    if len(columnas) >= 2:
        k = len(columnas)
        R = np.full((k, k), np.nan)
        P = np.full((k, k), np.nan)
        N = np.zeros((k, k), dtype=int)
        for i in range(k):
            for j in range(k):
                m = ~np.isnan(datos[columnas[i]]) & ~np.isnan(datos[columnas[j]])
                N[i, j] = m.sum()
                if N[i, j] > 2:
                    if i == j:
                        R[i, j], P[i, j] = 1.0, 0.0
                    else:
                        res = stats.pearsonr(datos[columnas[i]][m], datos[columnas[j]][m])
                        R[i, j], P[i, j] = float(res.statistic), float(res.pvalue)
        filas = []
        for i, a in enumerate(columnas):
            fila = {"variable": a}
            for j in range(k):
                if i == j:
                    fila[f"c{j}"] = "1"
                elif R[i, j] == R[i, j]:
                    estrellas = "***" if P[i, j] < 0.001 else "**" if P[i, j] < 0.01 else "*" if P[i, j] < 0.05 else ""
                    fila[f"c{j}"] = f"{R[i, j]:.3f}{estrellas}"
                else:
                    fila[f"c{j}"] = "—"
            filas.append(fila)
        r.tabla("Correlaciones fenotípicas (Pearson)", [("variable", "")] + [(f"c{j}", c) for j, c in enumerate(columnas)],
                filas, id="correlaciones", nota="* p < 0,05 · ** p < 0,01 · *** p < 0,001 (pares completos).")
        r.grafico("matriz_calor", "Correlaciones fenotípicas", nombres=columnas, valores=R.tolist())
        if k == 2:
            m = ~np.isnan(datos[columnas[0]]) & ~np.isnan(datos[columnas[1]])
            x, y = datos[columnas[0]][m], datos[columnas[1]][m]
            puntos = list(zip(x.tolist(), y.tolist()))
            if len(puntos) > 3000:
                idx = np.random.default_rng(1).choice(len(puntos), 3000, replace=False)
                puntos = [puntos[i] for i in sorted(idx)]
            b = np.polyfit(x, y, 1) if len(x) > 2 else [0, 0]
            r.grafico("dispersion", f"{columnas[0]} vs. {columnas[1]}", puntos=puntos,
                      recta={"a": float(b[1]), "b": float(b[0])}, eje_x=columnas[0], eje_y=columnas[1])
    if grupo and grupo in vals:
        g = [a_id(v) for v in vals[grupo]]
        niveles = []
        vistos = set()
        for x in g:
            if x is not None and x not in vistos:
                vistos.add(x)
                niveles.append(x)
        filas = []
        for niv in niveles[:500]:
            m = np.array([x == niv for x in g])
            fila = {"grupo": niv}
            for c in columnas:
                v = datos[c][m]
                v = v[~np.isnan(v)]
                fila[f"n_{c}"] = len(v)
                fila[f"m_{c}"] = float(v.mean()) if len(v) else None
                fila[f"d_{c}"] = float(v.std(ddof=1)) if len(v) > 1 else None
            filas.append(fila)
        cols = [("grupo", grupo)]
        for c in columnas:
            cols += [(f"n_{c}", f"n {c}", "entero"), (f"m_{c}", f"Media {c}", "numero", 3),
                     (f"d_{c}", f"DE {c}", "numero", 3)]
        r.tabla(f"Resumen por {grupo}", cols, filas, id="por_grupo",
                nota=f"{len(niveles)} grupos" + (" (se muestran 500)" if len(niveles) > 500 else ""))
        c = columnas[0]
        grupos = [{"grupo": f["grupo"], "media": f[f"m_{c}"], "n": f[f"n_{c}"],
                   "ee": (f[f"d_{c}"] / math.sqrt(f[f"n_{c}"])) if f[f"d_{c}"] and f[f"n_{c}"] else None}
                  for f in filas if f[f"m_{c}"] is not None]
        grupos.sort(key=lambda x: x["media"])
        v = datos[c]
        r.grafico("medias_grupo", f"Media de {c} por {grupo}", grupos=grupos[:400], etiqueta_grupo=grupo,
                  media_general=float(np.nanmean(v)))
    return r.json()
