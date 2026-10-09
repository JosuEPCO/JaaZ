"""Preparación de datos para los análisis: valores faltantes, identificadores,
factores y matriz de efectos fijos.

Los datos llegan del navegador como columnas (listas JSON). Aquí se convierten a
arreglos de numpy, se descartan los registros incompletos y se informa cuántos
quedaron fuera, para que el usuario siempre sepa con qué datos se calculó.
"""
import math
import re

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

# Códigos habituales de "padre/madre desconocido" (igual que Pedigree Viewer: 0 # * . "")
PADRE_DESCONOCIDO = {"", "0", "0.0", ".", "#", "*", "-", "?", "na", "n/a", "nan", "null", "none",
                     "desconocido", "desc", "s/d", "sd"}
TEXTO_FALTANTE = {"", ".", "na", "n/a", "nan", "null", "none", "-", "?", "s/d"}
_NUM_COMA = re.compile(r"^[+-]?\d+,\d+$")


class ErrorDatos(ValueError):
    """Problema en los datos con un mensaje claro para el usuario."""


class Cancelado(Exception):
    """El usuario canceló el análisis."""


def a_numero(v, faltantes=()):
    """Convierte un valor de celda a float; NaN si falta o no es numérico.

    Acepta coma decimal ("5,53") porque Excel en español la exporta así.
    """
    if v is None or isinstance(v, bool):
        return math.nan
    if isinstance(v, (int, float)):
        x = float(v)
    else:
        s = str(v).strip()
        if s.lower() in TEXTO_FALTANTE:
            return math.nan
        if _NUM_COMA.match(s):
            s = s.replace(",", ".")
        try:
            x = float(s)
        except ValueError:
            return math.nan
    if math.isnan(x) or math.isinf(x):
        return math.nan
    if faltantes and x in faltantes:
        return math.nan
    return x


def columna_numerica(valores, faltantes=()):
    return np.array([a_numero(v, faltantes) for v in valores], dtype=float)


def a_id(v, desconocidos=PADRE_DESCONOCIDO):
    """Normaliza un identificador: 12.0 -> "12", espacios fuera; None si es desconocido."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if v.is_integer():
            v = int(v)
    s = str(v).strip()
    if s.lower() in desconocidos:
        return None
    return s


def columna_ids(valores, permitir_cero=False):
    desconocidos = PADRE_DESCONOCIDO - {"0", "0.0"} if permitir_cero else PADRE_DESCONOCIDO
    return [a_id(v, desconocidos) for v in valores]


def codificar(etiquetas):
    """Etiquetas -> (códigos 0..k-1, lista de niveles en orden de aparición)."""
    niveles = {}
    codigos = np.empty(len(etiquetas), dtype=np.int64)
    for i, e in enumerate(etiquetas):
        c = niveles.get(e)
        if c is None:
            c = niveles[e] = len(niveles)
        codigos[i] = c
    return codigos, list(niveles)


def obtener(tabla, roles, rol, requerido=True):
    """Devuelve la lista de valores de la columna asignada a `rol`."""
    nombre = (roles or {}).get(rol)
    if not nombre:
        if requerido:
            raise ErrorDatos(f"Falta asignar la columna «{rol}».")
        return None
    columnas = tabla.get("valores", {})
    if nombre not in columnas:
        raise ErrorDatos(f"La columna «{nombre}» no está en los datos enviados.")
    return columnas[nombre]


def nombre_de(roles, rol):
    return (roles or {}).get(rol) or rol


class Registros:
    """Conjunto de registros completos listo para analizar.

    Reúne rasgos (numéricos), identificadores (texto) y efectos fijos, y descarta
    toda fila a la que le falte algo de lo requerido.
    """

    def __init__(self, tabla, roles, rasgos, ids=(), ids_opcionales=(), faltantes=(),
                 fijos=(), covariables=()):
        n = int(tabla.get("n") or 0)
        if n == 0:
            primera = next(iter(tabla.get("valores", {}).values()), [])
            n = len(primera)
        if n == 0:
            raise ErrorDatos("La hoja no tiene filas de datos.")
        self.n_original = n
        self.nombres = {}
        validos = np.ones(n, dtype=bool)

        self.rasgos = {}
        for rol in rasgos:
            col = columna_numerica(obtener(tabla, roles, rol), faltantes)
            if len(col) != n:
                raise ErrorDatos("Las columnas enviadas no tienen la misma longitud.")
            self.rasgos[rol] = col
            self.nombres[rol] = nombre_de(roles, rol)
            validos &= ~np.isnan(col)

        self.ids = {}
        for rol in list(ids) + list(ids_opcionales):
            valores = obtener(tabla, roles, rol, requerido=rol in ids)
            if valores is None:
                continue
            col = columna_ids(valores, permitir_cero=False)
            self.ids[rol] = col
            self.nombres[rol] = nombre_de(roles, rol)
            if rol in ids:
                validos &= np.array([c is not None for c in col])

        self.fijos = []
        for nombre in fijos or ():
            valores = tabla.get("valores", {}).get(nombre)
            if valores is None:
                raise ErrorDatos(f"La columna de efecto fijo «{nombre}» no está en los datos.")
            col = [a_id(v, TEXTO_FALTANTE) for v in valores]
            self.fijos.append((nombre, col))
            validos &= np.array([c is not None for c in col])

        self.covariables = []
        for nombre in covariables or ():
            valores = tabla.get("valores", {}).get(nombre)
            if valores is None:
                raise ErrorDatos(f"La covariable «{nombre}» no está en los datos.")
            col = columna_numerica(valores, faltantes)
            self.covariables.append((nombre, col))
            validos &= ~np.isnan(col)

        self.indice = np.flatnonzero(validos)
        self.n = len(self.indice)
        self.descartados = n - self.n
        if self.n == 0:
            raise ErrorDatos("No queda ningún registro completo con las columnas elegidas.")
        for rol in self.rasgos:
            self.rasgos[rol] = self.rasgos[rol][self.indice]
        for rol in self.ids:
            col = self.ids[rol]
            self.ids[rol] = [col[i] for i in self.indice]
        self.fijos = [(nom, [col[i] for i in self.indice]) for nom, col in self.fijos]
        self.covariables = [(nom, col[self.indice]) for nom, col in self.covariables]

    def matriz_y(self, roles_rasgos):
        return np.column_stack([self.rasgos[r] for r in roles_rasgos])

    def nota_descartados(self):
        if self.descartados:
            return (f"Se excluyeron {self.descartados} de {self.n_original} filas por valores faltantes "
                    f"en las columnas usadas; se analizaron {self.n} registros.")
        return None


def matriz_fija(registros):
    """Matriz de incidencia de efectos fijos: media + factores (codificación de referencia)
    + covariables centradas. Elimina columnas linealmente dependientes.

    Devuelve (X csr, nombres de columnas, notas).
    """
    n = registros.n
    bloques = [np.ones((n, 1))]
    nombres = ["Media general"]
    notas = []
    filas_sp = []
    for nombre, col in registros.fijos:
        codigos, niveles = codificar(col)
        if len(niveles) < 2:
            notas.append(f"El efecto fijo «{nombre}» tiene un solo nivel y se omitió.")
            continue
        m = sp.csr_matrix((np.ones(n), (np.arange(n), codigos)), shape=(n, len(niveles)))[:, 1:]
        filas_sp.append(m)
        nombres += [f"{nombre} = {niv}" for niv in niveles[1:]]
    for nombre, col in registros.covariables:
        c = col - col.mean()
        if np.allclose(c, 0):
            notas.append(f"La covariable «{nombre}» es constante y se omitió.")
            continue
        bloques.append(c.reshape(-1, 1))
        nombres.append(f"{nombre} (regresión lineal)")
    densa = sp.csr_matrix(np.hstack(bloques))
    X = sp.hstack([densa[:, :1]] + filas_sp + [densa[:, 1:]]).tocsr() if filas_sp else densa
    # reordenar nombres en el mismo orden que las columnas
    n_cov = densa.shape[1] - 1
    nombres = [nombres[0]] + nombres[1 + n_cov:] + nombres[1:1 + n_cov]

    XtX = (X.T @ X).toarray()
    if XtX.shape[0] > 1:
        _, R, piv = sla.qr(XtX, pivoting=True)
        diag = np.abs(np.diag(R))
        rango = int(np.sum(diag > diag[0] * 1e-10)) if diag[0] > 0 else 0
        if rango < XtX.shape[0]:
            quitar = sorted(piv[rango:])
            notas.append("Se eliminaron columnas de efectos fijos redundantes (confundidas): "
                         + ", ".join(nombres[i] for i in quitar[:6]) + ("…" if len(quitar) > 6 else "") + ".")
            mantener = sorted(piv[:rango])
            X = X[:, mantener]
            nombres = [nombres[i] for i in mantener]
    return X.tocsr(), nombres, notas


def residuos_fijos(X, Y):
    """Residuos de mínimos cuadrados de Y (n x t) sobre la matriz dispersa X (ya de rango completo).

    Usa las ecuaciones normales X'X b = X'Y, que son p x p: nunca arma X densa (n x p), que con
    muchos registros y factores de muchos niveles (hato-año) no cabría en memoria."""
    Y = np.asarray(Y, dtype=float).reshape(X.shape[0], -1)
    XtX = (X.T @ X).toarray()
    XtY = np.asarray(X.T @ Y)
    try:
        beta = sla.solve(XtX, XtY, assume_a="pos")
    except (np.linalg.LinAlgError, ValueError):
        beta = np.linalg.lstsq(XtX, XtY, rcond=None)[0]
    return Y - np.asarray(X @ beta)


def ajustar_por_fijos(y, registros):
    """Residuos de mínimos cuadrados + media: corrige los datos por efectos fijos
    antes de un ANOVA (aproximación; para un ajuste exacto se usa REML)."""
    if not registros.fijos and not registros.covariables:
        return y, None
    X, nombres, _ = matriz_fija(registros)
    Y = y.reshape(len(y), -1)
    ajustado = residuos_fijos(X, Y) + Y.mean(axis=0)
    nota = ("Los datos se corrigieron por los efectos fijos (" + ", ".join(
        [n for n, _ in registros.fijos] + [n for n, _ in registros.covariables])
        + ") mediante mínimos cuadrados antes del ANOVA. Es una aproximación: "
          "para estimar todo a la vez use el método REML.")
    return ajustado.reshape(y.shape), nota
