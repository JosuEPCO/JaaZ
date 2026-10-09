"""Máxima verosimilitud restringida (REML) libre de derivadas para modelos mixtos
univariados y bivariados:

    y = X b + Z_1 u_1 + ... + Z_K u_K + e
    var(u_k) = G0_k ⊗ A_k   (A_k = matriz de parentesco, o identidad)
    var(e)   = R0 ⊗ I

-2 log L = n log|R0| + Σ q_k log|G0_k| + log|C| + y'R⁻¹y − ŝ'r   (+ constante)

donde C y r son la matriz y el lado derecho de las ecuaciones del modelo mixto
(Graser, Smith y Tier 1987; Meyer 1989). El log-determinante de C se obtiene por
factorización de Cholesky (densa) o LU dispersa. El efecto iid con más niveles
(p. ej. ambiente permanente, o el padre en un modelo de padre) se absorbe nivel
por nivel en bloques t x t, lo que hace cada evaluación muy rápida.

Los errores estándar salen de la matriz de información observada (hessiano
numérico de -2 log L) y el método delta para h², r y rG.
"""
import math
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import splu

from .datos import Cancelado, ErrorDatos


class Aleatorio:
    """Efecto aleatorio: códigos de nivel por registro y estructura de covarianza."""

    def __init__(self, nombre, codigos, q, Ainv=None, etiquetas=None, logdet_a=0.0):
        codigos = np.asarray(codigos, dtype=np.int64)
        n = len(codigos)
        self.nombre = nombre
        self.q = int(q)
        self.Z = sp.csr_matrix((np.ones(n), (np.arange(n), codigos)), shape=(n, self.q))
        self.Ainv = Ainv.tocsr() if Ainv is not None else None
        self.etiquetas = etiquetas
        self.logdet_a = logdet_a

    @property
    def iid(self):
        return self.Ainv is None


class ModeloMixto:
    """Ecuaciones del modelo mixto preparadas para evaluar -2 log L muchas veces."""

    def __init__(self, Y, X, aleatorios, absorber="auto"):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.n, self.t = Y.shape
        self.X = X.tocsr()
        self.p = X.shape[1]
        self.aleatorios = list(aleatorios)
        # estandarizar: estabilidad numérica; los resultados se re-escalan después
        self.media = Y.mean(axis=0)
        self.de = Y.std(axis=0, ddof=1)
        self.de[self.de == 0] = 1.0
        self.Y = (Y - self.media) / self.de

        self.abs = None
        if absorber == "auto":
            candidatos = [k for k, a in enumerate(self.aleatorios) if a.iid and a.q > 60]
            if candidatos:
                self.abs = max(candidatos, key=lambda k: self.aleatorios[k].q)
        elif absorber is not None:
            self.abs = absorber
        self.resto = [k for k in range(len(self.aleatorios)) if k != self.abs]

        bloques = [self.X] + [self.aleatorios[k].Z for k in self.resto]
        Wr = sp.hstack(bloques).tocsc()
        self.dim_r = Wr.shape[1]
        self.WrWr = (Wr.T @ Wr).tocsr()
        self.WrY = np.asarray(Wr.T @ self.Y)
        self.YY = self.Y.T @ self.Y
        # bloque de cada efecto no absorbido dentro de la ecuación de un rasgo
        self.inicio = {}
        pos = self.p
        self.matG = {}
        for k in self.resto:
            a = self.aleatorios[k]
            self.inicio[k] = pos
            A = (a.Ainv if a.Ainv is not None else sp.identity(a.q, format="csr")).tocoo()
            self.matG[k] = sp.csr_matrix((A.data, (A.row + pos, A.col + pos)), shape=(self.dim_r, self.dim_r))
            pos += a.q
        if self.abs is not None:
            Za = self.aleatorios[self.abs].Z
            self.conteo = np.asarray(Za.sum(axis=0)).ravel()
            self.M = (Wr.T @ Za).tocsr()
            self.ZaY = np.asarray(Za.T @ self.Y)
        # Cholesky densa solo para sistemas chicos: con ~1200 ecuaciones la LU dispersa ya es ~10 veces
        # más rápida por evaluación (las MME del modelo animal son muy dispersas)
        self.densa = self.t * self.dim_r <= 300
        self.evaluaciones = 0

    # -------------------------------------------------------------- ecuaciones
    def _sistema(self, R0, Gs):
        t = self.t
        Rinv = np.linalg.inv(R0)
        C = sp.kron(Rinv, self.WrWr, format="csr")
        for k in self.resto:
            C = C + sp.kron(np.linalg.inv(Gs[k]), self.matG[k], format="csr")
        r = (self.WrY @ Rinv).T.ravel()
        extra_logdet = 0.0
        extra_sr = 0.0
        datos_abs = None
        if self.abs is not None:
            Ga_inv = np.linalg.inv(Gs[self.abs])
            B = self.conteo[:, None, None] * Rinv[None, :, :] + Ga_inv[None, :, :]
            signo, ld = np.linalg.slogdet(B)
            if np.any(signo <= 0):
                raise np.linalg.LinAlgError("bloque absorbido no definido positivo")
            extra_logdet = float(ld.sum())
            Binv = np.linalg.inv(B)
            H = np.einsum("ij,ljk,km->lim", Rinv, Binv, Rinv)
            for i in range(t):
                for j in range(i, t):
                    K = (self.M @ sp.diags(H[:, i, j]) @ self.M.T).tocsr()
                    E = np.zeros((t, t))
                    E[i, j] = 1.0
                    if i != j:
                        E[j, i] = 1.0
                    C = C - sp.kron(E, K, format="csr")
            ra = self.ZaY @ Rinv                       # (qa, t)
            v = np.einsum("lij,lj->li", Binv, ra)      # C_aa^-1 r_a por nivel
            corr = (self.M @ v) @ Rinv                 # (nr, t)
            r = r - corr.T.ravel()
            extra_sr = float(np.sum(ra * v))
            datos_abs = (Binv, ra, Rinv)
        return C, r, Rinv, extra_logdet, extra_sr, datos_abs

    def _factorizar(self, C):
        if self.densa:
            Cd = C.toarray()
            try:
                L = np.linalg.cholesky(Cd)
                logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
                return logdet, ("chol", L)
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError("C no es definida positiva")
        lu = splu(C.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                  options={"SymmetricMode": True})
        diag = lu.U.diagonal()
        if np.any(diag <= 0):
            raise np.linalg.LinAlgError("C no es definida positiva")
        return float(np.sum(np.log(diag))), ("lu", lu)

    @staticmethod
    def _resolver(fact, r):
        tipo, f = fact
        if tipo == "chol":
            return sla.cho_solve((f, True), r)
        return f.solve(r)

    def m2logl(self, R0, Gs):
        """-2 log L restringida (escala estandarizada, sin constante)."""
        self.evaluaciones += 1
        C, r, Rinv, ld_extra, sr_extra, _ = self._sistema(R0, Gs)
        logdet_c, fact = self._factorizar(C)
        sol = self._resolver(fact, r)
        yry = float(np.sum(Rinv * self.YY))
        ypy = yry - float(sol @ r) - sr_extra
        _, ld_r = np.linalg.slogdet(R0)
        valor = self.n * ld_r + logdet_c + ld_extra + ypy
        for k, a in enumerate(self.aleatorios):
            _, ldg = np.linalg.slogdet(Gs[k])
            # log|G0 ⊗ A| = q·log|G0| + t·log|A|: el término de A no cambia las estimaciones, pero sin él
            # la prueba de razón de verosimilitud contra el modelo sin el efecto queda sesgada
            valor += a.q * ldg + self.t * a.logdet_a
        return valor

    def soluciones(self, R0, Gs, varianza_error=False):
        """Soluciones BLUE/BLUP en escala original.

        Devuelve dict con 'fijos' (p x t), 'aleatorios' {k: q x t} y, si se pide,
        las varianzas del error de predicción (PEV) de los efectos no absorbidos.
        """
        C, r, Rinv, _, _, datos_abs = self._sistema(R0, Gs)
        _, fact = self._factorizar(C)
        sol = self._resolver(fact, r)
        t, nr = self.t, self.dim_r
        S = sol.reshape(t, nr).T                     # (nr, t)
        escala = self.de[None, :]
        res = {"fijos": S[:self.p] * escala, "aleatorios": {}, "pev": {}}
        for k in self.resto:
            a = self.aleatorios[k]
            res["aleatorios"][k] = S[self.inicio[k]:self.inicio[k] + a.q] * escala
        if self.abs is not None:
            Binv, ra, _ = datos_abs
            # u_a = C_aa^-1 (r_a - C_ar s_r)
            Cs = (self.M.T @ S) @ Rinv                # (qa, t)
            ua = np.einsum("lij,lj->li", Binv, ra - Cs)
            res["aleatorios"][self.abs] = ua * escala
        res["fijos"][0] += self.media  # la media general vuelve a la escala original
        if varianza_error:
            for k in self.resto:
                a = self.aleatorios[k]
                idx = [self.inicio[k] + i * nr for i in range(t)]
                pev = np.zeros((a.q, t))
                for j, base in enumerate(idx):
                    pev[:, j] = _diagonal_inversa(fact, nr * t, base, a.q) * self.de[j] ** 2
                res["pev"][k] = pev
        return res


def _diagonal_inversa(fact, dim, inicio, cantidad, lote=256):
    """Elementos diagonales de C^-1 en un rango de índices (resolviendo por lotes)."""
    salida = np.empty(cantidad)
    for a in range(0, cantidad, lote):
        b = min(cantidad, a + lote)
        E = np.zeros((dim, b - a))
        E[inicio + np.arange(a, b), np.arange(b - a)] = 1.0
        X = ModeloMixto._resolver(fact, E)
        salida[a:b] = X[inicio + np.arange(a, b), np.arange(b - a)]
    return salida


# ---------------------------------------------------------------- parámetros
def _a_cholesky(S):
    L = np.linalg.cholesky(S)
    t = S.shape[0]
    v = []
    for i in range(t):
        for j in range(i + 1):
            v.append(math.log(L[i, i]) if i == j else L[i, j])
    return v


def _de_cholesky(v, t):
    L = np.zeros((t, t))
    pos = 0
    for i in range(t):
        for j in range(i + 1):
            L[i, j] = math.exp(v[pos]) if i == j else v[pos]
            pos += 1
    return L @ L.T


def _tri(t):
    return t * (t + 1) // 2


def _a_vector(S):
    t = S.shape[0]
    return [S[i, j] for i in range(t) for j in range(i + 1)]


def _de_vector(v, t):
    S = np.zeros((t, t))
    pos = 0
    for i in range(t):
        for j in range(i + 1):
            S[i, j] = S[j, i] = v[pos]
            pos += 1
    return S


class ResultadoREML:
    pass


def ajustar(modelo, inicio=None, progreso=None, cancelado=None, max_eval=6000):
    """Estima por REML las matrices de covarianza.

    `inicio`: dict opcional {"R": t x t, "G": [t x t, ...]} en escala ORIGINAL.
    Devuelve ResultadoREML con matrices en escala original, -2logL, errores estándar
    (matriz de covarianza de los parámetros) y soluciones.
    """
    t = modelo.t
    K = len(modelo.aleatorios)
    de = modelo.de
    escala = np.outer(de, de)
    if inicio and len(inicio.get("G") or []) != K:
        inicio = None  # valores iniciales de otro modelo: se usan los genéricos
    if inicio:
        R0 = np.asarray(inicio["R"], dtype=float) / escala
        G0 = [np.asarray(g, dtype=float) / escala for g in inicio["G"]]
    else:
        R0 = np.eye(t) * 0.6
        G0 = [np.eye(t) * (0.4 / max(K, 1)) for _ in range(K)]
    # asegurar matrices definidas positivas de partida
    def _pd(S, minimo):
        w, V = np.linalg.eigh((S + S.T) / 2)
        w = np.maximum(w, minimo)
        return (V * w) @ V.T
    R0 = _pd(R0, 0.05)
    G0 = [_pd(g, 0.01) for g in G0]

    nt = _tri(t)
    theta0 = np.array(_a_cholesky(R0) + sum((_a_cholesky(g) for g in G0), []))
    mejor = {"valor": math.inf, "theta": theta0.copy()}
    inicio_t = time.time()

    def desempacar(theta):
        R = _de_cholesky(theta[:nt], t)
        Gs = [_de_cholesky(theta[nt * (k + 1):nt * (k + 2)], t) for k in range(K)]
        return R, Gs

    def objetivo(theta):
        if cancelado and cancelado():
            raise Cancelado()
        # límite inferior suave para las desviaciones (evita sobre/infra-flujo)
        diag_idx = [nt * blk + i * (i + 1) // 2 + i for blk in range(K + 1) for i in range(t)]
        if np.any(theta[diag_idx] < -11) or np.any(theta[diag_idx] > 6):
            return 1e12
        R, Gs = desempacar(theta)
        try:
            v = modelo.m2logl(R, Gs)
        except (np.linalg.LinAlgError, RuntimeError, ValueError):
            return 1e12
        if not np.isfinite(v):
            return 1e12
        if v < mejor["valor"]:
            mejor["valor"] = v
            mejor["theta"] = theta.copy()
        if progreso and modelo.evaluaciones % 10 == 0:
            progreso(None, f"REML · evaluación {modelo.evaluaciones} · −2 log L = {mejor['valor']:.4f}")
        return v

    m = len(theta0)
    # 1) Nelder–Mead (robusto ante superficies planas y límites). Una diferencia de 1e-6 en
    #    -2logL ya es despreciable para los componentes de varianza.
    opciones = {"xatol": 1e-5, "fatol": 1e-6, "maxfev": max_eval, "adaptive": m > 3}
    res = minimize(objetivo, theta0, method="Nelder-Mead", options=opciones)
    # 2) reinicio con un símplex pequeño alrededor del mejor punto: descarta una
    #    convergencia prematura sin volver a recorrer toda la superficie
    base = mejor["theta"].copy()
    simplex = np.vstack([base] + [base + 0.02 * np.eye(m)[i] for i in range(m)])
    res2 = minimize(objetivo, base, method="Nelder-Mead", options=dict(opciones, initial_simplex=simplex))
    # 3) pulido cuasi-Newton cuando hay pocos parámetros
    if m <= 6:
        try:
            minimize(objetivo, mejor["theta"], method="BFGS", options={"gtol": 1e-5, "maxiter": 40})
        except Cancelado:
            raise
        except Exception:
            pass
    R, Gs = desempacar(mejor["theta"])
    salida = ResultadoREML()
    salida.m2logl_std = mejor["valor"]
    # -2logL en escala original: + 2·(n - p)·Σ log(de) por el cambio de escala (jacobiano)
    salida.m2logl = mejor["valor"] + 2 * (modelo.n - modelo.p) * float(np.sum(np.log(de)))
    salida.R = R * escala
    salida.G = [g * escala for g in Gs]
    salida.convergio = bool(res.success or res2.success)
    salida.evaluaciones = modelo.evaluaciones
    salida.segundos = time.time() - inicio_t
    salida.t = t
    salida.std = (R, Gs)
    salida.limite = [bool(np.any(np.linalg.eigvalsh(g) < 1e-6 * np.trace(escala) / t)) for g in salida.G]
    if progreso:
        progreso(None, "Calculando errores estándar (matriz de información)…")
    salida.cov_param, salida.nombres_param = _covarianza_parametros(modelo, R, Gs, cancelado)
    return salida


def _covarianza_parametros(modelo, R, Gs, cancelado=None):
    """Inversa de la información observada (½ hessiano de -2logL) para los
    parámetros en escala original: vech(R0), vech(G0_1), ..."""
    t = modelo.t
    nt = _tri(t)
    de = modelo.de
    escala = np.outer(de, de)
    v0 = np.array(_a_vector(R) + sum((_a_vector(g) for g in Gs), []))
    esc = np.array(_a_vector(escala) * (len(Gs) + 1))

    def f(v):
        if cancelado and cancelado():
            raise Cancelado()
        Rv = _de_vector(v[:nt], t)
        Gv = [_de_vector(v[nt * (k + 1):nt * (k + 2)], t) for k in range(len(Gs))]
        try:
            for S in [Rv] + Gv:
                np.linalg.cholesky(S)
            return modelo.m2logl(Rv, Gv)
        except np.linalg.LinAlgError:
            return math.nan

    m = len(v0)
    pasos = np.zeros(m)
    for i in range(m):
        # paso relativo a la escala de la varianza involucrada
        pasos[i] = 1e-3 * max(abs(v0[i]), 0.02)
    H = np.full((m, m), math.nan)
    f0 = f(v0)
    for i in range(m):
        for j in range(i, m):
            ei = np.zeros(m)
            ej = np.zeros(m)
            ei[i] = pasos[i]
            ej[j] = pasos[j]
            if i == j:
                fp, fm = f(v0 + ei), f(v0 - ei)
                H[i, i] = (fp - 2 * f0 + fm) / pasos[i] ** 2
            else:
                fpp, fpm = f(v0 + ei + ej), f(v0 + ei - ej)
                fmp, fmm = f(v0 - ei + ej), f(v0 - ei - ej)
                H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * pasos[i] * pasos[j])
    nombres = []
    etiquetas = ["e"] + [a.nombre for a in modelo.aleatorios]
    for et in etiquetas:
        for i in range(t):
            for j in range(i + 1):
                nombres.append((et, i, j))
    if not np.all(np.isfinite(H)):
        return None, nombres
    try:
        cov_std = 2.0 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None, nombres
    if np.any(np.diag(cov_std) < 0):
        return None, nombres
    cov = cov_std * np.outer(esc, esc)
    return cov, nombres


def delta(funcion, valores, cov):
    """Valor y error estándar de funcion(valores) por el método delta."""
    valores = np.asarray(valores, dtype=float)
    val = funcion(valores)
    if cov is None:
        return val, math.nan
    grad = np.zeros_like(valores)
    for i in range(len(valores)):
        h = max(abs(valores[i]) * 1e-5, 1e-10)
        e = np.zeros_like(valores)
        e[i] = h
        grad[i] = (funcion(valores + e) - funcion(valores - e)) / (2 * h)
    var = float(grad @ cov @ grad)
    return val, (math.sqrt(var) if var >= 0 else math.nan)


def vector_parametros(res):
    return np.array(_a_vector(res.R) + sum((_a_vector(g) for g in res.G), []))


def matrices_de_vector(v, t, K):
    nt = _tri(t)
    R = _de_vector(v[:nt], t)
    G = [_de_vector(v[nt * (k + 1):nt * (k + 2)], t) for k in range(K)]
    return R, G


def verificar_tamano(modelo):
    if modelo.n < 10:
        raise ErrorDatos("Muy pocos registros para REML (mínimo 10).")
