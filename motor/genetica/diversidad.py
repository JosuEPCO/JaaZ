"""Diversidad genética a partir del pedigrí: los métodos de ENDOG v4.8
(Gutiérrez y Goyache, 2005) implementados sobre matrices dispersas.

Todas las operaciones con la matriz de parentesco usan A = (I − ½P)⁻¹ D (I − ½P')⁻¹
(Quaas, 1976) y resoluciones triangulares, sin formar A completa:
  · AR (relación promedio): c' = (1/n)·1'A
  · contribuciones génicas de fundadores y ancestros: (I − ½P')⁻¹ w
  · coascendencias medias entre grupos: G'AG / 2

Validado contra los resultados publicados en el manual de ENDOG para su archivo
de ejemplo (ver tests/test_diversidad.py).
"""
import math
import re
from collections import Counter, defaultdict
from datetime import date, timedelta

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular

from .datos import Cancelado, ErrorDatos


# ================================================================== fechas
_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})")
_DMY = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})")
_ANIO = re.compile(r"^(1[89]\d{2}|2[01]\d{2})$")


def anio_decimal(v):
    """Fecha → años decimales (días/365,25). Acepta fechas ISO, dd/mm/aaaa, número de
    serie de Excel o solo el año (se toma el 1 de julio)."""
    if v is None or isinstance(v, bool):
        return math.nan
    try:
        if isinstance(v, (int, float)):
            x = float(v)
            if math.isnan(x):
                return math.nan
            if 1800 <= x <= 2200:
                return date(int(x), 7, 1).toordinal() / 365.25
            if 2000 <= x <= 80000:  # número de serie de Excel (1905–2119)
                return (date(1899, 12, 30) + timedelta(days=x)).toordinal() / 365.25
            return math.nan
        s = str(v).strip()
        m = _ISO.match(s)
        if m:
            return date(int(m[1]), int(m[2]), int(m[3])).toordinal() / 365.25
        m = _DMY.match(s)
        if m:
            return date(int(m[3]), int(m[2]), int(m[1])).toordinal() / 365.25
        if _ANIO.match(s):
            return date(int(s), 7, 1).toordinal() / 365.25
    except (ValueError, OverflowError):
        return math.nan
    return math.nan


def anio_de(x):
    """Años decimales → año calendario."""
    return date.fromordinal(int(round(x * 365.25))).year if x == x else None


def es_referencia(v):
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return float(v) == 1.0
    return str(v).strip().lower() in ("1", "1.0", "si", "sí", "x", "s", "true", "verdadero", "ref")


# ================================================================== núcleo
class Diversidad:
    """Herramientas de diversidad sobre un Pedigri ya ordenado (índices 0..n-1)."""

    def __init__(self, ped, cancelado=None, progreso=None):
        self.ped = ped
        self.n = ped.n
        self.s = ped.s[1:].astype(np.int64) - 1     # -1 = desconocido
        self.d = ped.d[1:].astype(np.int64) - 1
        self.F = ped.F()
        self.Dm = ped.D()[1:]
        self.ecg, self.gmax, self.gcom = ped.generaciones_equivalentes()
        self._cancelado = cancelado
        self._progreso = progreso
        self._M = self.matriz_m()
        self._Mt = self._M.T.tocsr()

    def _chequear(self):
        if self._cancelado and self._cancelado():
            raise Cancelado()

    def _avance(self, f, msg):
        if self._progreso:
            self._progreso(f, msg)

    # ------------------------------------------------------------ álgebra
    def matriz_m(self, s=None, d=None):
        """M = I − ½P (triangular inferior)."""
        s = self.s if s is None else s
        d = self.d if d is None else d
        n = self.n
        i = np.arange(n)
        filas, cols, vals = [i], [i], [np.ones(n)]
        for p in (s, d):
            m = p >= 0
            filas.append(i[m])
            cols.append(p[m])
            vals.append(np.full(int(m.sum()), -0.5))
        return sp.coo_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))), shape=(n, n)).tocsr()

    def resolver_m(self, B, M=None):
        """M X = B  (genes que cada animal recibe de las columnas de B)."""
        return spsolve_triangular(self._M if M is None else M, B, lower=True)

    def resolver_mt(self, B, Mt=None):
        """M' X = B  (contribuciones hacia atrás, de la progenie a los ancestros)."""
        return spsolve_triangular(self._Mt if Mt is None else Mt, B, lower=False)

    def a_por(self, B):
        """A·B sin formar A."""
        W = self.resolver_mt(B)
        W = W * (self.Dm[:, None] if W.ndim == 2 else self.Dm)
        return self.resolver_m(W)

    # ------------------------------------------------------------ por individuo
    def ar(self):
        """Relación promedio (AR): (1/n)·A·1 (Gutiérrez et al., 2003)."""
        return self.a_por(np.ones(self.n)) / self.n

    def progenie(self):
        c = np.bincount(self.s[self.s >= 0], minlength=self.n) + np.bincount(self.d[self.d >= 0], minlength=self.n)
        auto = (self.s >= 0) & (self.s == self.d)
        c -= np.bincount(self.s[auto], minlength=self.n)  # autofecundación: contar una vez
        return c

    def delta_f(self, version=2009):
        """Incremento individual de endogamia ΔFi = 1 − (1 − Fi)^(1/t).
        2009 (ENDOG 4.8, Gutiérrez et al. 2009): t = GE − 1;  2008: t = GE.
        NaN donde no está definido (t ≤ 0)."""
        t = self.ecg - 1 if version == 2009 else self.ecg
        out = np.full(self.n, np.nan)
        ok = t > 0
        out[ok] = 1 - (1 - self.F[ok]) ** (1 / t[ok])
        return out

    # ------------------------------------------------------------ tamaño efectivo
    @staticmethod
    def ne_desde_media(valores, por_pares=False):
        """Ne = 1/(2·media) con EE = 2·Ne²·DE/√n (Gutiérrez et al., 2008)."""
        v = np.asarray(valores, dtype=float)
        v = v[~np.isnan(v)]
        if len(v) < 2 or v.mean() <= 0:
            return {"ne": math.nan, "ee": math.nan, "n": len(v), "media": float(v.mean()) if len(v) else math.nan}
        ne = 1 / (2 * v.mean())
        return {"ne": ne, "ee": 2 * ne ** 2 * v.std(ddof=1) / math.sqrt(len(v)), "n": len(v), "media": float(v.mean())}

    def ne_por_generacion(self, gen):
        """Tabla por generación (PorG/PorC de ENDOG): N, F, % consanguíneos, F de
        consanguíneos, AR y Ne = 1/(2ΔF) con ΔF = (Ft − Ft−1)/(1 − Ft−1)."""
        ar = self._ar_cache()
        filas = []
        previo = None
        for g in range(int(np.nanmax(gen)) + 1 if len(gen) else 0):
            m = gen == g
            if not m.any():
                continue
            Ft = float(self.F[m].mean())
            consang = self.F[m] > 1e-12
            fila = {"generacion": g, "n": int(m.sum()), "f": Ft, "pct_consanguineos": 100 * float(consang.mean()),
                    "f_consanguineos": float(self.F[m][consang].mean()) if consang.any() else 0.0,
                    "ar": float(ar[m].mean()), "delta_f": None, "ne": None}
            if previo is not None and Ft > previo:
                dF = (Ft - previo) / (1 - previo)
                fila["delta_f"] = dF
                fila["ne"] = 1 / (2 * dF)
            previo = Ft
            filas.append(fila)
        return filas

    def _ar_cache(self):
        if not hasattr(self, "_ar"):
            self._ar = self.ar()
        return self._ar

    def ne_regresion_generaciones(self):
        """Ne a partir de la pendiente de F sobre generaciones (máximas, completas y
        equivalentes) en todo el pedigrí: ΔF ≈ b, Ne = 1/(2b)."""
        out = {}
        for clave, g in (("maximas", self.gmax), ("completas", self.gcom), ("equivalentes", self.ecg)):
            if np.ptp(g) <= 0:
                out[clave] = {"b": math.nan, "ne": math.nan, "media": float(g.mean())}
                continue
            b = float(np.polyfit(g, self.F, 1)[0])
            out[clave] = {"b": b, "ne": 1 / (2 * b) if b > 0 else math.nan, "media": float(g.mean())}
        return out

    def ne_regresion_ref(self, ref):
        """ENDOG: ΔF = b/(1 − (F̄ − b)) con b la pendiente de F sobre GE en la referencia."""
        idx = np.flatnonzero(ref)
        if len(idx) < 3 or np.ptp(self.ecg[idx]) <= 0:
            return {"b": math.nan, "ne": math.nan}
        b = float(np.polyfit(self.ecg[idx], self.F[idx], 1)[0])
        Ft = float(self.F[idx].mean())
        dF = b / (1 - (Ft - b))
        return {"b": b, "delta_f": dF, "ne": 1 / (2 * dF) if dF > 0 else math.nan}

    def coascendencias(self, idx, lote=256):
        """Matriz de coascendencia (A/2) entre los animales `idx`."""
        idx = np.asarray(idx)
        k = len(idx)
        C = np.empty((k, k))
        for a in range(0, k, lote):
            self._chequear()
            b = min(k, a + lote)
            E = np.zeros((self.n, b - a))
            E[idx[a:b], np.arange(b - a)] = 1.0
            C[:, a:b] = self.a_por(E)[idx, :] / 2
        return C

    def ne_coascendencia(self, ref, max_animales=3000):
        """Ne por incremento de coascendencia entre pares (Cervantes et al., 2010):
        Δc_jk = 1 − (1 − c_jk)^(2/(g_j + g_k)), Ne = 1/(2·Δc̄)."""
        idx = np.flatnonzero(ref)
        muestreo = False
        if len(idx) > max_animales:
            idx = np.sort(np.random.default_rng(11).choice(idx, max_animales, replace=False))
            muestreo = True
        C = self.coascendencias(idx)
        g = self.ecg[idx]
        iu = np.triu_indices(len(idx), 1)
        c = C[iu]
        t = (g[iu[0]] + g[iu[1]]) / 2
        ok = t > 0
        dc = 1 - (1 - c[ok]) ** (1 / t[ok])
        if len(dc) < 2 or dc.mean() <= 0:
            return {"ne": math.nan, "ee": math.nan, "pares": int(ok.sum()), "muestreo": muestreo, "n": len(idx)}
        ne = 1 / (2 * dc.mean())
        return {"ne": ne, "ee": 2 * ne ** 2 * dc.std(ddof=1) / math.sqrt(len(dc)), "pares": len(dc),
                "muestreo": muestreo, "n": len(idx), "media": float(dc.mean())}

    # ------------------------------------------------------------ endogamia reciente
    def endogamia_reciente(self, k, idx):
        """F calculada con solo k generaciones de pedigrí (tablas Inbreed_k de ENDOG)."""
        s, d = self.s, self.d
        out = np.zeros(len(idx))
        for pos, a in enumerate(idx):
            if pos % 500 == 0:
                self._chequear()
            dist = {a: 0}
            frente = [a]
            for g in range(1, k + 1):
                sig = []
                for x in frente:
                    for p in (s[x], d[x]):
                        if p >= 0 and p not in dist:
                            dist[p] = g
                            sig.append(p)
                frente = sig
            if s[a] < 0 or d[a] < 0:
                continue
            nodos = sorted(dist)
            loc = {x: i for i, x in enumerate(nodos)}
            m = len(nodos)
            A = np.zeros((m, m))
            for i, x in enumerate(nodos):
                ps = [loc[p] for p in (s[x], d[x]) if p >= 0 and dist[x] < k and p in loc]
                if i:
                    if len(ps) == 2:
                        A[i, :i] = 0.5 * (A[ps[0], :i] + A[ps[1], :i])
                    elif len(ps) == 1:
                        A[i, :i] = 0.5 * A[ps[0], :i]
                    A[:i, i] = A[i, :i]
                A[i, i] = 1 + (0.5 * A[ps[0], ps[1]] if len(ps) == 2 else 0.0)
            out[pos] = A[loc[a], loc[a]] - 1
        return out

    # ------------------------------------------------------------ fundadores y ancestros
    def tipos_base(self):
        completo = (self.s < 0) & (self.d < 0)
        medio = (self.s < 0) ^ (self.d < 0)
        return completo, medio

    def contribuciones(self, ref, s=None, d=None):
        """Contribución génica esperada de cada animal a la población de referencia."""
        w = ref.astype(float) / ref.sum()
        if s is None:
            return self.resolver_mt(w)
        return spsolve_triangular(self.matriz_m(s, d).T.tocsr(), w, lower=False)

    def fundadores(self, ref):
        """Contribuciones de fundadores (los medio-fundadores aportan un fantasma)."""
        c = self.contribuciones(ref)
        completo, medio = self.tipos_base()
        filas = []
        for i in np.flatnonzero(completo | medio):
            q = c[i] if completo[i] else 0.5 * c[i]
            filas.append({"i": int(i), "fantasma": bool(medio[i]), "q": float(q)})
        q = np.array([f["q"] for f in filas]) if filas else np.zeros(0)
        fe = 1 / float(np.sum(q ** 2)) if q.size and np.sum(q ** 2) > 0 else math.nan
        filas.sort(key=lambda f: -f["q"])
        return {"filas": filas, "fe": fe, "base": int(completo.sum() + medio.sum()),
                "base_real": float(completo.sum() + 0.5 * medio.sum()),
                "contribuyentes": int(np.sum(q > 1e-12))}

    def ancestros(self, ref, max_ancestros=2000, tolerancia=1e-9):
        """Contribuciones marginales de ancestros (Boichard et al., 1997)."""
        n = self.n
        w = ref.astype(float) / ref.sum()
        s2, d2 = self.s.copy(), self.d.copy()
        elegidos, marginales, brutas = [], [], []
        indicador = np.zeros(n)
        acumulado = 0.0
        for it in range(max_ancestros):
            self._chequear()
            if it % 10 == 0:
                self._avance(None, f"Ancestros (Boichard): {it} seleccionados · {100 * acumulado:.1f} % explicado")
            M = self.matriz_m(s2, d2)
            c = spsolve_triangular(M.T.tocsr(), w, lower=False)
            r = spsolve_triangular(M, indicador, lower=True) if elegidos else np.zeros(n)
            marg = c * (1 - r)
            if elegidos:
                marg[elegidos] = -1.0
            k = int(np.argmax(marg))
            if marg[k] <= tolerancia:
                break
            elegidos.append(k)
            marginales.append(float(marg[k]))
            brutas.append(float(c[k]))
            acumulado += marg[k]
            indicador[k] = 1.0
            s2[k] = -1
            d2[k] = -1
            if acumulado >= 1 - 1e-9:
                break
        p = np.array(marginales)
        fa = 1 / float(np.sum(p ** 2)) if p.size else math.nan
        acum = np.cumsum(p)
        return {"elegidos": elegidos, "marginal": marginales, "bruta": brutas, "fa": fa,
                "explican_50": int(np.searchsorted(acum, 0.5) + 1) if p.size and acum[-1] >= 0.5 else None,
                "acumulado": float(acum[-1]) if p.size else 0.0}

    def coascendencia_media(self, ref):
        w = ref.astype(float) / ref.sum()
        return float(w @ self.a_por(w)) / 2

    def gci(self, max_celdas=40_000_000):
        """Índice de conservación genética (Alderson, 1992): 1/Σ p_f², p_f proporción de
        genes de cada fundador (y fantasma) en el animal."""
        completo, medio = self.tipos_base()
        cols = np.flatnonzero(completo | medio)
        if self.n * len(cols) > max_celdas:
            return None
        suma = np.zeros(self.n)
        for a in range(0, len(cols), 512):
            self._chequear()
            b = cols[a:a + 512]
            E = np.zeros((self.n, len(b)))
            E[b, np.arange(len(b))] = 1.0
            T = self.resolver_m(E)                     # proporción de genes que pasan por cada base
            # un fundador completo es él mismo la fuente; en un medio-fundador la fuente es el
            # progenitor desconocido (fantasma), que aporta la mitad de lo que pasa por él
            esc = np.where(completo[b], 1.0, 0.5)
            suma += np.sum((T * esc) ** 2, axis=1)
        with np.errstate(divide="ignore"):
            return np.where(suma > 0, 1 / suma, np.nan)

    def endogamia_parcial(self, focos, ref, cortar=False, max_animales=3500):
        """Coeficientes de endogamia parcial de Lacy et al. (1996) por fundador o ancestro.

        Devuelve (índices de la referencia, matriz ref × focos). Con `cortar` el foco se
        trata como fundador (se eliminan sus padres): coeficientes F_a de ENDOG."""
        relevantes = np.zeros(self.n, bool)
        pila = list(np.flatnonzero(ref))
        while pila:
            x = pila.pop()
            if relevantes[x]:
                continue
            relevantes[x] = True
            for p in (self.s[x], self.d[x]):
                if p >= 0 and not relevantes[p]:
                    pila.append(p)
        idx = np.flatnonzero(relevantes)
        if len(idx) > max_animales:
            raise ErrorDatos(f"La endogamia parcial necesita la matriz completa de {len(idx)} animales (referencia + "
                             f"ancestros); el límite es {max_animales}. Elija una población de referencia más pequeña.")
        loc = -np.ones(self.n, dtype=np.int64)
        loc[idx] = np.arange(len(idx))
        ref_loc = loc[np.flatnonzero(ref)]
        res = np.zeros((len(ref_loc), len(focos)))
        m = len(idx)
        for j, f in enumerate(focos):
            self._chequear()
            s, d = self.s.copy(), self.d.copy()
            if cortar:
                s[f] = d[f] = -1
            E = np.zeros(self.n)
            E[f] = 1.0
            pi = spsolve_triangular(self.matriz_m(s, d), E, lower=True) if cortar else self.resolver_m(E)
            G = np.zeros((m, m))
            Fp = np.zeros(m)
            for i_loc, x in enumerate(idx):
                ps, pd = s[x], d[x]
                a = loc[ps] if ps >= 0 else -1
                b = loc[pd] if pd >= 0 else -1
                if i_loc:
                    fila = np.zeros(i_loc)
                    if a >= 0:
                        fila += 0.5 * G[a, :i_loc]
                    if b >= 0:
                        fila += 0.5 * G[b, :i_loc]
                    G[i_loc, :i_loc] = fila
                    G[:i_loc, i_loc] = fila
                Fp[i_loc] = G[a, b] if (a >= 0 and b >= 0) else 0.0
                G[i_loc, i_loc] = 0.5 * pi[x] + 0.5 * Fp[i_loc]
            res[:, j] = Fp[ref_loc]
        return np.flatnonzero(ref), res

    # ------------------------------------------------------------ calidad del pedigrí
    def completitud(self, generaciones=5):
        """Proporción de ancestros conocidos por generación y línea, e índice de
        MacCluer et al. (1983) por animal: PEC = 4·Cp·Cm/(Cp + Cm)."""
        n = self.n
        s_ext = np.append(self.s, -1)
        d_ext = np.append(self.d, -1)
        s_ext[s_ext < 0] = n
        d_ext[d_ext < 0] = n
        s_ext[n] = n
        d_ext[n] = n
        pat = s_ext[:n].reshape(n, 1)
        mat = d_ext[:n].reshape(n, 1)
        conocidos_p, conocidos_m, posiciones = [], [], []
        for g in range(1, generaciones + 1):
            # cada línea aporta como máximo la mitad de los 2^g ancestros de la generación g
            conocidos_p.append((pat < n).sum(axis=1) / 2 ** g)
            conocidos_m.append((mat < n).sum(axis=1) / 2 ** g)
            if g <= 3:
                posiciones.append(np.hstack([pat, mat]) < n)
            pat = np.hstack([s_ext[pat], d_ext[pat]]) if g < generaciones else pat
            mat = np.hstack([s_ext[mat], d_ext[mat]]) if g < generaciones else mat
        cp = np.mean(conocidos_p, axis=0)
        cm = np.mean(conocidos_m, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            pec = np.where(cp + cm > 0, 4 * cp * cm / (cp + cm), 0.0)
        por_gen = [np.asarray(conocidos_p[g]) + np.asarray(conocidos_m[g]) for g in range(generaciones)]
        return {"pec": pec, "por_generacion": por_gen, "posiciones": posiciones}

    def apareamientos_cercanos(self):
        """Apareamientos entre parientes cercanos registrados (HighInbred de ENDOG)."""
        s, d = self.s, self.d
        hc = mh = pp = 0
        ejemplos = defaultdict(list)
        for i in range(self.n):
            a, b = s[i], d[i]
            if a < 0 or b < 0:
                continue
            mismos_p = s[a] >= 0 and s[a] == s[b]
            mismas_m = d[a] >= 0 and d[a] == d[b]
            if mismos_p and mismas_m:
                hc += 1
                ejemplos["hc"].append(i)
            elif mismos_p or mismas_m:
                mh += 1
                ejemplos["mh"].append(i)
            if s[b] == a or d[a] == b:
                pp += 1
                ejemplos["pp"].append(i)
        return {"hermanos_completos": hc, "medios_hermanos": mh, "padre_hijo": pp, "ejemplos": ejemplos}


# ================================================================== intervalo generacional
def intervalos(div, nac, sexo, solo_reproductores, mascara=None):
    """Edad de los padres al nacimiento de su progenie por las 4 vías.
    `solo_reproductores`: solo la progenie que a su vez es progenitor (intervalo generacional)."""
    s, d, n = div.s, div.d, div.n
    es_padre = np.zeros(n, bool)
    es_padre[s[s >= 0]] = True
    es_padre[d[d >= 0]] = True
    vias = {k: [] for k in ("padre-hijo", "padre-hija", "madre-hijo", "madre-hija")}
    otras = []
    negativas = 0
    for i in range(n):
        if mascara is not None and not mascara[i]:
            continue
        if solo_reproductores and not es_padre[i]:
            continue
        if nac[i] != nac[i]:
            continue
        sx = sexo[i]
        for p, via in ((s[i], "padre"), (d[i], "madre")):
            if p < 0 or nac[p] != nac[p]:
                continue
            edad = nac[i] - nac[p]
            if edad < 0:          # progenitor nacido después que la cría: error de fecha
                negativas += 1
                continue
            if sx == "M":
                vias[f"{via}-hijo"].append(edad)
            elif sx == "H":
                vias[f"{via}-hija"].append(edad)
            else:
                otras.append(edad)
    filas = []
    todas = []
    for via, v in vias.items():
        v = np.array(v)
        todas.append(v)
        filas.append(_resumen_edades(via, v))
    if otras:
        filas.append(_resumen_edades("sexo de la cría desconocido", np.array(otras)))
        todas.append(np.array(otras))
    total = np.concatenate(todas) if todas else np.zeros(0)
    filas.append(_resumen_edades("total", total))
    filas[-1]["negativas"] = negativas
    return filas, (float(total.mean()) if total.size else math.nan)


def _resumen_edades(via, v):
    n = len(v)
    de = float(v.std(ddof=1)) if n > 1 else math.nan
    return {"via": via, "n": n, "media": float(v.mean()) if n else math.nan, "de": de,
            "ee": de / math.sqrt(n) if n > 1 else math.nan}


def ne_fecha(div, nac, L, mascara):
    """Ne por regresión de F (y de ln(1 − F)) sobre la fecha de nacimiento (ENDOG):
    lineal: ΔF = L·b/(1 − (F̄ − L·b));  logarítmica (Pérez-Enciso, 1995): Ne = −1/(2·L·b')."""
    idx = np.flatnonzero(mascara & ~np.isnan(nac))
    out = {"n": len(idx), "ne": math.nan, "ne_log": math.nan}
    if len(idx) < 3 or not (L == L and L > 0) or np.ptp(nac[idx]) <= 0:
        return out
    F = div.F[idx]
    b = float(np.polyfit(nac[idx], F, 1)[0])
    Ft = float(F.mean())
    dF = L * b / (1 - (Ft - L * b))
    bl = float(np.polyfit(nac[idx], np.log(1 - F), 1)[0])
    out.update({"b": b, "delta_f": dF, "ne": 1 / (2 * dF) if dF > 0 else math.nan,
                "b_log": bl, "ne_log": -1 / (2 * L * bl) if bl < 0 else math.nan})
    return out


def ne_hill(div, sexo, conjunto):
    """Ne por varianza del tamaño de familia (Hill, 1979), suponiendo que el conjunto
    corresponde a una generación. El tamaño de familia es el número de hijos e hijas
    que a su vez son reproductores (James, 1977)."""
    s, d, n = div.s, div.d, div.n
    es_padre = np.zeros(n, bool)
    es_padre[s[s >= 0]] = True
    es_padre[d[d >= 0]] = True
    hijos = np.zeros(n)
    hijas = np.zeros(n)
    for i in np.flatnonzero(es_padre):
        for p in {s[i], d[i]}:
            if p >= 0:
                if sexo[i] == "M":
                    hijos[p] += 1
                elif sexo[i] == "H":
                    hijas[p] += 1
    machos = np.flatnonzero(conjunto & es_padre & (sexo == "M"))
    hembras = np.flatnonzero(conjunto & es_padre & (sexo == "H"))
    M, Fh = len(machos), len(hembras)
    if M < 2 or Fh < 2:
        return {"ne": math.nan, "machos": M, "hembras": Fh}
    vmm, vmf = np.var(hijos[machos], ddof=1), np.var(hijas[machos], ddof=1)
    cm = float(np.cov(hijos[machos], hijas[machos])[0, 1])
    vfm, vff = np.var(hijos[hembras], ddof=1), np.var(hijas[hembras], ddof=1)
    cf = float(np.cov(hijos[hembras], hijas[hembras])[0, 1])
    t_m = (2 + vmm + 2 * (M / Fh) * cm + (M / Fh) ** 2 * vmf) / (16 * M)
    t_f = (2 + (Fh / M) ** 2 * vfm + 2 * (Fh / M) * cf + vff) / (16 * Fh)
    return {"ne": 1 / (t_m + t_f), "machos": M, "hembras": Fh, "var_mm": vmm, "var_mf": vmf, "cov_m": cm,
            "var_fm": vfm, "var_ff": vff, "cov_f": cf,
            "media_hijos_machos": float(hijos[machos].mean() + hijas[machos].mean()),
            "media_hijos_hembras": float(hijos[hembras].mean() + hijas[hembras].mean())}


# ================================================================== subpoblaciones
def estadisticos_f(div, grupos, tamano_comun=None):
    """Estadísticos F de Wright y distancias desde la genealogía (Caballero y Toro, 2000,
    2002), con ajuste opcional de la coascendencia por tamaño (Bartolomé et al., 2010)."""
    etiquetas = sorted({g for g in grupos if g is not None}, key=lambda x: (str(type(x)), x))
    k = len(etiquetas)
    if k < 2:
        raise ErrorDatos("Se necesitan al menos 2 subpoblaciones.")
    if k > 80:
        raise ErrorDatos(f"Hay {k} subpoblaciones; el máximo es 80.")
    pos = {e: i for i, e in enumerate(etiquetas)}
    G = np.zeros((div.n, k))
    for i, g in enumerate(grupos):
        if g is not None:
            G[i, pos[g]] = 1.0
    N = G.sum(axis=0)
    S = G.T @ div.a_por(G) / 2            # sumas de coascendencias entre grupos
    fij = S / np.outer(N, N)
    Fi = np.array([div.F[G[:, j] > 0].mean() for j in range(k)])
    si = (1 + Fi) / 2
    ge = np.array([div.ecg[G[:, j] > 0].mean() for j in range(k)])
    if tamano_comun:
        Mc = float(tamano_comun)
        for j in range(k):
            if N[j] > 1:
                f_fuera = (N[j] * fij[j, j] - si[j]) / (N[j] - 1)
                fij[j, j] = si[j] / Mc + (1 - 1 / Mc) * f_fuera
        Nw = np.full(k, Mc)
    else:
        Nw = N
    return _resumen_f(etiquetas, Nw, N, fij, Fi, si, ge)


def _resumen_f(etiquetas, Nw, N, fij, Fi, si, ge):
    k = len(etiquetas)
    NT = Nw.sum()
    pesos = np.outer(Nw, Nw) / NT ** 2
    f_dentro = float(np.sum(Nw * np.diag(fij)) / NT)
    f_meta = float(np.sum(pesos * fij))
    F_meta = float(np.sum(Nw * Fi) / NT)
    s_meta = float(np.sum(Nw * si) / NT)
    diag = np.diag(fij)
    Dnei = (diag[:, None] + diag[None, :]) / 2 - fij
    Dk = (si[:, None] + si[None, :]) / 2 - fij
    fst_par = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                fst_par[i, i] = (Fi[i] - fij[i, i]) / (1 - fij[i, i])   # Fis de la subpoblación
                continue
            ni, nj = Nw[i], Nw[j]
            fw = (ni * fij[i, i] + nj * fij[j, j]) / (ni + nj)
            ft = (ni * ni * fij[i, i] + nj * nj * fij[j, j] + 2 * ni * nj * fij[i, j]) / (ni + nj) ** 2
            fst_par[i, j] = (fw - ft) / (1 - ft)
    return {
        "etiquetas": etiquetas, "N": N, "fij": fij, "F": Fi, "s": si, "ge": ge,
        "f_dentro": f_dentro, "f_meta": f_meta, "F_meta": F_meta, "s_meta": s_meta, "d_nei": f_dentro - f_meta,
        "fis": (F_meta - f_dentro) / (1 - f_dentro), "fst": (f_dentro - f_meta) / (1 - f_meta),
        "fit": (F_meta - f_meta) / (1 - f_meta), "Dnei": Dnei, "Dk": Dk, "fst_par": fst_par,
        "fg": 1 / (2 * np.diag(fij)), "Nw": Nw,
    }


def contribucion_diversidad(res):
    """Pérdida o ganancia de diversidad génica (GD = 1 − f̃) al quitar cada subpoblación,
    separada en diversidad interna y distancia entre subpoblaciones (Caballero y Toro, 2002)."""
    fij, Nw = res["fij"], res["Nw"]
    k = len(Nw)

    def componentes(sel):
        n = Nw[sel]
        f = fij[np.ix_(sel, sel)]
        NT = n.sum()
        f_dentro = float(np.sum(n * np.diag(f)) / NT)
        f_meta = float(np.sum(np.outer(n, n) * f) / NT ** 2)
        return 1 - f_meta, 1 - f_dentro, f_dentro - f_meta

    gd, gd_w, dist = componentes(np.arange(k))
    filas = []
    for i in range(k):
        sel = np.array([j for j in range(k) if j != i])
        g, w, dd = componentes(sel)
        filas.append({"subpoblacion": res["etiquetas"][i], "gd_sin": g,
                      "interna": 100 * (w - gd_w) / gd, "distancia": 100 * (dd - dist) / gd,
                      "total": 100 * (g - gd) / gd})
    return {"gd": gd, "gd_interna": gd_w, "distancia": dist, "filas": filas}


# ================================================================== rebaños
def estructura_rebanos(div, rebano):
    """Clasificación de rebaños (Vassallo et al., 1986) según el origen de los padres."""
    s = div.s
    datos = defaultdict(lambda: {"crias": 0, "padre_propio": 0, "padre_ajeno": 0, "como_padre": 0,
                                 "en_su_rebano": 0, "en_otro": 0})
    for i in range(div.n):
        h = rebano[i]
        if h is None:
            continue
        datos[h]["crias"] += 1
        p = s[i]
        if p >= 0 and rebano[p] is not None:
            hp = rebano[p]
            datos[hp]["como_padre"] += 1
            if hp == h:
                datos[h]["padre_propio"] += 1
                datos[hp]["en_su_rebano"] += 1
            else:
                datos[h]["padre_ajeno"] += 1
                datos[hp]["en_otro"] += 1
    filas = []
    for h, x in datos.items():
        usa_ajenos = x["padre_ajeno"] > 0
        usa_propios = x["padre_propio"] > 0
        vende = x["en_otro"] > 0
        if not (usa_ajenos or usa_propios):
            tipo = "Sin padres identificados"
        elif not usa_ajenos and vende:
            tipo = "Núcleo"
        elif usa_ajenos and vende:
            tipo = "Multiplicador"
        elif usa_ajenos and not vende:
            tipo = "Comercial"
        else:
            tipo = "Aislado"
        con_padre = x["padre_propio"] + x["padre_ajeno"]
        filas.append({"rebano": h, "tipo": tipo, **x,
                      "pct_propio": 100 * x["padre_propio"] / con_padre if con_padre else None,
                      "pct_ajeno": 100 * x["padre_ajeno"] / con_padre if con_padre else None,
                      "pct_dentro": 100 * x["en_su_rebano"] / x["como_padre"] if x["como_padre"] else None,
                      "pct_otro": 100 * x["en_otro"] / x["como_padre"] if x["como_padre"] else None,
                      "usa_ajenos": usa_ajenos, "usa_propios": usa_propios, "vende": vende})
    filas.sort(key=lambda f: -f["crias"])
    return filas


def robertson(div, rebano, mascara, generaciones=4):
    """Número real y efectivo de rebaños que aportan padres, abuelos y bisabuelos
    paternos (Robertson, 1953): 1/Σ p_h²."""
    s = div.s
    filas = []
    actual = np.where(mascara)[0]
    for g in range(1, generaciones + 1):
        actual = np.array([s[i] for i in actual if s[i] >= 0], dtype=np.int64)
        rebanos = [rebano[i] for i in actual if rebano[i] is not None]
        if not rebanos:
            break
        cuenta = Counter(rebanos)
        p = np.array(list(cuenta.values()), dtype=float) / len(rebanos)
        filas.append({"generacion": g, "via": "padre" + " del padre" * (g - 1), "registros": len(rebanos),
                      "rebanos": len(cuenta), "efectivos": 1 / float(np.sum(p ** 2))})
    return filas
