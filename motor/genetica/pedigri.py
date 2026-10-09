"""Pedigrí: depuración, ordenamiento, endogamia, parentesco y matriz A inversa.

- Los padres que no figuran como individuos se agregan como animales base
  (igual que Pedigree Viewer).
- El pedigrí se reordena para que los padres precedan a sus crías.
- Endogamia: algoritmo de Meuwissen y Luo (1992) Genet. Sel. Evol. 24:305.
- A inversa: reglas de Henderson (1976) con la corrección de Quaas (1976) por endogamia.
- Parentesco de un animal con todos: A = (I-P)^-1 D (I-P)^-T (Colleau, 2002).
"""
import heapq
import math
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular

from .datos import Cancelado, ErrorDatos


class Pedigri:
    """Pedigrí ordenado y codificado 1..n (0 = desconocido)."""

    def __init__(self, ids, padres, madres, sexos=None, progreso=None, cancelado=None):
        self.avisos = []          # lista de (tipo, mensaje, [ids])
        self._progreso = progreso
        self._cancelado = cancelado
        if not (len(ids) == len(padres) == len(madres)):
            raise ErrorDatos("Las columnas de ID, padre y madre deben tener la misma longitud.")

        filas = {}
        duplicados = []
        sin_id = 0
        conflictos = []
        for k, (i, s, d) in enumerate(zip(ids, padres, madres)):
            if i is None:
                sin_id += 1
                continue
            if i in filas:
                duplicados.append(i)
                previo = filas[i]
                if (previo[0], previo[1]) != (s, d):
                    conflictos.append(i)
                continue
            filas[i] = (s, d, k)
        if not filas:
            raise ErrorDatos("Ninguna fila tiene identificación del animal: revise la columna de ID.")
        if sin_id:
            self._aviso("info", f"{sin_id} filas sin identificación del animal se ignoraron en el pedigrí.", [])
        if duplicados:
            unicos = sorted(set(duplicados), key=str)
            self._aviso("info", f"{len(unicos)} animales aparecen en varias filas (registros repetidos); "
                                "se usa la primera aparición para el pedigrí.", unicos)
        if conflictos:
            self._aviso("error", f"{len(set(conflictos))} animales aparecen con padres distintos en filas distintas.",
                        sorted(set(conflictos), key=str))

        propios = []
        for i, (s, d, k) in filas.items():
            if s is not None and s == i:
                propios.append(i)
                s = None
            if d is not None and d == i:
                propios.append(i)
                d = None
            filas[i] = (s, d, k)
        if propios:
            self._aviso("error", "Animales registrados como su propio padre o madre (se anuló ese progenitor).",
                        sorted(set(propios), key=str))

        # progenitores ausentes como individuos -> animales base
        agregados = []
        for i, (s, d, k) in list(filas.items()):
            for p in (s, d):
                if p is not None and p not in filas:
                    filas[p] = (None, None, -1)
                    agregados.append(p)
        self.agregados = set(agregados)
        if agregados:
            self._aviso("info", f"{len(agregados)} progenitores no figuraban como individuos y se agregaron "
                                "como animales base (padres desconocidos).", sorted(agregados, key=str)[:200])

        # bisexualidad: el mismo animal usado como padre y como madre
        como_padre = {s for s, d, k in filas.values() if s is not None}
        como_madre = {d for s, d, k in filas.values() if d is not None}
        bisexuales = como_padre & como_madre
        autofec = [i for i, (s, d, k) in filas.items() if s is not None and s == d]
        if bisexuales:
            self._aviso("advertencia", f"{len(bisexuales)} animales figuran como padre y también como madre.",
                        sorted(bisexuales, key=str))
        if autofec:
            self._aviso("advertencia", f"{len(autofec)} animales tienen el mismo individuo como padre y madre "
                                       "(autofecundación).", sorted(autofec, key=str))

        if sexos is not None:
            sexo_de = {}
            for i, sx in zip(ids, sexos):
                if i is not None and i not in sexo_de:
                    sexo_de[i] = normalizar_sexo(sx)
            malos_p = sorted([p for p in como_padre if sexo_de.get(p) == "H"], key=str)
            malos_m = sorted([m for m in como_madre if sexo_de.get(m) == "M"], key=str)
            if malos_p:
                self._aviso("error", f"{len(malos_p)} padres están registrados como hembras.", malos_p)
            if malos_m:
                self._aviso("error", f"{len(malos_m)} madres están registradas como machos.", malos_m)
            self.sexo_registrado = sexo_de
        else:
            self.sexo_registrado = {}

        self._ordenar(filas)
        self._codificar()

    # ------------------------------------------------------------------ armado
    def _aviso(self, tipo, mensaje, ids):
        self.avisos.append({"tipo": tipo, "mensaje": mensaje, "ids": [str(x) for x in ids][:500],
                            "total": len(ids)})

    def _ordenar(self, filas):
        """Ordena por generación (Kahn); detecta ciclos y los rompe."""
        hijos = defaultdict(list)
        pendientes = {}
        for i, (s, d, k) in filas.items():
            padres = {p for p in (s, d) if p is not None}
            pendientes[i] = len(padres)
            for p in padres:
                hijos[p].append(i)
        generacion = {}
        cola = [i for i, c in pendientes.items() if c == 0]
        for i in cola:
            generacion[i] = 0
        pos = 0
        while pos < len(cola):
            i = cola[pos]
            pos += 1
            for h in hijos[i]:
                generacion[h] = max(generacion.get(h, 0), generacion[i] + 1)
                pendientes[h] -= 1
                if pendientes[h] == 0:
                    cola.append(h)
        bloqueados = [i for i, c in pendientes.items() if c > 0]
        if bloqueados:
            # Los bloqueados son los animales de los ciclos y sus descendientes; solo se cortan los
            # enlaces dentro de cada ciclo (componente fuertemente conexa), así los descendientes
            # conservan a sus padres.
            componente = _componentes_fuertes(bloqueados, filas)
            tam = Counter(componente.values())
            en_ciclo = [i for i in bloqueados if tam[componente[i]] > 1]
            if not en_ciclo:  # no debería ocurrir; evita un bucle infinito
                en_ciclo, componente = bloqueados, {i: 0 for i in bloqueados}
            self._aviso("error", f"{len(en_ciclo)} animales forman ciclos imposibles (un animal es ancestro de sí "
                                 "mismo). Se anularon los progenitores dentro de cada ciclo para poder continuar; "
                                 "sus descendientes conservan sus padres.",
                        sorted(en_ciclo, key=str))
            for i in en_ciclo:
                s, d, k = filas[i]
                c = componente[i]
                filas[i] = (None if s is not None and componente.get(s) == c else s,
                            None if d is not None and componente.get(d) == c else d, k)
            return self._ordenar(filas)
        self.filas = filas
        orden_original = {i: (k if k >= 0 else -1) for i, (s, d, k) in filas.items()}
        self.orden = sorted(filas, key=lambda i: (generacion[i], orden_original[i]))
        self.generacion_dict = generacion

    def _codificar(self):
        self.n = len(self.orden)
        self.codigo = {i: c + 1 for c, i in enumerate(self.orden)}
        self.ids = self.orden
        s = np.zeros(self.n + 1, dtype=np.int64)
        d = np.zeros(self.n + 1, dtype=np.int64)
        for c, i in enumerate(self.orden, start=1):
            pa, ma, _ = self.filas[i]
            s[c] = self.codigo[pa] if pa is not None else 0
            d[c] = self.codigo[ma] if ma is not None else 0
        self.s, self.d = s, d
        self.generacion = np.array([0] + [self.generacion_dict[i] for i in self.orden], dtype=np.int64)
        self._F = None
        self._D = None

    # ------------------------------------------------------------------ endogamia
    def endogamia(self):
        """Coeficientes de endogamia (vector de longitud n+1, posición 0 = desconocido)."""
        if self._F is not None:
            return self._F
        s, d, n = self.s, self.d, self.n
        F = np.zeros(n + 1)
        F[0] = -1.0
        D = np.zeros(n + 1)
        L = np.zeros(n + 1)
        memo = {}
        paso = max(1, n // 50)
        for i in range(1, n + 1):
            si, di = s[i], d[i]
            D[i] = 0.5 - 0.25 * (F[si] + F[di])
            if si == 0 or di == 0:
                F[i] = 0.0
            else:
                clave = (si, di) if si <= di else (di, si)
                f = memo.get(clave)
                if f is None:
                    L[i] = 1.0
                    monton = [-i]
                    visto = {i}
                    f = -1.0
                    while monton:
                        j = -heapq.heappop(monton)
                        sj, dj = s[j], d[j]
                        lj = L[j]
                        if sj:
                            if sj not in visto:
                                visto.add(sj)
                                heapq.heappush(monton, -sj)
                            L[sj] += 0.5 * lj
                        if dj:
                            if dj not in visto:
                                visto.add(dj)
                                heapq.heappush(monton, -dj)
                            L[dj] += 0.5 * lj
                        f += lj * lj * D[j]
                        L[j] = 0.0
                    memo[clave] = f
                F[i] = f
            if i % paso == 0:
                if self._cancelado and self._cancelado():
                    raise Cancelado()
                if self._progreso:
                    self._progreso(i / n, f"Endogamia: {i:,} de {n:,} animales".replace(",", " "))
        F[0] = -1.0
        self._F, self._D = F, D
        return F

    def F(self):
        """Endogamia sin la posición 0 (alineado con self.ids)."""
        return self.endogamia()[1:].copy()

    def D(self):
        self.endogamia()
        return self._D

    # ------------------------------------------------------------------ matrices
    def a_inversa(self):
        """A^-1 (n x n, csr) con endogamia; índices 0..n-1 alineados con self.ids."""
        self.endogamia()
        n = self.n
        s, d = self.s[1:], self.d[1:]
        b = 1.0 / self._D[1:]
        i = np.arange(n)
        filas, cols, vals = [i], [i], [b]
        for p in (s, d):
            m = p > 0
            ii, pp, bb = i[m], p[m] - 1, b[m]
            filas += [ii, pp, pp]
            cols += [pp, ii, pp]
            vals += [-bb / 2, -bb / 2, bb / 4]
        m = (s > 0) & (d > 0)
        ss, dd, bb = s[m] - 1, d[m] - 1, b[m]
        filas += [ss, dd]
        cols += [dd, ss]
        vals += [bb / 4, bb / 4]
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))), shape=(n, n))
        return A.tocsr()

    def log_det_a(self):
        return float(np.sum(np.log(self.D()[1:])))

    def _m_triangular(self):
        n = self.n
        s, d = self.s[1:], self.d[1:]
        i = np.arange(n)
        filas, cols, vals = [i], [i], [np.ones(n)]
        for p in (s, d):
            m = p > 0
            filas.append(i[m])
            cols.append(p[m] - 1)
            vals.append(np.full(m.sum(), -0.5))
        return sp.coo_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))),
                             shape=(n, n)).tocsr()

    def columnas_a(self, codigos):
        """Columnas de A para los animales indicados (códigos 1..n). Devuelve n x k."""
        self.endogamia()
        M = self._m_triangular()
        E = np.zeros((self.n, len(codigos)))
        for j, c in enumerate(codigos):
            E[c - 1, j] = 1.0
        W = spsolve_triangular(M.T.tocsr(), E, lower=False)
        U = W * self._D[1:, None]
        return spsolve_triangular(M, U, lower=True)

    def submatriz_a(self, codigos):
        cols = self.columnas_a(codigos)
        idx = np.array(codigos) - 1
        return cols[idx, :]

    # ------------------------------------------------------------------ descriptivos
    def generaciones_equivalentes(self):
        """Generaciones completas equivalentes (ECG), máximas trazadas y completas."""
        n = self.n
        ecg = np.zeros(n + 1)
        maxg = np.zeros(n + 1)
        comp = np.zeros(n + 1)
        s, d = self.s, self.d
        for i in range(1, n + 1):
            si, di = s[i], d[i]
            ecg[i] = ((ecg[si] + 1) if si else 0) / 2 + ((ecg[di] + 1) if di else 0) / 2
            maxg[i] = max((maxg[si] + 1) if si else 0, (maxg[di] + 1) if di else 0)
            comp[i] = (min(comp[si], comp[di]) + 1) if (si and di) else 0
        return ecg[1:], maxg[1:], comp[1:]

    def ancestros(self, codigo, max_gen=None):
        """Conjunto de códigos de ancestros (con su distancia mínima en generaciones)."""
        dist = {codigo: 0}
        pila = [codigo]
        while pila:
            j = pila.pop()
            g = dist[j]
            if max_gen is not None and g >= max_gen:
                continue
            for p in (self.s[j], self.d[j]):
                if p and (p not in dist or dist[p] > g + 1):
                    dist[p] = g + 1
                    pila.append(p)
        return dist

    def descendientes(self):
        hijos = defaultdict(list)
        for i in range(1, self.n + 1):
            for p in (self.s[i], self.d[i]):
                if p:
                    hijos[int(p)].append(i)
        return hijos

    def estadisticas(self):
        n = self.n
        s, d = self.s[1:], self.d[1:]
        F = self.F()
        ecg, maxg, comp = self.generaciones_equivalentes()
        ambos = int(np.sum((s > 0) & (d > 0)))
        solo_p = int(np.sum((s > 0) & (d == 0)))
        solo_m = int(np.sum((s == 0) & (d > 0)))
        ninguno = int(np.sum((s == 0) & (d == 0)))
        hijos_padre = Counter(int(x) for x in s if x)
        hijos_madre = Counter(int(x) for x in d if x)
        n_padres, n_madres = len(hijos_padre), len(hijos_madre)
        consanguineos = F > 1e-12
        # incremento individual de endogamia (Gutiérrez et al., 2009) y Ne realizado
        con_ecg = ecg > 1
        delta_f = np.full(n, np.nan)
        delta_f[con_ecg] = 1 - (1 - F[con_ecg]) ** (1 / (ecg[con_ecg] - 1))
        ne_realizado = (1 / (2 * np.nanmean(delta_f))) if np.any(con_ecg) and np.nanmean(delta_f) > 0 else math.nan
        ne_sexos = (4 * n_padres * n_madres / (n_padres + n_madres)) if n_padres and n_madres else math.nan
        por_generacion = []
        for g in range(int(self.generacion[1:].max()) + 1 if n else 0):
            m = self.generacion[1:] == g
            if m.any():
                por_generacion.append({"generacion": g, "animales": int(m.sum()),
                                       "f_media": float(F[m].mean()), "f_max": float(F[m].max()),
                                       "consanguineos": int((F[m] > 1e-12).sum())})
        return {
            "animales": n,
            "agregados": len(self.agregados),
            "ambos_padres": ambos, "solo_padre": solo_p, "solo_madre": solo_m, "sin_padres": ninguno,
            "padres": n_padres, "madres": n_madres,
            "progenie_por_padre_media": float(np.mean(list(hijos_padre.values()))) if n_padres else 0.0,
            "progenie_por_padre_max": max(hijos_padre.values()) if n_padres else 0,
            "progenie_por_madre_media": float(np.mean(list(hijos_madre.values()))) if n_madres else 0.0,
            "progenie_por_madre_max": max(hijos_madre.values()) if n_madres else 0,
            "generaciones": int(self.generacion[1:].max()) + 1 if n else 0,
            "ecg_media": float(ecg.mean()) if n else 0.0,
            "gen_max_media": float(maxg.mean()) if n else 0.0,
            "gen_completas_media": float(comp.mean()) if n else 0.0,
            "f_media": float(F.mean()) if n else 0.0,
            "f_media_consanguineos": float(F[consanguineos].mean()) if consanguineos.any() else 0.0,
            "f_max": float(F.max()) if n else 0.0,
            "consanguineos": int(consanguineos.sum()),
            "delta_f_media": float(np.nanmean(delta_f)) if np.any(con_ecg) else math.nan,
            "ne_realizado": ne_realizado,
            "ne_sexos": ne_sexos,
            "por_generacion": por_generacion,
            "top_padres": [(self.ids[k - 1], v) for k, v in hijos_padre.most_common(15)],
            "top_madres": [(self.ids[k - 1], v) for k, v in hijos_madre.most_common(15)],
            "ecg": ecg, "maxg": maxg, "comp": comp,
        }


def _componentes_fuertes(nodos, filas):
    """Componentes fuertemente conexas (Kosaraju iterativo) del grafo progenitor → cría
    restringido a `nodos`. Devuelve {animal: número de componente}."""
    dentro = set(nodos)
    hijos = defaultdict(list)
    padres = defaultdict(list)
    for i in nodos:
        s, d, _ = filas[i]
        for p in {s, d}:
            if p is not None and p in dentro:
                hijos[p].append(i)
                padres[i].append(p)
    orden, visto = [], set()
    for raiz in nodos:
        if raiz in visto:
            continue
        visto.add(raiz)
        pila = [(raiz, iter(hijos[raiz]))]
        while pila:
            v, it = pila[-1]
            w = next(it, None)
            if w is None:
                pila.pop()
                orden.append(v)
            elif w not in visto:
                visto.add(w)
                pila.append((w, iter(hijos[w])))
    componente = {}
    for c, raiz in enumerate(reversed(orden)):
        if raiz in componente:
            continue
        componente[raiz] = c
        pila = [raiz]
        while pila:
            v = pila.pop()
            for w in padres[v]:
                if w not in componente:
                    componente[w] = c
                    pila.append(w)
    return componente


def normalizar_sexo(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s.endswith(".0"):
        s = s[:-2]
    if s in ("1", "m", "macho", "male", "h1", "toro", "carnero", "masculino", "♂"):
        return "M"
    if s in ("2", "0", "f", "h", "hembra", "female", "vaca", "oveja", "femenino", "♀"):
        return "H"
    return None
