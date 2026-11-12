"""
Simulador de reintentos de entrega en la última milla - versión 3.0
"Gestión predictiva de reintentos de entrega en la última milla" (López, 2026)

Cambios respecto de la v2.1 :
 * Franjas horarias configurables (--franja_h 2, 3 o 4 horas) dentro de una jornada de reparto (--hora_ini, --hora_fin).
 * Se evalúan TODOS los pedidos de cada cliente, desde el primero. El historial se arma con los pedidos
   anteriores resueltos bajo el mismo escenario. El primer pedido es el caso "cliente nuevo" (sin historial).
 * Los resultados se informan por nivel de historial: 0 pedidos previos, 1-2, 3 o más, y global.
 * Escenarios:  A = reglas fijas (franja de la ruta, reintento al día siguiente)
               B = modelo que aprende del historial de cada cliente (franja, día y sus éxitos y fallos previos)
               C = B + consulta previa al cliente según costo esperado (agente de decisión)
 * No se usan zonas: el modelo solo conoce la franja, el día y el historial propio del cliente.

Uso:  python simulador_reintentos_v3.py --n 10000 --semillas 30 --fallo 0.0668 --franja_h 3 --salida base.csv
"""
import argparse, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

# ----------------------------------------------------------------------------------------------
# PARÁMETROS (cada uno figura en la tabla de constantes del Anexo C con su justificación)
# ----------------------------------------------------------------------------------------------
P = dict(
    fallo_objetivo=0.0668,              # Sharma et al. (2018); niveles alternativos: ver --fallo
    max_intentos=3,                     # SUPUESTO del autor
    max_contactos=3,                    # SUPUESTO del autor (solo C)
    c_intento=1.0, c_contacto=0.1, c_devol=0.8,   # SUPUESTO (unidades relativas a una visita)
    p_respuesta_contacto=0.70,          # SUPUESTO del autor
    contraste=3.0,                      # SUPUESTO del autor: ausencia fuera de la franja preferida / ausencia dentro de ella
    tope_presencia=0.98,                # decisión de diseño
    hora_ini=9, hora_fin=21, franja_h=3,  # decisión de diseño (no hay estándar en la literatura)
    n_train=6000, T_train=6, visitas_expl=2,    # decisión de diseño
    rf_arboles=150, rf_hoja_min=20,             # decisión de diseño
    semilla_base=1000,
)
# Cada cliente tiene UNA franja preferida (en la que es más probable encontrarlo). Se sortea con la distribución de
# horas de entregas completadas del dataset (archivo horas_entrega.csv que genera analisis_horarios_lade.py).
# Si no se indica el archivo, las franjas son equiprobables.
def cargar_participacion(ruta, nf):
    if not ruta: return np.full(nf, 1.0 / nf)
    t = pd.read_csv(ruta, index_col=0)['total']
    horas = np.arange(P['hora_ini'], P['hora_fin'])
    v = np.array([t.get(h, 0.0) for h in horas]).reshape(nf, P['franja_h']).sum(1)
    return v / v.sum()

ESC = ['A', 'B', 'C']

def config():
    nf = (P['hora_fin'] - P['hora_ini']) // P['franja_h']
    assert nf * P['franja_h'] == P['hora_fin'] - P['hora_ini'], 'la jornada debe ser múltiplo de la franja'
    return nf

def generar_clientes(n, T, rng, nf, K, share):
    # ausencia dentro de la franja preferida = a; fuera de ella = contraste * a. Se elige a para que la tasa de
    # fallo promedio de una visita en franja al azar sea igual al fallo objetivo.
    a = P['fallo_objetivo'] * nf / (1 + (nf - 1) * P['contraste'])
    pref = rng.choice(nf, size=n, p=share)
    aus = np.where(np.arange(nf)[None, :] == pref[:, None], a, P['contraste'] * a)
    p = 1 - np.clip(aus, 0.0, P['tope_presencia'])
    return dict(n=n, T=T, pref=pref, p=p, u=rng.random((n, T, K, 2)),
                ruta=rng.integers(0, nf, (n, T, K)), dia0=rng.integers(0, 7, (n, T)))

def feat(franja, dia, att, hs, hf, ff):
    return np.column_stack([franja, dia, att, hs, hf, ff])

def entrenar(rng, nf, K, share):
    n, T = P['n_train'], P['T_train']
    d = generar_clientes(n, T, rng, nf, K, share); hs = np.zeros((n, nf)); hf = np.zeros((n, nf)); X, y = [], []
    ar = np.arange(n)
    for t in range(T):
        ff = np.zeros((n, nf)); ss = np.zeros((n, nf))
        for j in range(P['visitas_expl']):
            fr = d['ruta'][:, t, j]; ok = d['u'][:, t, j, 0] < d['p'][ar, fr]
            X.append(feat(fr, (d['dia0'][:, t] + j) % 7, np.full(n, j), hs, hf, ff)); y.append(ok)
            ff[ar[~ok], fr[~ok]] += 1; ss[ar[ok], fr[ok]] += 1
        hs += ss; hf += ff
    seed = int(rng.integers(1e6)); y = np.concatenate(y)
    mk = lambda: RandomForestClassifier(n_estimators=P['rf_arboles'], min_samples_leaf=P['rf_hoja_min'], n_jobs=-1, random_state=seed)
    return mk().fit(np.vstack(X), y)

def pedido(d, t, esc, rf, hs, hf, nf, K):
    n = d['n']; ar = np.arange(n)
    fin = np.zeros(n, bool); comp = np.zeros(n, bool); att = np.zeros(n); ncont = np.zeros(n)
    costo = np.zeros(n); dias = np.zeros(n); ff = np.zeros((n, nf)); ss = np.zeros((n, nf))
    for k in range(K):
        act = ~fin & (att < P['max_intentos'])
        if not act.any(): break
        dia = (d['dia0'][:, t] + k) % 7
        idx = ar[act]
        if esc == 'A':
            slot = d['ruta'][:, t, k]; pb = np.zeros(n)
        else:
            pm = np.zeros((n, nf))
            for f in range(nf):
                fr = np.full(len(idx), f)
                Xf = feat(fr, dia[idx], att[idx], hs[idx], hf[idx], ff[idx]); pm[idx, f] = rf.predict_proba(Xf)[:, 1]
            slot = pm.argmax(1); pb = pm.max(1)
        if esc == 'C':
            contactar = (ncont < P['max_contactos']) & (P['c_contacto'] < P['p_respuesta_contacto'] * (1 - pb) * P['c_intento'])
        else:
            contactar = np.zeros(n, bool)
        presente = d['u'][ar, t, k, 0] < d['p'][ar, slot]
        responde = d['u'][ar, t, k, 1] < P['p_respuesta_contacto']
        m = act & contactar
        costo[m] += P['c_contacto']; ncont[m] += 1
        evita = m & responde & ~presente            # el cliente avisa que no está: no se visita, no se consume intento
        dias[evita] += 1; ff[ar[evita], slot[evita]] += 1
        v = act & ~evita
        costo[v] += P['c_intento']; att[v] += 1; dias[v] += 1
        ok = v & presente; mal = v & ~presente
        comp[ok] = True; fin[ok] = True; ss[ar[ok], slot[ok]] += 1
        ff[ar[mal], slot[mal]] += 1
    costo[~comp] += P['c_devol']
    hs += ss; hf += ff
    return dict(att=att, costo=costo, comp=comp, dias=dias, ncont=ncont)

def simular(d, esc, rf, nf, K):
    n = d['n']; hs = np.zeros((n, nf)); hf = np.zeros((n, nf)); res = []
    for t in range(d['T']):
        r = pedido(d, t, esc, rf, hs, hf, nf, K); r['ord'] = np.full(n, t); res.append(r)
    return {k: np.concatenate([r[k] for r in res]) for k in res[0]}

NIVELES = {'global': lambda o: o >= 0, '0 pedidos previos': lambda o: o == 0,
           '1-2 pedidos previos': lambda o: (o >= 1) & (o <= 2), '3 o más pedidos previos': lambda o: o >= 3}

def indicadores(r, mk):
    c = max(r['comp'][mk].sum(), 1)
    return dict(IPE=r['att'][mk].sum() / c, costo_por_entrega=r['costo'][mk].sum() / c,
                no_completadas=100 * (1 - r['comp'][mk].mean()), dias=r['dias'][mk & r['comp']].mean(),
                contactos_por_entrega=r['ncont'][mk].sum() / c,
                fallo_primera_visita=100 * (1 - (r['comp'][mk] & (r['att'][mk] == 1)).mean()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10000); ap.add_argument('--semillas', type=int, default=30)
    ap.add_argument('--salida', default='resultados_v3.csv'); ap.add_argument('--ordenes', type=int, default=6)
    ap.add_argument('--fallo', type=float, default=P['fallo_objetivo'], help='tasa de fallo en la primera visita (regla A)')
    ap.add_argument('--franja_h', type=int, default=P['franja_h'], help='largo de la franja en horas (2, 3 o 4)')
    ap.add_argument('--hora_ini', type=int, default=P['hora_ini']); ap.add_argument('--hora_fin', type=int, default=P['hora_fin'])
    ap.add_argument('--c_contacto', type=float, default=P['c_contacto']); ap.add_argument('--p_resp', type=float, default=P['p_respuesta_contacto'])
    ap.add_argument('--max_intentos', type=int, default=P['max_intentos'])
    ap.add_argument('--contraste', type=float, default=P['contraste'], help='ausencia fuera de la franja preferida / dentro de ella')
    ap.add_argument('--horas', default=None, help='horas_entrega.csv generado por analisis_horarios_lade.py')
    a = ap.parse_args()
    P.update(fallo_objetivo=a.fallo, franja_h=a.franja_h, hora_ini=a.hora_ini, hora_fin=a.hora_fin,
             c_contacto=a.c_contacto, p_respuesta_contacto=a.p_resp, max_intentos=a.max_intentos, contraste=a.contraste)
    nf = config(); K = P['max_intentos'] + P['max_contactos']; share = cargar_participacion(a.horas, nf)
    print(f'franjas: {nf} de {P["franja_h"]} h entre {P["hora_ini"]} y {P["hora_fin"]} h | fallo objetivo {P["fallo_objetivo"]:.2%} | n={a.n} | semillas={a.semillas} | contraste={P["contraste"]}')
    print('participación de cada franja (para sortear la franja preferida):', np.round(share, 3))
    filas = []
    for s in range(a.semillas):
        rng = np.random.default_rng(P['semilla_base'] + s)
        rf = entrenar(rng, nf, K, share); d = generar_clientes(a.n, a.ordenes, rng, nf, K, share)
        for e in ESC:
            r = simular(d, e, rf, nf, K)
            for nivel, f in NIVELES.items():
                filas.append(dict(semilla=s, escenario=e, nivel=nivel, **indicadores(r, f(r['ord']))))
        print(f'semilla {s + 1}/{a.semillas} lista', flush=True)
    df = pd.DataFrame(filas); df.to_csv(a.salida, index=False)
    pd.set_option('display.width', 200)
    for nivel in NIVELES:
        print(f'\n=== {nivel} ===')
        print(df[df.nivel == nivel].drop(columns=['semilla', 'nivel']).groupby('escenario').mean().reindex(ESC).round(4).to_string())
    print('\nWilcoxon sobre el IPE (pares por semilla):')
    for nivel in NIVELES:
        w = df[df.nivel == nivel].pivot(index='semilla', columns='escenario', values='IPE')
        for x, y in [('B', 'A'), ('C', 'B'), ('C', 'A')]:
            dif = w[x] - w[y]
            p = wilcoxon(dif)[1] if (dif != 0).any() else float('nan')
            print(f'  {nivel:26s} {x} vs {y}: dif media IPE={dif.mean():+.4f}  p={p:.3g}')

if __name__ == '__main__':
    main()
