"""
Análisis de la hora de las entregas completadas en LaDe-D (Cainiao, Wu et al., 2023/2024).
Sirve para respaldar (a) la ventana horaria de reparto y (b) el largo de las franjas del simulador.

Qué hace
  1. Descarga LaDe-D desde Hugging Face (o lee archivos locales con --carpeta).
  2. Toma la hora de 'delivery_time' (momento en que el repartidor termina la entrega).
  3. Calcula la distribución de entregas por hora, por ciudad.
  4. Agrupa las zonas (aoi_id) con KMeans según su perfil horario y elige k con el coeficiente de silueta.
  5. Guarda: horas_entrega.csv, perfiles_clusters.csv y figura_horarios.png

Uso (Windows):  py analisis_horarios_lade.py --ciudades sh --muestra 2000000     (descarga sola el/los archivo/s)
                py analisis_horarios_lade.py --carpeta C:\\datos\\lade_d   (si ya descargaste los archivos)
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def descargar(ciudades=None, destino='lade_d_descarga'):
    """Descarga LaDe-D desde Hugging Face con la biblioteca estándar (sin huggingface_hub ni pyyaml)."""
    import json, urllib.request, urllib.error
    base = 'https://huggingface.co/datasets/Cainiao-AI/LaDe-D'
    def listar(ruta=''):
        url = f'https://huggingface.co/api/datasets/Cainiao-AI/LaDe-D/tree/main/{ruta}'.rstrip('/')
        with urllib.request.urlopen(url, timeout=60) as r: items = json.load(r)
        out = []
        for it in items:
            if it['type'] == 'directory': out += listar(it['path'])
            elif it['path'].lower().endswith(('.csv', '.parquet')): out.append(it['path'])
        return out
    try:
        archivos = listar()
        if ciudades: archivos = [f for f in archivos if any(c.lower() in os.path.basename(f).lower() for c in ciudades)]
        if not archivos: raise RuntimeError('no se encontraron archivos para las ciudades pedidas')
        os.makedirs(destino, exist_ok=True)
        for f in archivos:
            dst = os.path.join(destino, os.path.basename(f))
            if os.path.exists(dst) and os.path.getsize(dst) > 0: print('ya descargado:', dst); continue
            print('descargando', f, '(puede tardar varios minutos)...', flush=True)
            def prog(n, tam, total):
                if total > 0 and n % 200 == 0: print(f'  {min(n * tam, total) / 1e6:.0f} de {total / 1e6:.0f} MB', flush=True)
            urllib.request.urlretrieve(f'{base}/resolve/main/{f}', dst, prog)
        return destino
    except Exception as e:
        sys.exit(f"""No se pudo descargar automáticamente ({e}).
Descargá los archivos a mano:
  1. Entrá a {base}  ->  pestaña "Files and versions"
  2. Bajá el archivo de una o dos ciudades (por ejemplo delivery_sh, en .csv o .parquet)
  3. Ponelo en una carpeta y ejecutá:  py analisis_horarios_lade.py --carpeta <esa carpeta> --salida .""")

def leer_archivo(f):
    """CSV con pandas. Parquet con pyarrow; si no está disponible, con polars (que sí instala en Python 3.15)."""
    if not f.lower().endswith('.parquet'): return pd.read_csv(f)
    try:
        return pd.read_parquet(f)
    except ImportError:
        try:
            import polars as pl
        except ImportError:
            sys.exit('Para leer .parquet instalá una de estas librerías:  py -m pip install pyarrow   o   py -m pip install polars')
        cols = [c for c in ['city', 'aoi_id', 'delivery_time', 'ds'] if c in pl.read_parquet_schema(f)]
        return pd.DataFrame(pl.read_parquet(f, columns=cols).to_dict(as_series=False))

def leer(carpeta, ciudades, muestra):
    archivos = [f for f in glob.glob(os.path.join(carpeta, '**', '*'), recursive=True) if f.lower().endswith(('.csv', '.parquet'))]
    if ciudades: archivos = [f for f in archivos if any(c.lower() in os.path.basename(f).lower() for c in ciudades)]
    if not archivos: sys.exit('No se encontraron archivos de LaDe-D en la carpeta indicada.')
    partes = []
    for f in sorted(archivos):
        print('leyendo', os.path.basename(f), flush=True)
        df = leer_archivo(f)
        if 'delivery_time' not in df.columns:
            print('  (sin columna delivery_time, se omite; columnas:', list(df.columns)[:12], ')'); continue
        df = df[[c for c in ['city', 'aoi_id', 'delivery_time', 'ds'] if c in df.columns]].copy()
        if 'city' not in df.columns: df['city'] = os.path.splitext(os.path.basename(f))[0]
        if muestra and len(df) > muestra // max(len(archivos), 1): df = df.sample(muestra // max(len(archivos), 1), random_state=1)
        partes.append(df)
    return pd.concat(partes, ignore_index=True)

def hora_de(serie):
    h = serie.astype(str).str.extract(r'(\d{1,2}):(\d{2}):(\d{2})\s*$').astype(float)
    return (h[0] + h[1] / 60).dropna()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--carpeta', default=None); ap.add_argument('--ciudades', nargs='*', default=None)
    ap.add_argument('--muestra', type=int, default=0, help='máximo de filas en total (0 = todas)')
    ap.add_argument('--min_entregas', type=int, default=200, help='mínimo de entregas por zona para el clustering')
    ap.add_argument('--salida', default='.')
    a = ap.parse_args()
    carpeta = a.carpeta or descargar(a.ciudades)
    df = leer(carpeta, a.ciudades, a.muestra)
    h = hora_de(df['delivery_time']); df = df.loc[h.index].assign(hora=h)
    df['h'] = np.floor(df['hora']).astype(int)
    print(f'\nEntregas analizadas: {len(df):,}')
    # 1) distribución por hora
    dist = df.groupby(['city', 'h']).size().unstack(0).fillna(0); dist = dist / dist.sum()
    dist['total'] = df.groupby('h').size() / len(df)
    dist.round(5).to_csv(os.path.join(a.salida, 'horas_entrega.csv'))
    acum = dist['total'].sort_index().cumsum()
    print('\nParticipación de las entregas por hora (total):'); print((dist['total'] * 100).round(1).to_string())
    # ventana que concentra el 90 % central
    p5, p95 = df['hora'].quantile([0.05, 0.95]); print(f'\nEl 90 % central de las entregas ocurre entre las {p5:.1f} y las {p95:.1f} h')
    # 2) clustering de zonas por perfil horario
    cl = None
    if 'aoi_id' in df.columns:
        g = df.groupby(['city', 'aoi_id'])
        cnt = g.size(); ok = cnt[cnt >= a.min_entregas].index
        tab = df.set_index(['city', 'aoi_id']).loc[ok].reset_index().groupby(['city', 'aoi_id', 'h']).size().unstack(2).fillna(0)
        tab = tab.div(tab.sum(axis=1), axis=0)
        print(f'\nZonas con al menos {a.min_entregas} entregas: {len(tab):,}')
        if len(tab) >= 30:
            mejor, sil = None, -1
            for k in range(2, 7):
                km = KMeans(k, n_init=10, random_state=1).fit(tab.values)
                s = silhouette_score(tab.values, km.labels_, sample_size=min(5000, len(tab)), random_state=1)
                print(f'  k={k}: silueta={s:.3f}'); 
                if s > sil: mejor, sil = km, s
            cl = pd.DataFrame(mejor.cluster_centers_, columns=tab.columns); cl.index.name = 'cluster'
            cl['zonas'] = np.bincount(mejor.labels_); cl.round(4).to_csv(os.path.join(a.salida, 'perfiles_clusters.csv'))
            print(f'Se eligió k={mejor.n_clusters} (silueta {sil:.3f})')
    # 3) figura
    fig, ax = plt.subplots(1, 2 if cl is not None else 1, figsize=(12 if cl is not None else 6.5, 4.4)); ax = np.atleast_1d(ax)
    tot = (dist['total'] * 100)
    ax[0].bar(tot.index, tot.values, color='#2F5597'); ax[0].set_xlabel('Hora del día'); ax[0].set_ylabel('% de entregas completadas')
    ax[0].set_title('Entregas completadas por hora'); ax[0].set_xticks(range(0, 24, 2))
    if cl is not None:
        for i, r in cl.iterrows():
            ax[1].plot(r.drop('zonas').index, r.drop('zonas').values * 100, marker='o', ms=3, label=f'Grupo {i + 1} ({int(r["zonas"])} zonas)')
        ax[1].set_xlabel('Hora del día'); ax[1].set_ylabel('% de entregas de la zona'); ax[1].set_title('Perfiles horarios de las zonas (KMeans)')
        ax[1].legend(frameon=False, fontsize=8); ax[1].set_xticks(range(0, 24, 2))
    for x in ax: x.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(a.salida, 'figura_horarios.png'), dpi=200)
    print('\nListo. Archivos: horas_entrega.csv, perfiles_clusters.csv, figura_horarios.png')

if __name__ == '__main__':
    main()
