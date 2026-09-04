import math
import os

import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(REPO_DIR, "results")

#paleta fija para que las figuras sean coherentes entre si
COLORS = {"agent_0": "#1f77b4", "agent_1": "#d62728",
          "ground_truth": "#2ca02c", "radar_validated": "#ff7f0e",
          "radar_raw": "#9467bd",
          "vehiculo": "#1f77b4", "peaton": "#2ca02c", "estatico": "#7f7f7f"}

OUTCOME_ES = {"goal": "meta", "collision": "colision",
              "offroad": "fuera de ruta", "timeout": "timeout"}

PERCEPTION_ES = {"ground_truth": "ground truth", "radar_validated": "radar validado",
                 "radar_raw": "radar crudo"}


def setup_style():
    plt.rcParams.update({
        "figure.figsize": (8, 4.5),
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })




def load_csv(path, needed=None, label=None):
    """Cargamos el dataframe."""
    label = label or os.path.basename(path)
    if not os.path.exists(path):
        print(f" {label}: no existe ({path})")
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"{label}: sin filas")
        return None
    except Exception as e:
        print(f"  [error]  {label}: no se pudo leer ({e})")
        return None
    if df.empty:
        print(f"{label}: 0 filas")
        return None
    if needed:
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"{label}: faltan columnas {missing}")
            return None
    print(f"{label}: {len(df)} filas")
    return df


def report_runs(df, keys, latest_only=False):
    """Los CSV son append-only y acumulan ejecuciones."""
    if df is None or "run_id" not in df.columns:
        return df
    keys = [k for k in keys if k in df.columns]
    if not keys:
        return df

    order = df.groupby(keys + ["run_id"], as_index=False).size()
    dup_keys = order.groupby(keys, as_index=False).size()
    dup_keys = dup_keys[dup_keys["size"] > 1]

    if len(dup_keys):
        print("hay condiciones con varias ejecuciones acumuladas:")
        merged = order.merge(dup_keys[keys], on=keys)
        for _, r in merged.sort_values(keys + ["run_id"]).iterrows():
            combo = ", ".join(f"{k}={r[k]}" for k in keys)
            print(f"{combo}  ->  {r['run_id']}  ({r['size']} filas)")
        if latest_only:
            print("--latest: se conserva solo la mas reciente de cada una")
        else:
            print("se suman TODAS. Usa --latest para quedarte solo con la ultima de cada condicion")

    if not latest_only:
        return df

    keep = order.sort_values("run_id").groupby(keys, as_index=False).tail(1)
    return df[df["run_id"].isin(set(keep["run_id"]))].copy()



def poisson_ci(count, n_episodes, z=1.96):
    """Intervalo de confianza al 95% . El ruido entre ejercicios
    es enorme por ejemplo en llegar a la meta, si paso de 6 a 8 estoy mejorando o es puro ruido?."""
    if n_episodes <= 0:
        return 0.0, 0.0, 0.0
    rate = count / n_episodes
    half = z * math.sqrt(max(count, 0)) / n_episodes
    return rate, max(0.0, rate - half), rate + half


def auc_from_scores(scores_pos, scores_neg):
    """Nos va a decir si el radar sabe diferenciar por velocidad un coche de un muro."""
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.asarray(scores_neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank(method="average").to_numpy()
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def overlap_coefficient(a, b, bins=60):
    """como de ambiguo es lo anterior."""
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    return float(np.minimum(ha, hb).sum() * (edges[1] - edges[0]))


def rolling(series, window):
    """Media centrada, ya que con curriculum uno episodios pueden tener muchas colisiones y otras pocas"""
    return series.rolling(window, min_periods=max(2, window // 4), center=True).mean()




def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_fig(fig, name, fig_dir):
    ensure_dirs(fig_dir)
    path = os.path.join(fig_dir, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  figura -> {os.path.relpath(path, REPO_DIR)}")
    return path


def _tex_escape(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def _fmt(v):
    if isinstance(v, float):
        if not np.isfinite(v):
            return "-"
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def write_table(df, name, tab_dir, caption="", label=None):
    """Vuelca la tabla en CSV y LaTeX booktabs."""
    ensure_dirs(tab_dir)
    df = df.copy()

    csv_path = os.path.join(tab_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)

    cells = [[_fmt(v) for v in row] for row in df.itertuples(index=False)]
    cols = list(df.columns)

    tex = ["\\begin{table}[htbp]", "  \\centering",
           "  \\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}",
           "    \\toprule",
           "    " + " & ".join(_tex_escape(c) for c in cols) + " \\\\",
           "    \\midrule"]
    for r in cells:
        tex.append("    " + " & ".join(_tex_escape(x) for x in r) + " \\\\")
    tex += ["    \\bottomrule", "  \\end{tabular}"]
    if caption:
        tex.append(f"  \\caption{{{_tex_escape(caption)}}}")
    tex.append(f"  \\label{{tab:{label or name}}}")
    tex.append("\\end{table}")
    with open(os.path.join(tab_dir, f"{name}.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")

    print(f"  tabla  -> {os.path.relpath(csv_path, REPO_DIR)} (+ .tex)")
    return df


def add_common_args(parser, results_default=None, with_latest=True):
    parser.add_argument("--results-dir", default=results_default or DEFAULT_RESULTS_DIR,
                        help="Directorio con los CSV (por defecto results/)")
    parser.add_argument("--out-dir", default=None,
                        help="Donde escribir figuras y tablas (por defecto el de results)")
    if with_latest:
        parser.add_argument("--latest", action="store_true",
                            help="Si una condicion tiene varias ejecuciones acumuladas, "
                                 "quedarse solo con la mas reciente. Por defecto se "
                                 "suman todas y se avisa por pantalla")
    return parser


def resolve_dirs(args):
    out = args.out_dir or args.results_dir
    fig_dir = os.path.join(out, "figures")
    tab_dir = os.path.join(out, "tables")
    ensure_dirs(fig_dir, tab_dir)
    return fig_dir, tab_dir
