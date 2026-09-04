import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.report_common import (COLORS, add_common_args, load_csv,
                                    poisson_ci, resolve_dirs, rolling,
                                    save_fig, setup_style, write_table)

NEEDED = ["episode", "agent_id", "reward", "collisions", "goals",
          "mean_velocity", "route_completion"]


def artifact_mask(df):
    """True en las filas donde route_completion no llego a escribirse."""
    cols = [c for c in ("goals", "collisions", "offroad", "timeouts") if c in df.columns]
    if not cols:
        return pd.Series(False, index=df.index)
    return (df[cols].sum(axis=1) == 0)


def _series_plot(df, col, ylabel, title, name, fig_dir, window, pct=False):
    fig, ax = plt.subplots()
    for aid, g in df.groupby("agent_id"):
        g = g.sort_values("episode")
        c = COLORS.get(aid, None)
        ax.scatter(g["episode"], g[col], s=6, alpha=0.18, color=c, linewidths=0)
        ax.plot(g["episode"], rolling(g[col], window), color=c, lw=2, label=aid)
    ax.set_xlabel("Episodio (rollout de 2048 steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (media movil de {window} episodios)")
    if pct:
        ax.set_ylim(0, 1)
    ax.legend()
    return save_fig(fig, name, fig_dir)


def figures(df, fig_dir, window):
    print("\n--- Figuras de entrenamiento ---")
    _series_plot(df, "reward", "Recompensa por rollout",
                 "F1. Recompensa", "f1_recompensa", fig_dir, window)
    _series_plot(df, "collisions", "Colisiones por rollout",
                 "F2. Colisiones (muertes y respawns)", "f2_colisiones", fig_dir, window)

    ok = df[~artifact_mask(df)]
    if len(ok):
        _series_plot(ok, "route_completion", "Fraccion de ruta completada",
                     "F3. Ruta completada", "f3_ruta", fig_dir, window, pct=True)
    else:
        print("  [salto]  F3: todas las filas son artefacto de logging")

    _series_plot(df, "mean_velocity", "Velocidad media (m/s)",
                 "F4. Velocidad", "f4_velocidad", fig_dir, window)

    if "actor_loss" in df.columns and "critic_loss" in df.columns:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for aid, g in df.groupby("agent_id"):
            g = g.sort_values("episode")
            a1.plot(g["episode"], rolling(g["actor_loss"], window),
                    color=COLORS.get(aid), lw=1.8, label=aid)
        a1.set_ylabel("Perdida del actor")
        a1.set_title(f"F5. Perdidas  (media movil de {window} episodios)")
        a1.legend()

        cr = df.drop_duplicates(subset=["episode"]).sort_values("episode")
        a2.scatter(cr["episode"], cr["critic_loss"], s=6, alpha=0.18,
                   color="#555555", linewidths=0)
        a2.plot(cr["episode"], rolling(cr["critic_loss"], window),
                color="#555555", lw=2)
        a2.set_ylabel("Perdida del critico")
        a2.set_xlabel("Episodio (rollout de 2048 steps)")
        a2.set_yscale("log")
        save_fig(fig, "f5_perdidas", fig_dir)

    fig, ax = plt.subplots()
    for aid, g in df.groupby("agent_id"):
        g = g.sort_values("episode")
        ax.plot(g["episode"], g["goals"].cumsum(), color=COLORS.get(aid),
                lw=2, label=aid)
    ax.set_xlabel("Episodio (rollout de 2048 steps)")
    ax.set_ylabel("Goles acumulados")
    ax.set_title("F6. Llegadas a meta acumuladas")
    ax.legend()
    save_fig(fig, "f6_goles_acumulados", fig_dir)


def window_table(df, tab_dir, size):
    """T1: resumen por ventanas. Con el IC de Poisson en las llegadas a metea para no leer
    como mejora lo que es ruido de conteo."""
    print("\n--- T1: evolucion por ventanas ---")
    df = df.sort_values("episode").copy()
    df["_win"] = df.groupby("agent_id").cumcount() // size

    rows = []
    for (win, aid), g in df.groupby(["_win", "agent_id"]):
        a, b = int(g["episode"].min()), int(g["episode"].max())
        n = len(g)
        goles = int(g["goals"].sum())
        rate, cl, ch = poisson_ci(goles, n)
        ok = g[~artifact_mask(g)]
        rows.append({
            "ventana": f"ep{a}-{b}",
            "agente": aid,
            "n": n,
            "goles": goles,
            "goles/ep": round(rate, 3),
            "IC95": f"[{cl:.2f}, {ch:.2f}]",
            "colisiones": round(float(g["collisions"].mean()), 1),
            "velocidad": round(float(g["mean_velocity"].mean()), 2),
            "ruta media": round(float(ok["route_completion"].mean()), 3) if len(ok) else float("nan"),
            "% ruta>0.30": round(100.0 * float((ok["route_completion"] > 0.30).mean()), 0) if len(ok) else float("nan"),
            "n ruta": len(ok),
        })
    out = pd.DataFrame(rows).sort_values(["agente", "ventana"])
    return write_table(
        out, "t1_entrenamiento_por_ventanas", tab_dir,
        caption=(f"Evolucion del entrenamiento en ventanas de {size} episodios. "
                 "Un episodio son 2048 steps con respawn tras cada muerte, por lo que "
                 "'colisiones' cuenta muertes por rollout. El IC95 de las llegadas a meta es de "
                 "Poisson: con pocos sucesos las diferencias aparentes suelen ser ruido."),
        label="entrenamiento_ventanas")


def main():
    p = argparse.ArgumentParser(description="Informe de entrenamiento para la memoria")
    add_common_args(p, with_latest=False)
    p.add_argument("--window", type=int, default=20, help="Ventana de la media alrededor del episodio ya que un episodio aislado no dice nada")
    p.add_argument("--table-window", type=int, default=50,
                   help="Tamañoo de bloque de la tabla T1")
    p.add_argument("--run-id", default=None,
                   help="Analizar solo esta ejecucion (por defecto todas: al reanudar "
                        "el entrenamiento la curva continua entre varios run_id)")
    args = p.parse_args()

    setup_style()
    fig_dir, tab_dir = resolve_dirs(args)

    print("Cargando datos de entrenamiento...")
    path = os.path.join(args.results_dir, "train_episodes.csv")
    df = load_csv(path, needed=NEEDED, label="train_episodes.csv")
    if df is None:
        print("\nSin datos de entrenamiento: nada que hacer.")
        return 0

    if args.run_id:
        df = df[df["run_id"] == args.run_id].copy()
        if df.empty:
            print(f"\nNingun dato para run_id={args.run_id}")
            return 0
    elif "run_id" in df.columns and df["run_id"].nunique() > 1:
        print(f"{df['run_id'].nunique()} ejecuciones encadenadas "
              "(reanudaciones), se usan todas")

    df = df.sort_values("episode")
    dup = df.duplicated(subset=["episode", "agent_id"]).sum()
    if dup:
        print(f" {dup} filas con (episode, agent_id) repetido: "
              "hay reinicios que reusan numeracion, revisa --run-id")

    n_art = int(artifact_mask(df).sum())
    print(f"{n_art}/{len(df)} filas sin causa de terminacion: "
          "su route_completion no se escribio y se excluye de las metricas de ruta")

    figures(df, fig_dir, args.window)
    window_table(df, tab_dir, args.table_window)

    print(f"\nListo. Figuras en {fig_dir}\n       Tablas  en {tab_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
