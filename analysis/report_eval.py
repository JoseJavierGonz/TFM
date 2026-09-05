import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.report_common import (COLORS, OUTCOME_ES, PERCEPTION_ES,
                                    add_common_args, auc_from_scores,
                                    load_csv, overlap_coefficient,
                                    report_runs, resolve_dirs,
                                    save_fig, setup_style, write_table)

EVAL_NEEDED = ["scenario", "agent_id", "outcome", "reward", "route_completion"]
OUTCOMES = ["goal", "collision", "offroad", "timeout"]


def _pct(mask):
    """Para porcentajes(de llegadas a meta por ejemplo)"""
    return round(100.0 * float(np.mean(mask)), 1) if len(mask) else float("nan")


def summarize(g):
    """Las cifras de una condicion. velocidad media, recompensa media, rutas completadas"""
    out = {"n": len(g)}
    for o in OUTCOMES:
        out[f"% {OUTCOME_ES[o]}"] = _pct(g["outcome"] == o)
    out["recompensa"] = round(float(g["reward"].mean()), 0)
    if "mean_velocity" in g:
        out["vel media"] = round(float(g["mean_velocity"].mean()), 2)
    if "max_velocity" in g:
        out["vel max"] = round(float(g["max_velocity"].mean()), 2)
    out["ruta"] = round(float(g["route_completion"].mean()), 3)
    if "steps_alive" in g:
        out["steps"] = int(g["steps_alive"].mean())
    return out


def table_main(df, tab_dir):
    """T2: comparacion ENTRE ESCENARIOS. Se restringe a ground_truth."""
    print("\nT2: resultados principales")
    if "perception" in df.columns:
        n_all = len(df)
        df = df[df["perception"] == "ground_truth"]
        if len(df) < n_all:
            print(f"  [nota]   se excluyen {n_all - len(df)} filas de condiciones "
                  "de radar; la comparacion de percepcion es la tabla T3")
    if df.empty:
        print("T2: no hay filas con perception=ground_truth")
        return None

    rows = []
    for (sc, aid), g in df.groupby(["scenario", "agent_id"]):
        rows.append(dict(escenario=sc, agente=aid, **summarize(g)))
    for sc, g in df.groupby("scenario"):
        rows.append(dict(escenario=sc, agente="ambos", **summarize(g)))
    out = pd.DataFrame(rows).sort_values(["escenario", "agente"])
    return write_table(
        out, "t2_evaluacion_principal", tab_dir,
        caption=("Resultados de evaluacion por escenario y agente, con acciones "
                 "deterministas (media de la gaussiana). Cada episodio aporta un "
                 "unico desenlace."),
        label="eval_principal")


def table_perception(df, tab_dir, scenario="traffic"):
    """T3: la ablacion. Misma politica y mismo escenario, cambiando de donde salen
    las 6 features de percepcion."""
    sub = df[df["scenario"] == scenario]
    if sub.empty or "perception" not in sub.columns or sub["perception"].nunique() < 2:
        print(f"\nT3: hacen falta >=2 condiciones de percepcion en "
              f"scenario={scenario} (hay {sub['perception'].nunique() if len(sub) else 0})")
        return None

    print("\n T3: ablacion de percepcion ")
    rows = []
    for perc, g in sub.groupby("perception"):
        rows.append(dict(percepcion=PERCEPTION_ES.get(perc, perc), _key=perc,
                         **summarize(g)))
    out = pd.DataFrame(rows)

    order = [p for p in ("ground_truth", "radar_validated", "radar_raw")
             if p in set(out["_key"])]
    out["_o"] = out["_key"].apply(lambda k: order.index(k) if k in order else 99)
    out = out.sort_values("_o")

    if "ground_truth" in set(out["_key"]):
        base = out[out["_key"] == "ground_truth"].iloc[0]
        for col in ("% meta", "% colision", "ruta"):
            if col in out.columns:
                out[f"d {col}"] = (out[col] - base[col]).round(3)
    out = out.drop(columns=["_key", "_o"])

    return write_table(
        out, "t3_ablacion_percepcion", tab_dir,
        caption=(f"Ablacion del stack de percepcion en el escenario '{scenario}'. "
                 "La politica y los pesos son identicos en las tres filas: solo "
                 "cambia el origen de las features. Las columnas 'd' son la "
                 "diferencia frente a ground truth."),
        label="ablacion_percepcion")


def table_checkpoints(df, tab_dir):
    """T5: comparativa entre checkpoints, para evaluar en dos
    puntos del entrenamiento."""
    if "checkpoint" not in df.columns or df["checkpoint"].nunique() < 2:
        print("\nT5: solo hay un checkpoint evaluado")
        return None
    print("\nT5: comparativa entre checkpoints ")
    rows = []
    for (ck, sc, aid), g in df.groupby(["checkpoint", "scenario", "agent_id"]):
        rows.append(dict(checkpoint=os.path.basename(ck), escenario=sc,
                         agente=aid, **summarize(g)))
    out = pd.DataFrame(rows).sort_values(["escenario", "agente", "checkpoint"])
    return write_table(
        out, "t5_comparativa_checkpoints", tab_dir,
        caption=("Evaluacion de la misma politica en dos checkpoints del "
                 "entrenamiento. La evolucion no es monotona ni simetrica entre "
                 "agentes: conviene leer cada agente por separado."),
        label="comparativa_checkpoints")


def fig_outcomes(df, fig_dir, key, name, title, order=None):
    """Desenlaces por condicion. `key` es la columna que define la comparacion:
    'scenario' para comparar escenarios (con una sola percepcion) y 'perception'
    para la ablacion (dentro de un solo escenario). Mezclar ambas en una figura
    juntaria poblaciones distintas en la misma barra."""
    df = df.copy()
    df["_cond"] = (df[key].map(PERCEPTION_ES).fillna(df[key])
                   if key == "perception" else df[key])

    conds = [c for c in order if c in set(df["_cond"])] if order \
        else sorted(df["_cond"].unique())
    fig, ax = plt.subplots(figsize=(max(7, 1.9 * len(conds)), 4.5))
    bottom = np.zeros(len(conds))
    palette = {"goal": "#2ca02c", "collision": "#d62728",
               "offroad": "#ff7f0e", "timeout": "#7f7f7f"}
    for o in OUTCOMES:
        vals = [_pct(df[df["_cond"] == c]["outcome"] == o) for c in conds]
        vals = np.nan_to_num(np.array(vals, dtype=float))
        ax.bar(conds, vals, bottom=bottom, label=OUTCOME_ES[o], color=palette[o])
        bottom += vals
    ax.set_ylabel("% de episodios")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save_fig(fig, name, fig_dir)


def fig_route_ecdf(df, fig_dir, key, name, title, order=None):
    """Distribucion acumulada de ruta completada. Mismo criterio que fig_outcomes:
    `key` fija que compara la figura, para no mezclar escenarios con percepciones."""
    fig, ax = plt.subplots()
    vals = [v for v in order if v in set(df[key])] if order \
        else sorted(df[key].unique())
    for val in vals:
        x = np.sort(df.loc[df[key] == val, "route_completion"].dropna().to_numpy())
        if len(x) == 0:
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        lab = PERCEPTION_ES.get(val, val) if key == "perception" else val
        ax.step(x, y, where="post", lw=2, label=f"{lab} (n={len(x)})",
                color=COLORS.get(val))
    ax.set_xlabel("Fraccion de ruta completada")
    ax.set_ylabel("Fraccion acumulada de episodios")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    save_fig(fig, name, fig_dir)


def _category(type_id):
    t = str(type_id)
    if t.startswith("vehicle"):
        return "vehiculo"
    if t.startswith("walker"):
        return "peaton"
    return "estatico"


def table_funnel(rad, tab_dir):
    """T4: el embudo. Cuanto comprime el clustering y cuanto descarta la validacion."""
    ticks = rad[rad["row_type"] == "tick"]
    if ticks.empty:
        print("\nT4: no hay filas row_type=tick")
        return None
    print("\n--- T4: embudo de deteccion ---")
    rows = []
    for perc, g in ticks.groupby("perception"):
        rows.append({
            "percepcion": PERCEPTION_ES.get(perc, perc),
            "ticks": len(g),
            "detecciones/tick": round(float(g["n_detections"].mean()), 1),
            "clusters/tick": round(float(g["n_clusters"].mean()), 2),
            "validados/tick": round(float(g["n_matched"].mean()), 2),
            "usados/tick": round(float(g["n_used"].mean()), 2),
            "% clusters validados": round(
                100.0 * float(g["n_matched"].sum()) / max(1.0, float(g["n_clusters"].sum())), 1),
        })
    out = pd.DataFrame(rows)
    return write_table(
        out, "t4_embudo_radar", tab_dir,
        caption=("Embudo de procesamiento del radar por tick. El clustering agrupa "
                 "las multiples reflexiones de un mismo objeto; la validacion contra "
                 "ground truth descarta las que no corresponden a ningun actor. El "
                 "porcentaje final indica cuanta de la senal cruda es entorno estatico."),
        label="embudo_radar")


def fig_velocity_separability(rad, fig_dir, tab_dir):
    """F9: si vehiculos y estaticos se solapan en velocidad radial,
    ningun umbral sobre esa magnitud los separa."""
    det = rad[rad["row_type"] == "detection"].copy()
    if det.empty or "velocity" not in det.columns:
        print("\nF9: no hay filas row_type=detection")
        return None

    det["categoria"] = det["matched_type"].apply(_category)
    det["velocity"] = pd.to_numeric(det["velocity"], errors="coerce")
    det = det[np.isfinite(det["velocity"])]
    if det.empty:
        print("\nF9: ninguna velocidad valida")
        return None

    print("\n F9: separabilidad por velocidad radial")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

    present = [c for c in ("vehiculo", "peaton", "estatico")
               if (det["categoria"] == c).any()]
    lo, hi = det["velocity"].quantile([0.005, 0.995])
    bins = np.linspace(float(lo), float(hi), 60) if hi > lo else 30
    for c in present:
        v = det[det["categoria"] == c]["velocity"]
        a1.hist(v, bins=bins, density=True, alpha=0.55, label=f"{c} (n={len(v)})",
                color=COLORS.get(c))
    a1.set_xlabel("Velocidad radial (m/s)")
    a1.set_ylabel("Densidad")
    a1.set_title("F9a. Velocidad por tipo de objeto")
    a1.legend()

    dyn = det[det["categoria"].isin(["vehiculo", "peaton"])]["velocity"].abs()
    sta = det[det["categoria"] == "estatico"]["velocity"].abs()
    auc = auc_from_scores(dyn, sta)
    ov = overlap_coefficient(dyn, sta)

    if len(dyn) and len(sta):
        m = max(float(dyn.quantile(0.995)), float(sta.quantile(0.995)))
        b2 = np.linspace(0, m, 50) if m > 0 else 30
        a2.hist(sta, bins=b2, density=True, alpha=0.55, label=f"estatico (n={len(sta)})",
                color=COLORS["estatico"])
        a2.hist(dyn, bins=b2, density=True, alpha=0.55, label=f"actor real (n={len(dyn)})",
                color=COLORS["vehiculo"])
        a2.set_xlabel("|Velocidad radial| (m/s)")
        a2.set_ylabel("Densidad")
        a2.set_title(f"F9b. AUC = {auc:.3f}   solapamiento = {ov:.2f}")
        a2.legend()
    save_fig(fig, "f9_velocidad_separabilidad", fig_dir)

    if "perception" in det.columns and det["perception"].nunique() > 1:
        cond_rows = []
        for p, S in det.groupby("perception"):
            a = S[S["categoria"].isin(["vehiculo", "peaton"])]["velocity"].abs()
            e = S[S["categoria"] == "estatico"]["velocity"].abs()
            if len(a) < 20 or len(e) < 20:
                continue
            cond_rows.append({"condicion": PERCEPTION_ES.get(p, p),
                              "n actor real": len(a), "n estatico": len(e),
                              "AUC vel. radial": round(auc_from_scores(a, e), 3),
                              "solapamiento": round(overlap_coefficient(a, e), 3),
                              "media vel. actor": round(float(a.mean()), 3),
                              "media vel. estatico": round(float(e.mean()), 3)})
        if len(cond_rows) > 1:
            aucs = [r["AUC vel. radial"] for r in cond_rows]
            if min(aucs) < 0.5 < max(aucs):
                print("las condiciones tienen AUC a lados opuestos de 0.5: "
                      "el valor agregado se cancela y NO debe reportarse")
            write_table(
                pd.DataFrame(cond_rows), "t7_separabilidad_por_condicion", tab_dir,
                caption=("Separabilidad por velocidad radial calculada de forma "
                         "independiente en cada condicion. La velocidad radial de un "
                         "objeto estatico coincide con la del propio vehiculo, muy "
                         "distinta entre condiciones, por lo que agregar ambas "
                         "poblaciones cancela sus sesgos y produce un valor espurio."),
                label="separabilidad_por_condicion")

    rows = [{"comparacion": "actor real vs estatico",
             "n actor real": len(dyn), "n estatico": len(sta),
             "AUC vel. radial": round(auc, 3),
             "solapamiento": round(ov, 3),
             "media vel. actor": round(float(dyn.mean()), 3) if len(dyn) else float("nan"),
             "media vel. estatico": round(float(sta.mean()), 3) if len(sta) else float("nan")}]
    for c in ("vehiculo", "peaton"):
        s = det[det["categoria"] == c]["velocity"].abs()
        if len(s) and len(sta):
            rows.append({"comparacion": f"{c} vs estatico",
                         "n actor real": len(s), "n estatico": len(sta),
                         "AUC vel. radial": round(auc_from_scores(s, sta), 3),
                         "solapamiento": round(overlap_coefficient(s, sta), 3),
                         "media vel. actor": round(float(s.mean()), 3),
                         "media vel. estatico": round(float(sta.mean()), 3)})
    write_table(
        pd.DataFrame(rows), "t6_separabilidad_radar", tab_dir,
        caption=("Capacidad de un clasificador que solo disponga de la velocidad "
                 "radial para distinguir actores reales de entorno estatico. "
                 "AUC 0.5 significa indistinguibles; solapamiento 1.0 significa "
                 "distribuciones identicas."),
        label="separabilidad_radar")
    print(f"  AUC={auc:.3f}  solapamiento={ov:.2f}  "
          f"(AUC~0.5 => la velocidad radial no separa las clases)")
    return auc


def fig_depth(rad, fig_dir):
    """F10: a que distancia aparece cada tipo de deteccion."""
    det = rad[rad["row_type"] == "detection"].copy()
    if det.empty or "depth" not in det.columns:
        return
    det["depth"] = pd.to_numeric(det["depth"], errors="coerce")
    det = det[np.isfinite(det["depth"])]
    if det.empty:
        return
    det["categoria"] = det["matched_type"].apply(_category)
    fig, ax = plt.subplots()
    bins = np.linspace(0, float(det["depth"].max()), 40)
    for c in ("vehiculo", "peaton", "estatico"):
        v = det[det["categoria"] == c]["depth"]
        if len(v):
            ax.hist(v, bins=bins, density=True, alpha=0.55,
                    label=f"{c} (n={len(v)})", color=COLORS.get(c))
    ax.set_xlabel("Distancia de la deteccion (m)")
    ax.set_ylabel("Densidad")
    ax.set_title("F10. Distancia por tipo de objeto detectado")
    ax.legend()
    save_fig(fig, "f10_distancia", fig_dir)


def main():
    p = argparse.ArgumentParser(description="Informe de evaluacion para la memoria")
    add_common_args(p)
    p.add_argument("--exclude-scenario", nargs="*",
                   default=["smoke", "timing", "video", "test", "debug"],
                   help="Escenarios de diagnostico que no forman parte de los "
                        "resultados. Pasa --exclude-scenario sin valores para no "
                        "descartar ninguno")
    p.add_argument("--ablation-scenario", default="traffic",
                   help="Escenario donde se compara la percepcion (por defecto traffic)")
    args = p.parse_args()

    setup_style()
    fig_dir, tab_dir = resolve_dirs(args)

    print("Cargando datos de evaluacion...")
    ev = load_csv(os.path.join(args.results_dir, "eval_episodes.csv"),
                  needed=EVAL_NEEDED, label="eval_episodes.csv")
    rad = load_csv(os.path.join(args.results_dir, "radar_detections.csv"),
                   needed=["row_type", "perception"], label="radar_detections.csv")

    #fuera las tiradas de diagnostico, antes de calcular nada
    for name, df in (("eval_episodes.csv", ev), ("radar_detections.csv", rad)):
        if df is None or not args.exclude_scenario or "scenario" not in df.columns:
            continue
        drop = df["scenario"].isin(args.exclude_scenario)
        if drop.any():
            for sc, n in df.loc[drop, "scenario"].value_counts().items():
                print(f"  [excl]   {name}: se descartan {n} filas de scenario={sc}")
            df.drop(df.index[drop], inplace=True)

    if ev is not None:
        if "perception" not in ev.columns:
            ev["perception"] = "ground_truth"
        ev = report_runs(ev, ["scenario", "perception", "checkpoint"], args.latest)
        n_to = int((ev["outcome"] == "timeout").sum())
        if n_to and float(ev.loc[ev["outcome"] == "timeout", "route_completion"].max()) == 0.0:
            print(f"los {n_to} episodios en timeout tienen "
                  "route_completion=0: el arreglo de run.py no estaba aplicado "
                  "cuando se generaron; su ruta real no se midio")

        table_main(ev, tab_dir)
        table_perception(ev, tab_dir, args.ablation_scenario)
        table_checkpoints(ev, tab_dir)
        print("\nFiguras de evaluacion")
        PERC_ORDER = ["ground_truth", "radar_validated", "radar_raw"]

        #comparacion ENTRE ESCENARIOS: una sola percepcion, para no mezclar
        base = ev[ev["perception"] == "ground_truth"]
        if base["scenario"].nunique() > 1:
            fig_outcomes(base, fig_dir, "scenario", "f7a_desenlaces_escenario",
                         "F7a. Desenlace por escenario (percepcion de referencia)")
            fig_route_ecdf(base, fig_dir, "scenario", "f8a_ruta_escenario",
                           "F8a. Ruta completada por escenario (percepcion de referencia)")
        else:
            print("  [salto]  F7a/F8a: hace falta mas de un escenario con ground truth")

        #ablacion de PERCEPCION: un solo escenario, las tres condiciones
        abl = ev[ev["scenario"] == args.ablation_scenario]
        if abl["perception"].nunique() > 1:
            fig_outcomes(abl, fig_dir, "perception", "f7b_desenlaces_percepcion",
                         f"F7b. Desenlace por percepcion (escenario '{args.ablation_scenario}')",
                         order=PERC_ORDER)
            fig_route_ecdf(abl, fig_dir, "perception", "f8b_ruta_percepcion",
                           f"F8b. Ruta completada por percepcion (escenario '{args.ablation_scenario}')",
                           order=PERC_ORDER)
        else:
            print(f"  [salto]  F7b/F8b: hace falta mas de una percepcion en "
                  f"scenario={args.ablation_scenario}")
    else:
        print("\nSin datos de evaluacion: se salta el informe de evaluacion.")

    if rad is not None:
        rad = report_runs(rad, ["scenario", "perception"], args.latest)
        table_funnel(rad, tab_dir)
        fig_velocity_separability(rad, fig_dir, tab_dir)
        fig_depth(rad, fig_dir)
    else:
        print("\nSin datos de radar: se salta la ablacion de percepcion.")

    if ev is None and rad is None:
        print("\nNada que procesar todavia. Ejecuta primero train/run.py.")
        return 0

    print(f"\nListo. Figuras en {fig_dir}\n       Tablas  en {tab_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
