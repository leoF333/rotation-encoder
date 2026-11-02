"""B1 — analyse : courbes empiriques (données de rotation « real » et substituts gaussiens « surrogate »)
confrontées aux prédictions des répliques de `src/aerot/theory.py` (v0.2 : E_train, E_gen, E_OOD).

Entrées : results/b1/*.json (scripts/b1_replicas.py), data/spectra/*.npz (rendu analytic ss=4, ω_max = 4π/T).
Sorties :
  results/b1_aggregate.csv            moyennes, erreurs-types, médianes par (template, ρ, type, P, régresseur)
  results/b1_theory_comparison.md     tableau théorie vs substitut vs données réelles
  figures/b1_learning_curves_creneau.png, figures/b1_learning_curves_uniform.png
  figures/b1_gaussian_equivalence.png  rapports réel/théorie et substitut/théorie
  (+ parties « large » et « finite » si présentes : figures/b1_large_alpha.png, figures/b1_finite_size.png)
Usage : python scripts/b1_analysis.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aerot import render, spectra, templates
from aerot.paths import FIG, RES, SPECTRA
from aerot.plotting import setup, save, plt, PALETTE, INK2, TEMPLATE_COLOR
from matplotlib.ticker import NullFormatter

from aerot import theory as th

T = 10
WMAX = 4 * np.pi / T
TEMPL = ["gaussian", "quarter_circle", "cross", "segment"]
RIDGES = ["ridge1e-02", "ridge1e-04", "ridge1e-06", "ridge1e-08"]
RCOL = dict(zip(RIDGES, [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[6]]))
EIG_TOL = 1e-14      # modes numériquement nuls du spectre population (bruit de quadrature) ignorés


def load_runs(pattern="results/b1/*.json"):
    rows = []
    for f in glob.glob(str(RES.parent / pattern)):
        r = json.load(open(f))
        for reg, v in r["results"].items():
            rows.append(dict(template=r["template"], dist=r["dist"], d=r["d"], N=r["N"], kind=r["kind"], P=r["P"],
                             seed=r["seed"], reg=reg, lam1=r["lam1_pop"], lam_r=v.get("lam_r", np.nan),
                             label_noise=r.get("label_noise", 0.0),
                             Etrain=v["Etrain"], gen=v["gen"], ood=v.get("ood", np.nan)))
    return pd.DataFrame(rows)


def aggregate(df):
    g = df.groupby(["template", "dist", "d", "kind", "reg", "P"])
    out = []
    for q in ["Etrain", "gen", "ood"]:
        a = g[q].agg(["mean", "sem", "median", "count"])
        a.columns = [f"{q}_{c}" for c in a.columns]
        out.append(a)
    return pd.concat(out, axis=1).reset_index()


class Theory:
    """Prédictions de theory.py à partir du spectre population (analytic ss=4, quadrature n = 4000, identique à
    l'export data/spectra). Pour ρ = créneau, la matrice test (gap) dans la base propre d'entraînement est
    recalculée sur TOUS les modes non nuls (l'export la tronque à K_export ≤ 1000 modes, ce qui biaise E_OOD à petit
    ridge pour les templates discontinus) ; mise en cache dans results/b1_theory_cache/."""

    def __init__(self, name, dist, n=4000, n_test=2000):
        z = np.load(SPECTRA / f"{name}_{dist}_d28_T10_wmax{WMAX:.4f}.npz")
        lam = z["eigvals"]
        Kt = int(np.sum(lam > EIG_TOL * lam[0]))
        self.lam, self.c, self.o2 = lam[:Kt], z["c_proj"][:Kt], float(z["omega2"])
        self.lam1 = float(lam[0])
        self.test = None
        if dist == "creneau":
            cache = RES / "b1_theory_cache" / f"{name}_creneau_n{n}_tol{EIG_TOL:.0e}.npz"
            if not cache.exists():
                cache.parent.mkdir(exist_ok=True)
                om, w = render.quadrature("creneau", WMAX, n)
                S = spectra.GramSpectrum(render.make_videos(om, name, d=28, T=T).numpy(), w, om)
                assert abs(S.lam[0] - self.lam1) < 1e-9 * self.lam1
                K = int(np.sum(S.lam > EIG_TOL * S.lam[0]))
                og, wg = render.quadrature("gap", WMAX, n_test)
                tq = S.test_quantities(render.make_videos(og, name, d=28, T=T).numpy(), wg, og, K)
                np.savez(cache, lam=S.lam[:K], c=S.c_proj[:K], omega2=S.omega2, test_cov=tq["test_cov_in_train_basis"],
                         test_c=tq["test_c_proj"], test_omega2=tq["test_omega2"], test_trace=tq["test_trace"])
            zc = np.load(cache)
            self.lam, self.c, self.o2 = zc["lam"], zc["c"], float(zc["omega2"])
            self.test = (zc["test_cov"], zc["test_c"], float(zc["test_omega2"]))
            self.test_energy_frac = float(np.trace(zc["test_cov"]) / zc["test_trace"])

    def curve(self, r, Ps):
        out = dict(Etrain=[], gen=[], ood=[])
        for P in Ps:
            lr = r * P * self.lam1
            op = th.order_parameters(self.lam, self.c, self.o2, P, lr)
            out["Etrain"].append(op["Etrain"])
            out["gen"].append(op["Egen"])
            out["ood"].append(th.ood_error(self.lam, self.c, self.o2, *self.test, P, lr) if self.test else np.nan)
        return {k: np.array(v) for k, v in out.items()}


def fig_learning(agg, theo, dist):
    qs = ["Etrain", "gen"] + (["ood"] if dist == "creneau" else [])
    qlab = dict(Etrain="$E_{\\rm train}$", gen="$E_{\\rm gen}$", ood="$E_{\\rm OOD}$ (gap)")
    Ps = np.unique(np.round(np.logspace(np.log10(5), np.log10(20000), 70)))
    fig, axs = plt.subplots(len(qs), len(TEMPL), figsize=(3.2 * len(TEMPL), 2.7 * len(qs)), sharex=True)
    for j, name in enumerate(TEMPL):
        for reg in RIDGES:
            r = float(reg[5:])
            tc = theo[(name, dist)].curve(r, Ps)
            for i, q in enumerate(qs):
                ax = axs[i, j]
                ax.loglog(Ps, tc[q], color=RCOL[reg], lw=1.3, label=f"théorie, r={r:.0e}" if (i, j) == (0, 0) else None)
                for kind, mfc in [("surrogate", "white"), ("real", RCOL[reg])]:
                    s = agg[(agg.template == name) & (agg.dist == dist) & (agg.reg == reg) & (agg.kind == kind) & (agg.d == 28)]
                    ax.errorbar(s.P, s[f"{q}_mean"], yerr=s[f"{q}_sem"], fmt="o", ms=3.5, mfc=mfc, mec=RCOL[reg],
                                ecolor=RCOL[reg], elinewidth=0.8, capsize=0, lw=0,
                                label=(("données de rotation" if kind == "real" else "substitut gaussien")
                                       if (i, j) == (0, 0) and reg == RIDGES[0] else None))
        for i, q in enumerate(qs):
            axs[i, j].set_ylabel(qlab[q] if j == 0 else "")
        axs[0, j].set_title(templates.LABELS_FR[name])
        axs[-1, j].set_xlabel("P (nombre d'échantillons)")
    axs[0, 0].legend(fontsize=6, loc="lower left")
    fig.suptitle(f"B1 — encodeur linéaire ridge, ρ = {dist}, ω_max = 4π/T, λ_r = r·P·λ_1 "
                 f"(moyenne ± erreur-type, 5 tirages ; 3 pour P = 7000)", fontsize=9)
    save(fig, FIG / f"b1_learning_curves_{dist}.png",
         dict(dist=dist, wmax=WMAX, ridge_r=RIDGES, eig_tol=EIG_TOL, theory_version=th.__version__,
              note="points pleins = données de rotation, creux = substituts gaussiens de mêmes moments ; traits = répliques"))


def fig_equivalence(agg, theo):
    """Rapport (empirique / théorie) pour E_gen et E_OOD, médiane et moyenne sur les tirages, r = 1e-6."""
    reg = "ridge1e-06"
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    specs = [("uniform", "gen", "$E_{\\rm gen}$, ρ uniforme"), ("creneau", "gen", "$E_{\\rm gen}$, ρ créneau"),
             ("creneau", "ood", "$E_{\\rm OOD}$ (gap)")]
    for ax, (dist, q, title) in zip(axs, specs):
        for k, name in enumerate(TEMPL):
            col = PALETTE[k]
            for kind, ls, mk in [("real", "-", "o"), ("surrogate", "--", "s"), ("strat", ":", "^")]:
                s = agg[(agg.template == name) & (agg.dist == dist) & (agg.reg == reg) & (agg.kind == kind) & (agg.d == 28)
                        & (agg.P <= 7000)]
                if s.empty:
                    continue
                s = s.sort_values("P")
                tc = theo[(name, dist)].curve(1e-6, s.P.values)[q]
                lab = {"real": "rotation", "surrogate": "substitut", "strat": "rotation, ω stratifiés"}[kind]
                ax.loglog(s.P, s[f"{q}_median"] / tc, ls=ls, marker=mk, ms=3, color=col, lw=1.1,
                          mfc=col if kind != "surrogate" else "white", label=f"{templates.LABELS_FR[name]} ({lab})")
        ax.axhline(1, color=INK2, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("P")
    axs[0].set_ylabel("médiane empirique / théorie (r = 1e-6)")
    axs[-1].legend(fontsize=6, ncol=1, loc="upper right")
    save(fig, FIG / "b1_gaussian_equivalence.png", dict(reg=reg, stat="median over seeds", wmax=WMAX))


def table(agg, theo):
    lines = ["# B1 — théorie des répliques (theory.py v" + th.__version__ + ") vs simulations",
             "",
             "Encodeur linéaire sans biais, ω_max = 4π/T, rendu analytic ss=4, λ_r = r·P·λ_1. Moyenne sur 5 tirages "
             "(3 pour P = 7000) ; « réel méd. » = médiane. Écart = (empirique − théorie)/théorie.",
             ""]
    for dist in ["uniform", "creneau"]:
        qs = ["Etrain", "gen"] + (["ood"] if dist == "creneau" else [])
        for q in qs:
            lines += [f"## {q} — ρ = {dist}", "",
                      "| template | r | P | théorie | substitut (moy.) | réel (moy.) | réel (méd.) | écart substitut | écart réel (méd.) |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for name in TEMPL:
                for reg in ["ridge1e-02", "ridge1e-06"]:
                    r = float(reg[5:])
                    for P in [20, 80, 320, 1000, 7000]:
                        tv = theo[(name, dist)].curve(r, [P])[q][0]
                        row = {}
                        for kind in ["surrogate", "real"]:
                            s = agg[(agg.template == name) & (agg.dist == dist) & (agg.reg == reg) & (agg.kind == kind)
                                    & (agg.P == P) & (agg.d == 28)]
                            row[kind] = (float(s[f"{q}_mean"].iloc[0]), float(s[f"{q}_median"].iloc[0])) if len(s) else (np.nan, np.nan)
                        lines.append(f"| {name} | {r:.0e} | {P} | {tv:.3e} | {row['surrogate'][0]:.3e} | {row['real'][0]:.3e} | "
                                     f"{row['real'][1]:.3e} | {(row['surrogate'][0] - tv) / tv:+.2f} | {(row['real'][1] - tv) / tv:+.2f} |")
            lines.append("")
    lines += deviation_summary(agg, theo)
    (RES / "b1_theory_comparison.md").write_text("\n".join(lines) + "\n")


def deviation_summary(agg, theo):
    """Écart relatif médian |empirique (médiane des tirages) − théorie| / théorie, par type de données, observable et plage de P."""
    rows = []
    for name in TEMPL:
        for dist in ["uniform", "creneau"]:
            for reg in RIDGES:
                r = float(reg[5:])
                s = agg[(agg.template == name) & (agg.dist == dist) & (agg.reg == reg) & (agg.d == 28) & (agg.P <= 7000)]
                Ps = np.sort(s.P.unique())
                tc = theo[(name, dist)].curve(r, Ps)
                for kind in ["surrogate", "real"]:
                    for q in ["Etrain", "gen"] + (["ood"] if dist == "creneau" else []):
                        for P, tv in zip(Ps, tc[q]):
                            e = s[(s.kind == kind) & (s.P == P)][f"{q}_median"]
                            if len(e):
                                rows.append(dict(kind=kind, q=q, P=P, dev=abs(float(e.iloc[0]) - tv) / tv))
    d = pd.DataFrame(rows)
    d["Pbin"] = pd.cut(d.P, [0, 40, 320, 7000], labels=["5–40", "80–320", "1000–7000"])
    t = d.groupby(["kind", "q", "Pbin"], observed=True)["dev"].agg(["median", lambda x: np.percentile(x, 90), "count"])
    t.columns = ["médiane", "90e centile", "n"]
    lines = ["", "## Synthèse : écart relatif |empirique − théorie| / théorie", "",
             "Médiane des tirages pour l'empirique ; statistiques sur 4 templates × 2 ρ × 4 ridges × P de la plage.", "",
             "| données | observable | plage de P | écart médian | 90e centile | n |", "|---|---|---|---|---|---|"]
    for (kind, q, pb), v in t.iterrows():
        lines.append(f"| {'substitut gaussien' if kind == 'surrogate' else 'rotation'} | {q} | {pb} | {v['médiane']:.3f} | "
                     f"{v['90e centile']:.2f} | {int(v['n'])} |")
    return lines


def spectrum_for(name, dist, d, n=4000):
    om, w = render.quadrature(dist, WMAX, n)
    X = render.make_videos(om, name, d=d, T=T).numpy()
    S = spectra.GramSpectrum(X, w, om)
    Kt = int(np.sum(S.lam > EIG_TOL * S.lam[0]))
    return S.lam[:Kt], S.c_proj[:Kt], S.omega2


def fig_large(df):
    """α = P/N > 1 sans ridge : E_train empirique vs formule du mémoire (1 − 1/α)·σ²_res et théorie à rang effectif."""
    s = df[(df.P > 7840) & (df.d == 28)]
    if s.empty:
        return None
    lines = ["# B1 — régime α > 1 (P > N), ρ uniforme", "",
             "Mémoire : E_train = (1 − 1/α)·X (X = variance résiduelle, formule corrigée : σ²_res). "
             "Théorie (T2, λ_r → 0) : E_train = (1 − df/P)² E_gen avec df = nombre de modes appris.", "",
             "| template | type | P | α | régresseur | E_train (moy.) | E_gen (moy.) | σ²_res (pop., modes > 1e-14 λ₁) | (1−1/α)σ²_res | théorie répliques |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name in sorted(s.template.unique()):
        lam, c, o2 = spectrum_for(name, "uniform", 28)
        s2 = th.residual_variance(lam, c, o2)
        for (kind, P, reg), g in s.groupby(["kind", "P", "reg"]):
            if g.template.iloc[0] != name and not (g.template == name).any():
                continue
            g = g[g.template == name]
            if g.empty:
                continue
            a = P / 7840
            lr = g.lam_r.iloc[0] if reg.startswith("ridge") else 0.0
            thv = th.order_parameters(lam, c, o2, P, lr)["Etrain"] if reg.startswith("ridge") else np.nan
            lines.append(f"| {name} | {kind} | {P} | {a:.2f} | {reg} | {g.Etrain.mean():.3e} | {g.gen.mean():.3e} | "
                         f"{s2:.3e} | {(1 - 1 / a) * s2:.3e} | {thv:.3e} |")
    (RES / "b1_large_alpha.md").write_text("\n".join(lines) + "\n")
    return True


def fig_labelnoise(dfall):
    """Formule du mémoire E_train = (1 − 1/α)·σ²_res (α = P/N) vs E_train mesuré avec bruit d'étiquette σ_ε, pinv quasi
    sans troncature (1e-10) et ridge ; la bonne variable est le rang effectif r (modes appris), pas N."""
    s = dfall[dfall.label_noise > 0]
    if s.empty:
        return None
    se = float(s.label_noise.iloc[0])
    N = 7840
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    lines = ["# B1 — test de la formule du mémoire E_train = (1 − 1/α)·σ²_res", "",
             f"Bruit d'étiquette σ_ε = {se:g} (σ²_res ≈ σ_ε² = {se**2:.1e}), ρ uniforme, ω_max = 4π/T, moyenne sur 3 tirages. "
             "« rang » = nombre de modes conservés par la pinv 1e-10 (≈ rang numérique du jeu de P vidéos).", "",
             "| template | type | P | α = P/N | E_train (pinv 1e-10) | (1 − 1/α)σ² (mémoire) | σ²(1 − r/P), r = min(P, rang pop.) | E_train (ridge 1e-8) | répliques (ridge 1e-8) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for ax, name in zip(axs, ["gaussian", "cross"]):
        g = s[s.template == name]
        lam, c, o2 = spectrum_for(name, "uniform", 28)
        rank_pop = int(np.sum(lam > 1e-10 * lam[0]))
        for kind, mk in [("real", "o"), ("surrogate", "s")]:
            for reg, col in [("pinv1e-10", PALETTE[0]), ("ridge1e-08", PALETTE[1])]:
                a = g[(g.kind == kind) & (g.reg == reg)].groupby("P").Etrain.mean()
                ax.loglog(a.index, a.values / se**2, mk, color=col, mfc=col if kind == "real" else "white",
                          label=f"{reg}, {'rotation' if kind == 'real' else 'substitut'}")
        Ps = np.logspace(np.log10(30), np.log10(12000), 60)
        al = Ps / N
        ax.loglog(Ps[al > 1], (1 - 1 / al[al > 1]), color=INK2, lw=1.2, ls="--", label="mémoire : 1 − 1/α (α = P/N)")
        ax.loglog(Ps, np.clip(1 - np.minimum(Ps, rank_pop) / Ps, 1e-6, None), color=PALETTE[2], lw=1.0,
                  label=f"1 − r/P (r = rang {rank_pop})")
        tv = [th.order_parameters(lam, c, o2 + se**2, P, 1e-8 * P * lam[0])["Etrain"] / se**2 for P in Ps]
        ax.loglog(Ps, tv, color=PALETTE[1], lw=1.0, label="répliques (ridge 1e-8, σ² = σ_ε²)")
        ax.axvline(N, color=INK2, lw=0.6, ls=":")
        ax.set_title(templates.LABELS_FR[name])
        ax.set_xlabel("P (trait pointillé vertical : P = N)")
        for P in sorted(g.P.unique()):
            for kind in ["real", "surrogate"]:
                e1 = g[(g.kind == kind) & (g.reg == "pinv1e-10") & (g.P == P)].Etrain.mean()
                e2 = g[(g.kind == kind) & (g.reg == "ridge1e-08") & (g.P == P)].Etrain.mean()
                a_ = P / N
                tv1 = th.order_parameters(lam, c, o2 + se**2, P, 1e-8 * P * lam[0])["Etrain"]
                lines.append(f"| {name} | {kind} | {P} | {a_:.2f} | {e1:.3e} | {((1 - 1 / a_) * se**2 if a_ > 1 else float('nan')):.3e} | "
                             f"{se**2 * max(1 - rank_pop / P, 0):.3e} | {e2:.3e} | {tv1:.3e} |")
    axs[0].set_ylabel("E_train / σ_ε²")
    axs[0].set_ylim(1e-3, 2)
    axs[1].legend(fontsize=6, loc="lower right")
    save(fig, FIG / "b1_memoire_formula.png", dict(label_noise=se, N=N))
    (RES / "b1_memoire_formula.md").write_text("\n".join(lines) + "\n")
    return True


def fig_finite(df):
    s = df[(df.d != 28) | ((df.d == 28) & df.P.isin([784, 2352, 7840]))]
    s = s[(s.dist == "uniform") & (s.kind == "real")]
    if s[s.d != 28].empty:
        return None
    reg = "ridge1e-06"
    s = s[s.reg == reg]
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.3))
    for k, name in enumerate(sorted(s.template.unique())):
        for j, d in enumerate(sorted(s.d.unique())):
            g = s[(s.template == name) & (s.d == d)].groupby("P")["gen"].median().reset_index()
            N = d * d * T
            col = TEMPLATE_COLOR[name]
            mk = ["o", "s", "^"][j % 3]
            axs[0].loglog(g.P / N, g.gen, marker=mk, color=col, lw=1, ms=4, label=f"{templates.LABELS_FR[name]}, d={d}")
            axs[1].loglog(g.P, g.gen, marker=mk, color=col, lw=1, ms=4)
            lam, c, o2 = spectrum_for(name, "uniform", d, n=3000)
            Ps = np.logspace(np.log10(g.P.min()), np.log10(g.P.max()), 30)
            tv = [th.order_parameters(lam, c, o2, P, 1e-6 * P * lam[0])["Egen"] for P in Ps]
            axs[1].loglog(Ps, tv, color=col, lw=0.8, ls=":")
    axs[0].set_xlabel("α = P/N")
    axs[1].set_xlabel("P")
    for ax in axs:
        ax.xaxis.set_minor_formatter(NullFormatter())
    axs[0].set_ylabel("E_gen (médiane, r = 1e-6)")
    axs[0].legend(fontsize=6)
    axs[1].set_title("pointillés : répliques", fontsize=8)
    fig.suptitle("B1 — taille finie : d ∈ {14, 20, 28} à α fixé (ρ uniforme, ω_max = 4π/T)", fontsize=9)
    save(fig, FIG / "b1_finite_size.png", dict(reg=reg, dist="uniform", stat="median"))
    return True


def main():
    setup()
    dfall = load_runs()
    df = dfall[dfall.label_noise == 0]
    agg = aggregate(df)
    agg.to_csv(RES / "b1_aggregate.csv", index=False)
    theo = {(n, dd): Theory(n, dd) for n in TEMPL for dd in ["uniform", "creneau"]}
    for dist in ["uniform", "creneau"]:
        fig_learning(agg, theo, dist)
    fig_equivalence(agg, theo)
    table(agg, theo)
    fig_large(df)
    fig_finite(df)
    fig_labelnoise(dfall)
    print("ok")


if __name__ == "__main__":
    main()
