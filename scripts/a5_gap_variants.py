"""A5 (suite) — recherche des conventions qui reproduisent l'expérience de gap du mémoire (Fig. 6-7).

Valeurs cibles numérisées sur les images du PDF (results/memoire_digit/memoire_Eomega_fig6_7.json) :
pic de E(ω) dans le gap = 9.3e-6 (gaussienne), 1.5e-3 (segment), 1.2e-3 (quart de cercle), 1.3e-2 (croix),
E(ω) < 1e-7 hors du gap ; axe ω ∈ [0, 4π/T] ; zone grisée = [0.439, 0.819] = [0.35, 0.65]·(4π/T).

Facteurs balayés (un json par configuration, idempotent) :
  rendu ∈ {analytic ss=4, analytic ss=1 (échantillonnage ponctuel), interp (rotation bilinéaire)} ;
  ω_max ∈ {2π/T, 4π/T} ; définition du gap ∈ {tiers (CONVENTIONS), « fig » = [0.35, 0.65]·ω_max} ;
  unité de l'angle ∈ {rad, deg (seulement analytic4, 4π/T, tiers)}.
Régresseurs (population, quadrature n = 2000) : pinv tronquée λ_k/λ_1 > τ, τ ∈ {1e-4, 1e-8, 1e-12},
sans biais (CONVENTIONS) et avec biais (covariance centrée) pour τ ∈ {1e-4, 1e-8}.
Remarque : np.linalg.pinv(Φ, rcond=1e-4) coupe les valeurs singulières de Φ, soit λ_k/λ_1 > 1e-8.

Usage : python scripts/a5_gap_variants.py [--part conventions|sigma_r0|report|noise]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from aerot import render, spectra, templates
from aerot.paths import RES

OUT = RES / "a5_variants"
T, D = 10, 28
TEMPL = ["gaussian", "segment", "quarter_circle", "cross", "dipole"]
MEM_PEAK = {"gaussian": 9.3e-6, "segment": 1.5e-3, "quarter_circle": 1.2e-3, "cross": 1.3e-2}
TAUS = [1e-4, 1e-8, 1e-12]


def gap_bounds(gapdef, wmax):
    return (wmax / 3, 2 * wmax / 3) if gapdef == "thirds" else (0.35 * wmax, 0.65 * wmax)


def train_quadrature(gapdef, wmax, n):
    """Quadrature (Gauss-Legendre composite) sur [0, ω_max] privé du gap."""
    a, b = gap_bounds(gapdef, wmax)
    xg, wg = np.polynomial.legendre.leggauss(8)
    nodes, weights = [], []
    for lo, hi in [(0.0, a), (b, wmax)]:
        m = max(1, int(round(n * (hi - lo) / (wmax - (b - a)) / 8)))
        e = np.linspace(lo, hi, m + 1)
        for l0, h0 in zip(e[:-1], e[1:]):
            h = (h0 - l0) / 2
            nodes.append(l0 + h * (xg + 1))
            weights.append(h * wg)
    w = np.concatenate(weights)
    return np.concatenate(nodes), w / w.sum()


def videos(om, name, method, ss, unit, params=None):
    return render.make_videos(np.asarray(om) * unit, name, d=D, T=T, method=method, ss=ss, params=params).numpy()


def run_config(name, method, ss, wmax, gapdef, unit, n=2000, n_grid=601, params=None, tag=None):
    om, w = train_quadrature(gapdef, wmax, n)
    X = videos(om, name, method, ss, unit, params)
    og = np.linspace(0, wmax, n_grid)
    Xg = videos(og, name, method, ss, unit, params)
    a, b = gap_bounds(gapdef, wmax)
    gap = (og > a) & (og < b)
    res = dict(template=name, method=method, ss=ss, wmax=wmax, gapdef=gapdef, unit=unit, n=n,
               params=params or templates.template_params(name, D), regs={})
    for centered in (False, True):
        S = spectra.GramSpectrum(X, w, om, centered=centered)
        if not centered:
            res["lam_top"] = S.lam[:12].tolist()
            res["K"] = {f"{t:.0e}": S.n_above(t) for t in TAUS}
            res["n_999"] = int(np.searchsorted(np.cumsum(S.lam) / S.lam.sum(), 0.999) + 1)
        else:
            res["lam_top_centered"] = S.lam[:12].tolist()
        for tau in (TAUS if not centered else TAUS[:2]):
            K = S.n_above(tau)
            U = S.eigvecs(K)
            Wv = U @ (S.c_proj[:K] / S.lam[:K])
            b0 = 0.0
            if centered:
                b0 = S.omega_mean - (w @ X) @ Wv
            E = (og - (Xg @ Wv + b0)) ** 2
            key = f"{'bias_' if centered else ''}pinv{tau:.0e}"
            res["regs"][key] = dict(gap_max=float(E[gap].max()), gap_mean=float(E[gap].mean()),
                                    out_max=float(E[~gap].max()), out_med=float(np.median(E[~gap])), K=K)
            if key in ("pinv1e-04", "pinv1e-08"):
                res["regs"][key]["curve"] = E.astype(np.float32).tolist()
    res["omega_grid"] = og.tolist()
    return res


def part_conventions():
    confs = []
    for method, ss in [("analytic", 4), ("interp", 1), ("analytic", 1)]:
        for wm in [2, 4]:
            for gapdef in ["thirds", "fig"]:
                confs.append((method, ss, wm, gapdef, 1.0, "rad"))
    confs.append(("analytic", 4, 4, "thirds", np.pi / 180, "deg"))
    for method, ss, wm, gapdef, unit, ulab in confs:
        for name in TEMPL:
            f = OUT / f"conv_{name}_{method}{ss}_wmax{wm}piT_{gapdef}_{ulab}.json"
            if f.exists():
                continue
            t0 = time.time()
            r = run_config(name, method, ss, wm * np.pi / T, gapdef, unit)
            r["time_s"] = time.time() - t0
            f.write_text(json.dumps(r))
            g = r["regs"]
            print(f.name, " ".join(f"{k}:{v['gap_max']:.1e}/{v['out_max']:.1e}" for k, v in g.items()),
                  f"{r['time_s']:.0f}s", flush=True)


def part_sigma_r0():
    """Gaussienne : dépendance de l'erreur dans le gap en (σ, r0) (bande passante angulaire z0 = r0²/2σ²)."""
    for wm in [2, 4]:
        for sigma in [1.5, 2.0, 3.0, 4.0, 5.0]:
            for r0 in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
                f = OUT / f"sr_gaussian_s{sigma:g}_r{r0:g}_wmax{wm}piT.json"
                if f.exists():
                    continue
                p = dict(sigma=sigma, r0=r0, theta0=np.pi / 4)
                t0 = time.time()
                r = run_config("gaussian", "analytic", 4, wm * np.pi / T, "thirds", 1.0, params=p)
                r["time_s"] = time.time() - t0
                for k in r["regs"].values():
                    k.pop("curve", None)
                r.pop("omega_grid")
                f.write_text(json.dumps(r))
                print(f.name, f"z0={r0**2/(2*sigma**2):.2f}",
                      " ".join(f"{k}:{v['gap_max']:.1e}" for k, v in r["regs"].items()), flush=True)


def part_report():
    """Tableau de classement des conventions + figures (mémoire vs nous ; dépendance en (σ, r0))."""
    import glob
    import pandas as pd
    from aerot.paths import FIG
    from aerot.plotting import setup, save, plt, PALETTE, GAP_SHADE, INK2
    setup()
    rows, curves = [], {}
    for f in glob.glob(str(OUT / "conv_*.json")):
        r = json.load(open(f))
        key = (f"{r['method']}{r['ss']}", int(round(r["wmax"] / (np.pi / T))), r["gapdef"],
               "deg" if r["unit"] < 1 else "rad")
        for reg, v in r["regs"].items():
            rows.append(dict(rendu=key[0], wmax=f"{key[1]}π/T", gap=key[2], unite=key[3], reg=reg,
                             template=r["template"], gap_max=v["gap_max"], out_max=v["out_max"], K=v["K"]))
            if "curve" in v:
                curves[(r["template"],) + key + (reg,)] = (np.array(r["omega_grid"]), np.array(v["curve"]))
    df = pd.DataFrame(rows)
    idx = ["rendu", "wmax", "gap", "unite", "reg"]
    pk = df.pivot_table(index=idx, columns="template", values="gap_max")
    om = df.pivot_table(index=idx, columns="template", values="out_max")
    pk["D"] = np.sqrt(sum((np.log10(pk[t]) - np.log10(MEM_PEAK[t])) ** 2 for t in MEM_PEAK) / 4)
    pk["hors_gap_max"] = om[list(MEM_PEAK)].max(axis=1)
    pk = pk.sort_values("D").reset_index()
    lines = ["# A5 — quelles conventions reproduisent l'expérience de gap du mémoire ?", "",
             "Cibles (numérisées sur les Fig. 6–7 du PDF, `results/memoire_digit/`) : pic de E(ω) dans le gap = "
             + ", ".join(f"{k} {v:.1e}" for k, v in MEM_PEAK.items()) + " ; E < 1e-7 hors du gap.", "",
             "D = écart quadratique moyen en décades (log10) sur les 4 pics. Rendu : analytic4 = fonction continue "
             "anti-crénelée 4×4 (défaut), analytic1 = échantillonnage ponctuel, interp1 = rotation bilinéaire de l'image. "
             "gap « thirds » = [ω_max/3, 2ω_max/3] (texte du mémoire), « fig » = [0.35, 0.65]·ω_max (zone grisée des figures). "
             "Régresseur : pinvτ = pseudo-inverse tronquée à λ_k/λ_1 > τ (τ = 1e-8 ⇔ rcond = 1e-4 sur les valeurs "
             "singulières de Φ) ; bias_ = avec biais (covariance centrée). Population : quadrature n = 2000.", "",
             "| rang | rendu | ω_max | gap | unité | régresseur | gaussienne | segment | quart de cercle | croix | dipôle | D (décades) | max hors gap |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in pk.iterrows():
        lines.append(f"| {i + 1} | {r.rendu} | {r.wmax} | {r.gap} | {r.unite} | {r.reg} | {r.gaussian:.1e} | {r.segment:.1e} | "
                     f"{r.quarter_circle:.1e} | {r.cross:.1e} | {r.dipole:.1e} | {r.D:.2f} | {r.hors_gap_max:.1e} |")
    lines.append("| — | *mémoire* | *4π/T (axes)* | *fig* | | *« 1e-4 »* | *9.3e-06* | *1.5e-03* | *1.2e-03* | *1.3e-02* | — | 0 | *< 1e-7* |")
    (RES / "a5_variants_summary.md").write_text("\n".join(lines) + "\n")

    # ---- figure : mémoire vs nous (abscisse ω/ω_max)
    mem = json.load(open(RES / "memoire_digit" / "memoire_Eomega_fig6_7.json"))
    show = ["gaussian", "segment", "quarter_circle", "cross"]
    variants = [(("analytic4", 4, "thirds", "rad", "pinv1e-04"), "4π/T, λ/λ₁ > 1e-4 (protocole littéral)", PALETTE[1], "--"),
                (("analytic4", 2, "thirds", "rad", "pinv1e-08"), "2π/T, λ/λ₁ > 1e-8 (meilleur accord)", PALETTE[0], "-"),
                (("interp1", 2, "thirds", "rad", "pinv1e-08"), "2π/T, 1e-8, rendu interp", PALETTE[2], ":")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, n in zip(axs, show):
        m = mem[n]
        ax.axvspan(1 / 3, 2 / 3, color=GAP_SHADE, alpha=0.10, lw=0)
        ax.semilogy(np.array(m["omega"]) / (4 * np.pi / T), 10 ** np.array(m["log10E"]), color="#0b0b0b", lw=2.2,
                    label="mémoire (numérisé)")
        for key, lab, col, ls in variants:
            k = (n,) + key
            if k in curves:
                og, E = curves[k]
                ax.semilogy(og / og.max(), np.maximum(E, 1e-12), color=col, ls=ls, lw=1.2, label=lab)
        ax.set_ylim(1e-9, 1)
        ax.set_title(templates.LABELS_FR[n])
        ax.set_xlabel("ω / ω_max")
    axs[0].set_ylabel("E(ω) = (ω − W·φ(ω))²")
    axs[-1].legend(fontsize=6, loc="upper right")
    save(fig, FIG / "a5_gap_vs_memoir.png", dict(variants=[v[0] for v in variants], note="abscisse normalisée ω/ω_max ; "
                                                 "mémoire : ω_max = 4π/T d'après ses axes"))

    # ---- figure (σ, r0)
    rows = []
    for f in glob.glob(str(OUT / "sr_*.json")):
        r = json.load(open(f))
        p = r["params"]
        for reg, v in r["regs"].items():
            rows.append(dict(sigma=p["sigma"], r0=p["r0"], wmax=int(round(r["wmax"] / (np.pi / T))), reg=reg,
                             gap_max=v["gap_max"], K4=r["K"]["1e-04"], K8=r["K"]["1e-08"]))
    sr = pd.DataFrame(rows)
    sr = sr[sr.r0 > 0]
    sr["z0"] = sr.r0 ** 2 / (2 * sr.sigma ** 2)
    sr["b"] = sr.r0 / sr.sigma
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    sigs = sorted(sr.sigma.unique())
    for ax, wm in zip(axs, [2, 4]):
        for k, sg in enumerate(sigs):
            for reg, ls, mk in [("pinv1e-04", "--", "o"), ("pinv1e-08", "-", "s")]:
                g = sr[(sr.wmax == wm) & (sr.sigma == sg) & (sr.reg == reg)].sort_values("r0")
                ax.semilogy(g.b, g.gap_max, ls=ls, marker=mk, ms=3.5, color=PALETTE[k], lw=1.1,
                            label=f"σ = {sg:g}, {'1e-4' if reg == 'pinv1e-04' else '1e-8'}")
        ax.axhline(MEM_PEAK["gaussian"], color=INK2, lw=0.8, ls=":")
        ax.axvline(7.78 / 3.0, color=INK2, lw=0.8, ls=":")
        ax.set_title(f"ω_max = {wm}π/T")
        ax.set_xlabel("r0 / σ (distance au centre / largeur)")
    axs[0].set_ylabel("max de E(ω) dans le gap")
    axs[1].legend(fontsize=6, ncol=2)
    save(fig, FIG / "a5_gaussian_sigma_r0.png", dict(note="pointillés : géométrie du mémoire (r0/σ = 2.6) et pic du mémoire 9.3e-6"))
    sr.to_csv(RES / "a5_gaussian_sigma_r0.csv", index=False)


def part_noise():
    """Robustesse de la dichotomie (prédiction de la partie analytique, message 06:52) : bruit de pixel i.i.d. d'écart-type σ_n
    sur les vidéos d'entraînement (à P = ∞ : Σ → Σ + σ_n² I, C inchangé, i.e. ridge λ = σ_n²) et calcul en float32
    (SVD de Φ en simple précision, comme torch.linalg.pinv sur GPU). Convention « mémoire » : 2π/T, gap en tiers, rendu analytic4."""
    from aerot.paths import FIG
    from aerot.plotting import setup, save, plt, TEMPLATE_COLOR, INK2
    wm = 2
    wmax = wm * np.pi / T
    sig = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    res = {}
    for name in TEMPL:
        f = OUT / f"noise_{name}_wmax{wm}piT.json"
        if f.exists():
            res[name] = json.loads(f.read_text())
            continue
        om, w = train_quadrature("thirds", wmax, 2000)
        X = videos(om, name, "analytic", 4, 1.0)
        og = np.linspace(0, wmax, 601)
        Xg = videos(og, name, "analytic", 4, 1.0)
        a, b = gap_bounds("thirds", wmax)
        gap = (og > a) & (og < b)
        S = spectra.GramSpectrum(X, w, om)
        K = S.n_above(1e-15)
        U = S.eigvecs(K)
        Zg = Xg @ U
        r = dict(sigma_n=sig, gap_max={}, out_max={})
        for tau in [1e-4, 1e-8]:
            Kt = S.n_above(tau)
            E = (og - Zg[:, :Kt] @ (S.c_proj[:Kt] / S.lam[:Kt])) ** 2
            r[f"pinv{tau:.0e}"] = dict(gap_max=float(E[gap].max()), out_max=float(E[~gap].max()))
        for sn in sig[1:]:
            E = (og - Zg @ (S.c_proj[:K] / (S.lam[:K] + sn ** 2))) ** 2
            r["gap_max"][f"{sn:g}"] = float(E[gap].max())
            r["out_max"][f"{sn:g}"] = float(E[~gap].max())
        # float32 : pseudo-inverse par SVD de la matrice de données pondérée en simple précision (comme torch.linalg.pinv
        # sur GPU, rtol sur les valeurs singulières : rtol = 1e-2 ⇔ λ/λ₁ > 1e-4, rtol = 1e-4 ⇔ λ/λ₁ > 1e-8)
        A32 = (np.sqrt(w)[:, None] * X).astype(np.float32)
        y32 = (np.sqrt(w) * om).astype(np.float32)
        U32, s32, Vt32 = np.linalg.svd(A32, full_matrices=False)
        for tau in [1e-4, 1e-8]:
            keep = s32 > np.sqrt(tau) * s32[0]
            W32 = (Vt32[keep].T * (1.0 / s32[keep])) @ (U32[:, keep].T @ y32)
            E = (og - Xg @ W32.astype(np.float64)) ** 2
            r[f"float32_pinv{tau:.0e}"] = dict(gap_max=float(E[gap].max()), out_max=float(E[~gap].max()), K=int(keep.sum()))
        f.write_text(json.dumps(r))
        res[name] = r
        print(name, {k: v for k, v in r.items() if k.startswith(("pinv", "float32"))}, flush=True)
    setup()
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for name, r in res.items():
        col = TEMPLATE_COLOR.get(name, INK2)
        sn = np.array(sig[1:])
        ax.loglog(sn, [r["gap_max"][f"{s:g}"] for s in sig[1:]], "o-", color=col, ms=3.5, label=templates.LABELS_FR[name])
        ax.axhline(r["pinv1e-08"]["gap_max"], color=col, lw=0.7, ls=":")
        ax.plot([2e-7], [r["float32_pinv1e-08"]["gap_max"]], marker="*", ms=9, color=col, ls="none")
    ax.set_xlabel("écart-type du bruit de pixel σ_n (ridge équivalent λ = σ_n²)")
    ax.set_ylabel("max de E(ω) dans le gap")
    ax.set_title("2π/T : pointillés = pinv 1e-8 (float64) ; ★ = SVD float32, rtol 1e-4", fontsize=8)
    ax.legend(fontsize=7)
    save(fig, FIG / "a5_gap_noise.png", dict(wmax=wmax, sigma_n=sig, note="bruit à P = ∞ = ridge σ_n² ; données population n = 2000"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="conventions")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"conventions": part_conventions, "sigma_r0": part_sigma_r0, "report": part_report, "noise": part_noise}[a.part]()
