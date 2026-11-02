"""B2 — lois spectrales : exponentielle vs loi de puissance, contenu de Fourier angulaire, dépendance en (d, T, ω_max,
rendu), lien avec l'erreur dans le gap.

Parties (idempotentes ; résultats intermédiaires en json dans results/b2/) :
  --part fits     : ajustements de la queue λ_k/λ_1 ∈ [1e-8, 1e-2] (spectres exportés, rendu analytic ss=4, et modèle
                    continu d → ∞) par exp (a − b k), puissance (a − p log k), exp. étirée (a − b k^β) ; sélection par AIC.
  --part angular  : spectre de puissance angulaire E_m des templates (modèle continu), figure.
  --part scaling  : K(1e-4), K(1e-8), loi retenue et exposant en fonction de d, T, ω_max, rendu (ρ uniforme, n = 2000).
  --part link     : erreur dans le gap (résultats A5 : results/a5_variants/) vs descripteurs spectraux.
  --part all      : tout.
Usage : python scripts/b2_spectral_laws.py --part all
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
import scipy.special as sps

from aerot import continuum, render, spectra, templates
from aerot.paths import FIG, RES, SPECTRA
from aerot.plotting import setup, save, plt, TEMPLATE_COLOR, PALETTE, INK2

OUT = RES / "b2"
T0, D0 = 10, 28
MAIN = ["gaussian", "dipole", "quarter_circle", "cross", "segment", "arc"]
RMAX = {"gaussian": 19.8, "dipole": 19.8}   # autres : disque inscrit + bras de la croix (14.2)


# ------------------------------------------------------------------ ajustements de queue

def fit_tail(lam, lo=1e-8, hi=1e-2, kmax_frac=0.9):
    """Ajuste log10(λ_k/λ_1) sur la queue lo < λ_k/λ_1 < hi (défaut [1e-8, 1e-2] : évite la falaise de fin de spectre des
    rendus pixellisés et la saturation de rang de la quadrature) (k < kmax_frac × nombre de modes, pour éviter la saturation
    de rang de la quadrature). Renvoie les paramètres et l'AIC des trois modèles."""
    r = np.asarray(lam, float) / lam[0]
    k = np.arange(1, len(r) + 1)
    m = (r > lo) & (r < hi) & (k < kmax_frac * len(r))
    k, y = k[m], np.log10(r[m])
    n = len(k)
    if n < 8:
        return None
    out = dict(n=int(n), k_range=[int(k.min()), int(k.max())])

    def aic(res, p):
        return float(n * np.log(np.sum(res ** 2) / n) + 2 * p)

    A = np.vstack([np.ones(n), k]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    out["exp"] = dict(b=float(-c[1]), aic=aic(y - A @ c, 2))
    A = np.vstack([np.ones(n), np.log10(k)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    out["power"] = dict(p=float(-c[1]), aic=aic(y - A @ c, 2))
    f = lambda kk, a, b, be: a - b * kk ** be  # noqa: E731
    best = None
    for be0 in [0.3, 0.5, 1.0, 1.5]:
        try:
            p, _ = curve_fit(f, k, y, p0=[y[0], (y[0] - y[-1]) / (k[-1] ** be0 - k[0] ** be0 + 1e-9), be0],
                             bounds=([-np.inf, 0, 0.02], [np.inf, np.inf, 3.0]), maxfev=20000)
            a = aic(y - f(k, *p), 3)
            if best is None or a < best["aic"]:
                best = dict(b=float(p[1]), beta=float(p[2]), aic=a)
        except (RuntimeError, ValueError):
            pass
    out["stretched"] = best
    aics = {kk: out[kk]["aic"] for kk in ["exp", "power", "stretched"] if out.get(kk)}
    out["best"] = min(aics, key=aics.get)
    out["dAIC_exp_minus_power"] = out["exp"]["aic"] - out["power"]["aic"]
    return out


def continuum_spectrum(name, dist, wmax, T=T0, n=2000):
    Ek = continuum.angular_power(name, d=D0, r_max=RMAX.get(name, 14.2), nr=600, nth=2048, oversample=2)
    om, w = render.quadrature(dist, wmax, n)
    F = continuum.features(om, Ek, T=T, tol=1e-14)
    return spectra.GramSpectrum(F, w, om).lam, Ek


def part_fits():
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "fits.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    for name in MAIN:
        for dist in ["uniform", "creneau"]:
            for wl, wmax in [("2piT", 2 * np.pi / T0), ("4piT", 4 * np.pi / T0)]:
                key = f"{name}|{dist}|{wl}|pixel"
                if key not in res:
                    z = np.load(SPECTRA / f"{name}_{dist}_d28_T10_wmax{wmax:.4f}.npz")
                    res[key] = dict(fit=fit_tail(z["eigvals"]), K4=int(z["K_1e4"]), K8=int(z["K_1e8"]))
                key = f"{name}|{dist}|{wl}|continuum"
                if key not in res and dist == "uniform":
                    lam, _ = continuum_spectrum(name, dist, wmax)
                    # queue du modèle continu limitée à 1e-8 : au-delà, le rang de la quadrature (n = 2000) sature
                    res[key] = dict(fit=fit_tail(lam, lo=1e-8, kmax_frac=0.5), K4=int(np.sum(lam > 1e-4 * lam[0])),
                                    K8=int(np.sum(lam > 1e-8 * lam[0])))
                f.write_text(json.dumps(res))
                print(key, res.get(key, {}).get("fit", {}) and res[key]["fit"]["best"], flush=True)
    # tableau
    lines = ["# B2 — forme de la queue spectrale (λ_k/λ_1 ∈ [1e-8, 1e-2]) : exponentielle vs loi de puissance", "",
             "Modèles ajustés sur log10(λ_k/λ_1) : exp = a − b k ; puissance = a − p log10 k ; étirée = a − b k^β. "
             "ΔAIC = AIC(exp) − AIC(puissance) (> 0 : la loi de puissance est préférée). « pixel » : spectres exportés "
             "(rendu analytic ss=4, d = 28) ; « continuum » : modèle sans pixels (d → ∞, features exactes de continuum.py), "
             "restreint aux k < n/2 (saturation du rang de la quadrature n = 2000 au-delà).", "",
             "| template | ρ | ω_max | rendu | K(1e-4) | K(1e-8) | modèle retenu | b (exp) | p (puissance) | β (étirée) | ΔAIC exp−puis. |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, v in res.items():
        name, dist, wl, kind = key.split("|")
        ft = v["fit"]
        if ft is None:
            continue
        st = ft.get("stretched") or {}
        lines.append(f"| {name} | {dist} | {wl.replace('piT', 'π/T')} | {kind} | {v['K4']} | {v['K8']} | {ft['best']} | "
                     f"{ft['exp']['b']:.3g} | {ft['power']['p']:.3g} | {st.get('beta', float('nan')):.2f} | "
                     f"{ft['dAIC_exp_minus_power']:+.0f} |")
    (RES / "b2_fits.md").write_text("\n".join(lines) + "\n")
    return res


def part_angular():
    setup()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    data = {}
    for name in MAIN + ["disk"]:
        Ek = continuum.angular_power(name, d=D0, r_max=RMAX.get(name, 14.2), nr=600, nth=4096, oversample=2)
        data[name] = Ek.tolist()
        m = np.arange(len(Ek))
        rel = Ek / Ek[0]
        msk = (m > 0) & (rel > 1e-16)
        axs[0].loglog(m[msk], rel[msk], ".", ms=2.5, color=TEMPLATE_COLOR.get(name, INK2), label=templates.LABELS_FR[name])
        # énergie cumulée hors de la bande |m| ≤ M
        tail = 1 - (Ek[0] + 2 * np.cumsum(Ek[1:])) / (Ek[0] + 2 * Ek[1:].sum())
        axs[1].loglog(np.arange(1, len(Ek)), np.maximum(tail, 1e-17), color=TEMPLATE_COLOR.get(name, INK2), lw=1.2)
    s, r0 = 3.0, 8.0
    z0 = r0 ** 2 / (2 * s ** 2)
    mm = np.arange(1, 40)
    axs[0].loglog(mm, sps.ive(mm, z0) / sps.ive(0, z0), color=INK2, lw=0.8, ls="--", label="$I_m(z_0)/I_0(z_0)$ (Jacobi-Anger)")
    mm = np.arange(10, 1000)
    axs[0].loglog(mm, 0.5 * mm ** -2.0, color=INK2, lw=0.8, ls=":", label="$\\propto m^{-2}$")
    axs[0].set_ylim(1e-16, 2)
    axs[0].set_xlabel("harmonique angulaire m")
    axs[0].set_ylabel("$E_m / E_0$")
    axs[0].legend(fontsize=6, ncol=2)
    axs[0].set_title("Spectre de puissance angulaire (modèle continu)")
    axs[1].set_xlabel("M")
    axs[1].set_ylabel("énergie hors de |m| ≤ M")
    axs[1].set_ylim(1e-16, 1)
    axs[1].set_title("Queue d'énergie angulaire")
    save(fig, FIG / "b2_angular_power.png", dict(d=D0, note="E_m = ∫|a_m(r)|² r dr sur r ≤ r_max ; croix : m ≡ 0 mod 4 ; "
                                                 "dipôle : m pair ; disque : m = 0 seulement"))
    (OUT / "angular_power.json").write_text(json.dumps(data))


def spectrum_variant(name, d=D0, T=T0, wmax=4 * np.pi / T0, method="analytic", ss=4, n=2000, dist="uniform"):
    om, w = render.quadrature(dist, wmax, n)
    X = render.make_videos(om, name, d=d, T=T, method=method, ss=ss).numpy()
    return spectra.GramSpectrum(X, w, om).lam


def part_scaling():
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "scaling.json"
    (OUT / "spectra").mkdir(exist_ok=True)
    res = json.loads(f.read_text()) if f.exists() else {}
    for key in list(res):   # ré-ajustement à partir des spectres sauvegardés (si la plage d'ajustement a changé)
        sp = OUT / "spectra" / f"{key.replace('|', '_')}.npy"
        if sp.exists():
            res[key]["fit"] = fit_tail(np.load(sp))
        else:
            res.pop(key)
    variants = ([("d", dd, dict(d=dd)) for dd in [14, 20, 28, 42, 56]]
                + [("T", tt, dict(T=tt)) for tt in [5, 10, 20]]
                + [("wmax", wm, dict(wmax=wm * np.pi / T0)) for wm in [1, 2, 4]]
                + [("render", r, dict(method=r.split("_")[0], ss=int(r.split("_")[1])))
                   for r in ["analytic_1", "analytic_4", "analytic_8", "interp_1"]])
    for name in ["gaussian", "quarter_circle", "cross", "segment"]:
        for fac, val, kw in variants:
            key = f"{name}|{fac}|{val}"
            if key in res:
                continue
            if fac == "wmax":
                kw = dict(kw)
            lam = spectrum_variant(name, **kw)
            np.save(OUT / "spectra" / f"{key.replace('|', '_')}.npy", lam)
            ft = fit_tail(lam)
            res[key] = dict(K4=int(np.sum(lam > 1e-4 * lam[0])), K8=int(np.sum(lam > 1e-8 * lam[0])),
                            K12=int(np.sum(lam > 1e-12 * lam[0])), lam1=float(lam[0]), fit=ft,
                            n999=int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.999) + 1))
            f.write_text(json.dumps(res))
            print(key, res[key]["K4"], res[key]["K8"], ft and ft["best"], flush=True)
    setup()
    facs = [("d", "résolution d (px)", [14, 20, 28, 42, 56]), ("T", "nombre d'images T", [5, 10, 20]),
            ("wmax", "ω_max (× π/T)", [1, 2, 4])]
    fig, axs = plt.subplots(2, 4, figsize=(14, 6))
    for j, (fac, lab, vals) in enumerate(facs):
        for name in ["gaussian", "quarter_circle", "cross", "segment"]:
            k4 = [res[f"{name}|{fac}|{v}"]["K4"] for v in vals]
            k8 = [res[f"{name}|{fac}|{v}"]["K8"] for v in vals]
            axs[0, j].plot(vals, k4, "o-", color=TEMPLATE_COLOR[name], label=templates.LABELS_FR[name])
            axs[1, j].plot(vals, k8, "s--", color=TEMPLATE_COLOR[name])
        axs[0, j].set_xlabel(lab)
        axs[1, j].set_xlabel(lab)
        axs[0, j].set_yscale("log")
        axs[1, j].set_yscale("log")
    rl = ["analytic_1", "analytic_4", "analytic_8", "interp_1"]
    for name in ["gaussian", "quarter_circle", "cross", "segment"]:
        axs[0, 3].plot(range(4), [res[f"{name}|render|{r}"]["K4"] for r in rl], "o-", color=TEMPLATE_COLOR[name])
        axs[1, 3].plot(range(4), [res[f"{name}|render|{r}"]["K8"] for r in rl], "s--", color=TEMPLATE_COLOR[name])
    for i in range(2):
        axs[i, 3].set_xticks(range(4))
        axs[i, 3].set_xticklabels(["pt", "ss4", "ss8", "interp"])
        axs[i, 3].set_yscale("log")
    axs[0, 0].set_ylabel("K(1e-4)")
    axs[1, 0].set_ylabel("K(1e-8)")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("B2 — nombre de modes vs résolution, durée, vitesse maximale et rendu (ρ uniforme ; défauts d = 28, T = 10, "
                 "ω_max = 4π/T, analytic ss=4)", fontsize=9)
    save(fig, FIG / "b2_scaling.png", dict(variants=[(a, b) for a, b, _ in variants]))
    return res


def part_link():
    """Erreur dans le gap (A5) vs descripteurs spectraux, sur toutes les configurations A5 (rendu analytic4, rad, tiers)."""
    setup()
    rows = []
    for fpath in glob.glob(str(RES / "a5_variants" / "*.json")):
        r = json.load(open(fpath))
        if r["method"] != "analytic" or r["ss"] != 4 or r["gapdef"] != "thirds" or r["unit"] != 1.0:
            continue
        p = r["params"]
        if r["template"] == "gaussian" and p.get("r0", 1) == 0:
            continue
        rows.append(dict(template=r["template"], wmax=r["wmax"], K4=r["K"]["1e-04"], K8=r["K"]["1e-08"], n999=r["n_999"],
                         e4=r["regs"]["pinv1e-04"]["gap_max"], e8=r["regs"]["pinv1e-08"]["gap_max"],
                         sweep=Path(fpath).name.startswith("sr_"), b=(p.get("r0", np.nan) / p["sigma"]) if "sigma" in p else np.nan))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    out = {}
    for ax, (xk, xl) in zip(axs, [("K4", "K(1e-4) (modes au-dessus du seuil)"), ("n999", "modes pour 99.9 % de l'énergie")]):
        for q, mk, lab in [("e4", "o", "pinv 1e-4"), ("e8", "s", "pinv 1e-8")]:
            x = np.array([r[xk] for r in rows], float)
            y = np.array([r[q] for r in rows], float)
            col = [TEMPLATE_COLOR.get(r["template"], INK2) for r in rows]
            ax.scatter(x, y, c=col, marker=mk, s=[10 if r["sweep"] else 40 for r in rows],
                       facecolors=col if q == "e8" else "none", edgecolors=col, lw=0.8)
            rho = spearmanr(np.log(x), np.log(y))[0]
            out[f"{xk}|{q}"] = float(rho)
            ax.plot([], [], mk, color=INK2, mfc="none" if q == "e4" else INK2, label=f"{lab} (Spearman ρ = {rho:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("max de E(ω) dans le gap")
    fig.suptitle("B2 — erreur dans le gap vs taille effective du spectre (gros symboles : 5 templates × 2 ω_max ; "
                 "petits : balayage (σ, r0) de la gaussienne)", fontsize=8)
    save(fig, FIG / "b2_gap_vs_spectrum.png", dict(spearman=out, n=len(rows)))
    (OUT / "link.json").write_text(json.dumps(dict(spearman=out, rows=rows)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = ["fits", "angular", "scaling", "link"] if a.part == "all" else a.part.split(",")
    for p in parts:
        {"fits": part_fits, "angular": part_angular, "scaling": part_scaling, "link": part_link}[p]()
