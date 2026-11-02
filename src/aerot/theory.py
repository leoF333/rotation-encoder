"""Replica-theory formulas for the linear encoder ω̂ = W·φ (ridge regression, in- and out-of-distribution).

Dérivations : `theory/derivations/0*.md`. Tests : `theory/checks/test_theory.py`.

CONVENTIONS (identiques à `docs/CONVENTIONS.md`)
---------------------------------------------------
* `eigvals` : valeurs propres λ_k du second moment NON centré Σ = ⟨φφᵀ⟩ (ordre quelconque, λ_k ≤ 0 ignorées).
* `c_proj`  : projections c_k = u_k · C, C = ⟨ω φ⟩ (non centré), dans la même base propre.
* `omega2`  : ⟨ω²⟩ (second moment NON centré de la cible).
* `P`       : nombre d'échantillons d'entraînement (réel > 0 accepté). α = P/N n'intervient jamais seul :
              les formules ne dépendent de N qu'à travers le spectre (nombre de modes).
* `lam_r`   : ridge λ_r du Hamiltonien H(W) = ½ Σ_μ (ω_μ − W·φ_μ)² + (λ_r/2)‖W‖²   (PAS de 1/P devant la somme).
              Le ridge se compare aux valeurs propres via λ_r/P ; limite thermodynamique non triviale : λ_r ∝ P.
* `beta`    : température inverse (défaut ∞ = estimateur ridge, T = 0).

Erreurs renvoyées (toutes : par échantillon, SANS facteur ½, SANS terme ridge) :
* E_train = (1/P) Σ_μ (ω_μ − W*·φ_μ)²                         moyenné sur les jeux d'entraînement ;
* E_gen   = ⟨(ω − W*·φ)²⟩ sur un nouvel échantillon de la même loi, moyenné sur les jeux d'entraînement ;
* E_OOD   = idem avec ω ~ ρ_test (voir `ood_error`).

Modèle : équivalent gaussien centré de mêmes seconds moments, ω = θ*·φ + ε avec θ*_k = c_k/λ_k,
ε ~ N(0, σ²_res) indépendant de φ, σ²_res = ⟨ω²⟩ − Σ_k c_k²/λ_k (résidu du meilleur prédicteur linéaire sans biais,
inclut la puissance des modes non fournis si le spectre est tronqué).
Résultat central (T2) : κ (1 − df(κ)/P) = λ_r/P, df(κ) = Σ_k λ_k/(λ_k+κ),
    E_gen = (σ²_res + B0)/(1 − γ),  E_train = (1 − df/P)² E_gen,
    γ = (1/P) Σ_k λ_k²/(λ_k+κ)²,  B0 = κ² Σ_k t_k/(λ_k+κ)²,  t_k = c_k²/λ_k.
Hors distribution (T4, `theory/derivations/04_generalisation_gap.md`) :
    E_OOD = E_test(W̄_κ) + (E_gen/P) Σ_k (Σ_test)_kk λ_k/(λ_k+κ)²,   W̄_κ,k = c_k/(λ_k+κ)   (`ood_error`) ;
    à P = ∞ et pinv tronquée : E_OOD = E_test^min + Σ_j (μ_j+2)²/μ_j ⟨ε_u, g_j⟩²_train   (`runge_decomposition`).

Historique : v0.1 (T1–T3) ; v0.2 (T4 : ood_error, ood_error_truncated, pointwise_error, generalized_ratio_spectrum,
runge_decomposition).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar

__version__ = "0.2"

# ----------------------------------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------------------------------

def _clean(eigvals, c_proj=None, eig_tol=0.0):
    """Garde les modes λ_k > eig_tol·max(λ). Renvoie (λ, c) en float64."""
    lam = np.asarray(eigvals, dtype=float).ravel()
    keep = lam > max(eig_tol * lam.max(), 0.0)
    if c_proj is None:
        return lam[keep], None
    c = np.asarray(c_proj, dtype=float).ravel()
    if c.shape != lam.shape:
        raise ValueError("eigvals et c_proj doivent avoir la même taille")
    return lam[keep], c[keep]


def teacher_power(eigvals, c_proj, eig_tol=0.0):
    """Puissance du professeur par mode t_k = c_k²/λ_k = λ_k θ*_k² (modes λ_k > eig_tol·λ_max)."""
    lam, c = _clean(eigvals, c_proj, eig_tol)
    return lam, c**2 / lam


def residual_variance(eigvals, c_proj, omega2, eig_tol=0.0):
    """σ²_res = ⟨ω²⟩ − Σ_k c_k²/λ_k : erreur de population du meilleur prédicteur linéaire sans biais
    (restreint aux modes conservés). Peut être ≈ 0 (voire légèrement < 0 par erreurs d'arrondi) si ω est
    une fonction presque linéaire de φ ; on renvoie max(·, 0)."""
    _, t = teacher_power(eigvals, c_proj, eig_tol)
    return max(float(omega2) - t.sum(), 0.0)


def df_kappa(eigvals, kappa):
    """Degrés de liberté effectifs df(κ) = Σ_k λ_k/(λ_k+κ)."""
    lam = np.asarray(eigvals, float)
    return float(np.sum(lam / (lam + kappa)))


# ----------------------------------------------------------------------------------------------
# Équation auto-cohérente
# ----------------------------------------------------------------------------------------------

def solve_kappa(eigvals, P, lam_r, eig_tol=0.0):
    """Résout κ (1 − df(κ)/P) = λ_r/P  (κ ≥ 0, unique).

    * λ_r > 0 : racine unique dans ]λ_r/(2P), λ_r/P + Σλ/P].
    * λ_r = 0 (ridgeless) : κ = 0 si P ≥ rang (nombre de modes λ_k>0), sinon κ0 > 0 tel que df(κ0) = P
      (interpolateur de norme minimale).
    κ a la dimension d'une valeur propre : les modes λ_k ≫ κ sont appris, λ_k ≪ κ ignorés.
    """
    lam, _ = _clean(eigvals, None, eig_tol)
    P = float(P)
    if lam_r < 0:
        raise ValueError("lam_r doit être ≥ 0")
    if lam_r == 0.0:
        if P >= lam.size:
            return 0.0
        f = lambda lk: np.sum(lam / (lam + np.exp(lk))) - P
        lo, hi = np.log(lam.min()) - 40.0, np.log(lam.sum() / P) + 1.0   # df(lo) ≈ rang > P, df(hi) < P
        return float(np.exp(brentq(f, lo, hi, xtol=1e-13, rtol=1e-13, maxiter=500)))
    a = lam_r / P
    h = lambda lk: np.exp(lk) * (1.0 - np.sum(lam / (lam + np.exp(lk))) / P) - a
    lo, hi = np.log(0.5 * a), np.log(a + lam.sum() / P) + 1e-9
    return float(np.exp(brentq(h, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)))


# ----------------------------------------------------------------------------------------------
# Paramètres d'ordre et erreurs en distribution (T2, T3)
# ----------------------------------------------------------------------------------------------

def order_parameters(eigvals, c_proj, omega2, P, lam_r, beta=np.inf, eig_tol=0.0):
    """Solution RS complète (modèle gaussien centré, ridge λ_r, P échantillons, température 1/β).

    Renvoie un dict :
      kappa   : échelle spectrale auto-cohérente κ
      df      : Σ λ/(λ+κ) (nombre effectif de modes appris) ;  df2 = Σ λ²/(λ+κ)² ;  gamma = df2/P
      chi     : β·δQ = df/(P−df) (susceptibilité ; δQ = χ/β = largeur thermique, >0 pour tout α si λ_r>0)
      sigma2  : σ²_res
      B0      : κ² Σ t_k/(λ_k+κ)²  (biais « brut »)
      Q0      : (γσ² + B0)/(1−γ) = w Σ w au col (erreur en excès à T=0)
      learnability : L_k = λ_k/(λ_k+κ) (⟨W_k⟩ = L_k θ*_k)
      Egen, Etrain : erreurs (conventions du module ; + corrections thermiques si β fini)
      bias, variance : décomposition E_gen = σ² + bias + variance (T=0), bias = B0/(1−γ), variance = γ(σ²+B0)/(1−γ)
      bias_modes, var_modes : contributions par mode à l'excès : κ² t_k/(λ_k+κ)² et L_k² E_gen/P
                              (Σ bias_modes + Σ var_modes = E_gen − σ²)
    """
    lam, t = teacher_power(eigvals, c_proj, eig_tol)
    s2 = max(float(omega2) - t.sum(), 0.0)
    P = float(P)
    kap = solve_kappa(lam, P, lam_r)
    L = lam / (lam + kap)
    df = float(L.sum())
    df2 = float(np.sum(L**2))
    gam = df2 / P
    B0 = float(kap**2 * np.sum(t / (lam + kap) ** 2)) if kap > 0 else 0.0
    # γ ≥ 1 : pic d'interpolation exact (λ_r = 0, P = rang)
    Eg = (s2 + B0) / (1.0 - gam) if gam < 1.0 else np.inf
    chi = df / (P - df) if P > df else np.inf
    Et = (1.0 - df / P) ** 2 * Eg if np.isfinite(Eg) else np.nan
    out = dict(kappa=kap, df=df, df2=df2, gamma=gam, chi=chi, sigma2=s2, B0=B0,
               Q0=Eg - s2, learnability=L, Egen=Eg, Etrain=Et,
               bias=B0 / (1 - gam) if gam < 1 else np.inf,
               variance=gam * (s2 + B0) / (1 - gam) if gam < 1 else np.inf,
               bias_modes=(kap**2 * t / (lam + kap) ** 2), var_modes=L**2 * Eg / P,
               eigvals=lam, t=t)
    if np.isfinite(beta):
        out["deltaQ"] = chi / beta
        out["Egen"] = Eg + chi / beta
        out["Etrain"] = Et + chi / (beta * (1.0 + chi))
    else:
        out["deltaQ"] = 0.0
    return out


def gen_error(eigvals, c_proj, omega2, P, lam_r, beta=np.inf, eig_tol=0.0):
    """E_gen (par échantillon, sans ½) — voir `order_parameters`."""
    return order_parameters(eigvals, c_proj, omega2, P, lam_r, beta, eig_tol)["Egen"]


def train_error(eigvals, c_proj, omega2, P, lam_r, beta=np.inf, eig_tol=0.0):
    """E_train (par échantillon, sans ½, sans terme ridge) = (1 − df/P)² E_gen à T=0 — voir `order_parameters`."""
    return order_parameters(eigvals, c_proj, omega2, P, lam_r, beta, eig_tol)["Etrain"]


def learning_curve(eigvals, c_proj, omega2, P_list, lam_r, which="Egen", eig_tol=0.0):
    """Tableau de E_gen (ou 'Etrain', 'kappa', 'df', …) pour une liste de P, λ_r fixé."""
    return np.array([order_parameters(eigvals, c_proj, omega2, P, lam_r, eig_tol=eig_tol)[which] for P in P_list])


def optimal_ridge(eigvals, c_proj, omega2, P, eig_tol=0.0, bounds=(-12.0, 6.0)):
    """λ_r optimal (minimisant E_gen) et E_gen correspondant. Recherche en log10(λ_r/(P λ_max)) ∈ bounds.
    Résultat connu (T3) : pour un professeur « isotrope » (t_k = τ² λ_k) l'optimum est λ_r = σ²/τ² = σ² Σλ_k/Σt_k,
    indépendant de P (estimateur bayésien)."""
    lam, _ = _clean(eigvals, None, eig_tol)
    scale = P * lam.max()
    f = lambda x: gen_error(eigvals, c_proj, omega2, P, scale * 10.0**x, eig_tol=eig_tol)
    r = minimize_scalar(f, bounds=bounds, method="bounded", options=dict(xatol=1e-4))
    return scale * 10.0**r.x, float(r.fun)


def free_energy_rs(eigvals, c_proj, omega2, P, lam_r, beta):
    """⟨ln Z⟩ RS (constantes incluses, mesure de Lebesgue dW) pour le Hamiltonien du module, et l'entropie
    S = β⟨H⟩ + ⟨ln Z⟩. Vérifié contre le calcul exact (Gibbs gaussien) dans `theory/checks/t2_ridge_rs_check.py`.
    Renvoie dict(lnZ, S) (extensifs, pas divisés par N)."""
    lam, t = teacher_power(eigvals, c_proj)
    theta2 = t / lam
    op = order_parameters(eigvals, c_proj, omega2, P, lam_r)
    s2, chi, Q0 = op["sigma2"], op["chi"], op["Q0"]
    dQ = chi / beta
    R = P * beta / (1 + chi)
    qh = P * beta**2 * (Q0 + s2) / (1 + chi) ** 2
    a = lam * R + beta * lam_r
    G = (0.5 * lam.size * np.log(2 * np.pi) + 0.5 * (R * (dQ + Q0) - qh * dQ)
         - 0.5 * P * (np.log1p(beta * dQ) + beta * (Q0 + s2) / (1 + beta * dQ))
         + np.sum(-0.5 * np.log(a) + 0.5 * (lam * qh + beta**2 * lam_r**2 * theta2) / a - 0.5 * beta * lam_r * theta2))
    logdet = np.sum(np.log(lam_r + P * lam / (1 + chi))) + P * np.log1p(chi) - P * chi / (1 + chi)
    S = 0.5 * lam.size * (1 + np.log(2 * np.pi / beta)) - 0.5 * logdet
    return dict(lnZ=float(G), S=float(S))


# ----------------------------------------------------------------------------------------------
# Formules du mémoire (ridge nul, α > 1), originales et corrigées (T1)
# ----------------------------------------------------------------------------------------------

def memoire_ridgeless(Jcc, Jphic, Jphiphi, wbar, sigma2, alpha, corrected=True):
    """Résultats RS du mémoire (annexe p. 28–29), notations du mémoire :
    J_xy = x^T M^{-1} y avec M la covariance CENTRÉE (correction E1), wbar = ω̄, sigma2 = Var(ω).
    corrected=True : m* = J_φc(J_φc − ω̄)/(1+J_φφ) − J_cc (correction E2) ; False : formule du mémoire (+ω̄).
    Renvoie dict(qtilde, m, c, X, Etrain, Egen, e_memoire) avec
      X = q̃² + 2m + σ² + c (= σ²_res si corrected), Etrain = (1−1/α) X (par échantillon, sans ½),
      Egen = α X/(α−1) (= q0* + q̃²), e_memoire = (α−1) X / 2 (= ⟨ε⟩ du mémoire = ⟨H⟩/N à T=0).
    """
    qt = (wbar - Jphic) / (1 + Jphiphi)
    sgn = -1.0 if corrected else 1.0
    m = Jphic * (Jphic + sgn * wbar) / (1 + Jphiphi) - Jcc
    wt = qt - wbar
    c = (Jphiphi * m**2 - 2 * Jphic * m * wt + Jcc * wt**2) / (Jcc * Jphiphi - Jphic**2)
    X = qt**2 + 2 * m + sigma2 + c
    return dict(qtilde=qt, m=m, c=c, X=X, Etrain=(1 - 1 / alpha) * X, Egen=alpha * X / (alpha - 1),
                e_memoire=(alpha - 1) * X / 2)


# ----------------------------------------------------------------------------------------------
# Hors distribution (T4) : entraînement sur ρ_train, test sur ρ_test (changement de covariables)
# ----------------------------------------------------------------------------------------------

def _train_basis_arrays(eigvals, c_proj, K):
    lam = np.asarray(eigvals, float).ravel()[:K]
    c = np.asarray(c_proj, float).ravel()[:K]
    if np.any(np.diff(lam) > 1e-12 * abs(lam[0])):
        raise ValueError("eigvals doit être trié par ordre décroissant (alignement avec test_cov)")
    return lam, c


def ridge_mean_predictor(eigvals, c_proj, kappa, K=None):
    """Prédicteur moyen W̄_κ dans la base propre d'entraînement : W̄_k = c_k/(λ_k+κ) (λ_k > 0), 0 sinon.
    C'est l'estimateur ridge « population » de régularisation effective κ (= E[W*] dans le modèle gaussien)."""
    K = len(eigvals) if K is None else K
    lam, c = _train_basis_arrays(eigvals, c_proj, K)
    W = np.zeros_like(lam)
    pos = lam + kappa > 0
    W[pos] = c[pos] / (lam[pos] + kappa)
    if kappa == 0:
        W[lam <= 0] = 0.0
    return W


def ood_error(eigvals, c_proj, omega2, test_cov, test_c_proj, test_omega2, P, lam_r, beta=np.inf,
              return_parts=False):
    """E_OOD = ⟨(ω − W*·φ)²⟩_{ρ_test}, moyenné sur les jeux d'entraînement (P échantillons de ρ_train, ridge λ_r).

    Formule (répliques, `theory/derivations/04_generalisation_gap.md`) :
        E_OOD = E_test(W̄_κ) + (E_gen/P) Σ_k (Σ_test)_kk λ_k/(λ_k+κ)²  [+ (1+χ)/(βP) Σ_k (Σ_test)_kk/(λ_k+κ) si β fini]
    avec κ, E_gen (T=0) de la distribution d'ENTRAÎNEMENT, W̄_κ,k = c_k/(λ_k+κ), et
        E_test(W) = test_omega2 − 2 test_c_proj·W + Wᵀ test_cov W.
    Entrées (base propre d'ENTRAÎNEMENT, ordre décroissant des λ_k, conventions du module) :
      eigvals, c_proj, omega2 : spectre/projections/⟨ω²⟩ d'entraînement (tous les modes) ;
      test_cov  : (K×K) u_kᵀ Σ_test u_l (Σ_test non centré) ; test_c_proj : u_k·C_test ; test_omega2 : ⟨ω²⟩_test.
      Les modes k ≥ K (absents de test_cov) sont ignorés côté test (énergie test manquante : voir test_trace).
      ATTENTION : si ces modes sont appris (κ ≲ λ_K), E_OOD est biaisé (constaté numériquement : segment, ridge
      1e-8, théorie 2e-2 au lieu de 2.5e-3) ; un avertissement est émis quand λ_K/(λ_K+κ) > 1e-2. Utiliser alors une matrice
      test sur tous les modes non nuls (cache `results/b1_theory_cache/`).
    Hypothèse : données d'entraînement ≈ gaussiennes (équivalent de mêmes seconds moments) ; AUCUNE hypothèse
    sur ρ_test au-delà de ses seconds moments (E_OOD est quadratique en W).
    return_parts=True : dict(Eood, bias=E_test(W̄_κ), variance, thermal, kappa, Egen_train).
    """
    test_cov = np.asarray(test_cov, float)
    K = test_cov.shape[0]
    op = order_parameters(eigvals, c_proj, omega2, P, lam_r)
    kap = op["kappa"]
    Eg0 = op["sigma2"] + op["Q0"]
    lam, c = _train_basis_arrays(eigvals, c_proj, K)
    lam_all = np.asarray(eigvals, float).ravel()
    if lam_all.size > K and lam_all[K] > 0 and lam_all[K] / (lam_all[K] + kap) > 1e-2:
        import warnings
        warnings.warn(f"ood_error : les modes k ≥ {K} (absents de test_cov) sont appris (L_K = {lam_all[K]/(lam_all[K]+kap):.2g}) ; "
                      "E_OOD peut être biaisé — fournir test_cov sur plus de modes.", RuntimeWarning, stacklevel=2)
    W = ridge_mean_predictor(eigvals, c_proj, kap, K)
    ct = np.asarray(test_c_proj, float).ravel()[:K]
    bias = float(test_omega2 - 2 * ct @ W + W @ test_cov @ W)
    dg = np.diag(test_cov)
    lp = np.clip(lam, 0, None)
    var = float(Eg0 / P * np.sum(dg * lp / (lp + kap) ** 2)) if np.isfinite(Eg0) else np.inf
    thermal = 0.0
    if np.isfinite(beta):
        thermal = float((1 + op["chi"]) / (beta * P) * np.sum(dg / (lp + kap)))
    E = bias + var + thermal
    if return_parts:
        return dict(Eood=E, bias=bias, variance=var, thermal=thermal, kappa=kap, Egen_train=Eg0)
    return E


def ood_error_truncated(eigvals, c_proj, test_cov, test_c_proj, test_omega2, tau=1e-4):
    """Procédure du mémoire à P = ∞ : W_k = c_k/λ_k pour λ_k > τ λ_1 (pinv tronquée), 0 sinon ;
    renvoie E_test(W). Reproduit `ref_Eood_pop_trunc1e-4` des fichiers de data/spectra pour tau=1e-4."""
    test_cov = np.asarray(test_cov, float)
    K = test_cov.shape[0]
    lam, c = _train_basis_arrays(eigvals, c_proj, K)
    keep = lam > tau * lam[0]
    W = np.where(keep, c / np.where(keep, lam, 1.0), 0.0)
    ct = np.asarray(test_c_proj, float).ravel()[:K]
    return float(test_omega2 - 2 * ct @ W + W @ test_cov @ W)


def pointwise_error(eigvals, c_proj, omega2, z_test, omega_test, P, lam_r):
    """Erreur ponctuelle moyenne E(ω) = ⟨(ω − W*·φ(ω))²⟩_jeux  (courbe E(ω) du mémoire), pour des points de test donnés.
    z_test : (n_test × K) projections u_k·φ(ω) des points de test sur la base propre d'entraînement (K premiers modes) ;
    omega_test : (n_test,) valeurs de ω.
        E(ω) = (ω − W̄_κ·z(ω))² + (E_gen/P) Σ_k z_k(ω)² λ_k/(λ_k+κ)²."""
    z = np.atleast_2d(np.asarray(z_test, float))
    K = z.shape[1]
    op = order_parameters(eigvals, c_proj, omega2, P, lam_r)
    kap = op["kappa"]; Eg0 = op["sigma2"] + op["Q0"]
    lam, _ = _train_basis_arrays(eigvals, c_proj, K)
    W = ridge_mean_predictor(eigvals, c_proj, kap, K)
    lp = np.clip(lam, 0, None)
    return (np.asarray(omega_test, float) - z @ W) ** 2 + Eg0 / P * (z**2 @ (lp / (lp + kap) ** 2))


def generalized_ratio_spectrum(eigvals, test_cov, eig_tol=1e-14):
    """Valeurs propres généralisées μ du faisceau (Σ_test, Σ_train) restreint aux K modes de test_cov :
    Σ_test v = μ Σ_train v. μ ≫ 1 : directions « concentrées dans le gap » (peu d'énergie d'entraînement,
    beaucoup d'énergie test) — analogues des fonctions de Slepian. Renvoie μ par ordre décroissant."""
    test_cov = np.asarray(test_cov, float)
    K = test_cov.shape[0]
    lam = np.asarray(eigvals, float)[:K]
    keep = lam > eig_tol * lam[0]
    s = 1 / np.sqrt(lam[keep])
    M = test_cov[np.ix_(keep, keep)] * s[:, None] * s[None, :]
    return np.sort(np.linalg.eigvalsh(0.5 * (M + M.T)))[::-1]


def runge_decomposition(eigvals, c_proj, omega2, test_cov, test_c_proj, test_omega2, K=None, w_train=2 / 3):
    """Décomposition « de Runge » de l'erreur dans le gap pour la pinv tronquée aux K premiers modes d'entraînement (P = ∞).

    Dans F_K = vect{u_k·φ, k < K} : fonctions propres généralisées g_j = v_j·φ, Σ_test v_j = μ_j Σ_train v_j, v_jᵀΣ_train v_j = 1.
    Référence : meilleur prédicteur f_u sur ρ_u = w_train ρ_train + (1−w_train) ρ_test (loi sans gap ; w_train = 2/3 pour le
    créneau du mémoire), résidu ε_u = ω − f_u. Identité exacte (04 §4) :
        E_OOD(K) = E_test^min(K) + Σ_j (μ_j + ρ)²/μ_j · e_j²,   ρ = w_train/(1−w_train),   e_j = ⟨ε_u, g_j⟩_train,
    avec Σ_j e_j² ≤ ‖ε_u‖²_train (borne de type Lebesgue : E_OOD ≲ E_test^min + μ_max ‖ε_u‖²_train).
    Entrées : base propre d'ENTRAÎNEMENT (comme `ood_error`) ; K ≤ taille de test_cov (défaut : tous les modes λ_k > 1e-13 λ_1).
    Renvoie dict(Eood, Emin, shift, mu, e2, eps_u2 (sur ρ_u), eps_tr2 (sur ρ_train), mu_max, n_mu_gt1).
    Attention : test_cov n'est précis qu'à ~1e-8 relatif ; les μ_j ≲ 1e-8 sont du bruit numérique (sans effet sur Eood).
    """
    from scipy.linalg import eigh
    S = np.asarray(test_cov, float)
    lam = np.asarray(eigvals, float).ravel()
    c = np.asarray(c_proj, float).ravel()
    ct = np.asarray(test_c_proj, float).ravel()
    if K is None:
        K = int(np.sum(lam[:S.shape[0]] > 1e-13 * lam[0]))
    L = lam[:K]; St = S[:K, :K]; Str = np.diag(L); cK = c[:K]; ctK = ct[:K]
    wt = 1.0 - w_train
    Su = w_train * Str + wt * St
    Cu = w_train * cK + wt * ctK
    o2u = w_train * omega2 + wt * test_omega2
    thu = np.linalg.solve(Su, Cu)
    eps_u2 = float(o2u - Cu @ thu)
    eps_tr2 = float(omega2 - 2 * cK @ thu + thu @ Str @ thu)
    mu, V = eigh(St, Str)
    e = V.T @ cK - V.T @ Str @ thu
    rho = w_train / wt
    shift = float(np.sum((mu + rho) ** 2 / mu * e**2))
    W = cK / L
    Eood = float(test_omega2 - 2 * ctK @ W + W @ St @ W)
    Emin = float(test_omega2 - ctK @ np.linalg.lstsq(St, ctK, rcond=1e-10)[0])
    return dict(Eood=Eood, Emin=Emin, shift=shift, mu=mu, e2=e**2, eps_u2=eps_u2, eps_tr2=eps_tr2,
                mu_max=float(mu.max()), n_mu_gt1=int(np.sum(mu > 1)))
