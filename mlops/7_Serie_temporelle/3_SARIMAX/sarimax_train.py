# mlops/7_Serie_temporelle/3_SARIMAX/sarimax_train.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import itertools
import warnings
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import mlflow
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, normal_ad
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.base.tsa_model import ValueWarning
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
import joblib

# ───────────────────────── Warnings ─────────────────────────
warnings.filterwarnings("ignore", category=ValueWarning,
                        message="No frequency information was provided.*")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge.*")
warnings.filterwarnings("ignore", message="Too few observations to estimate starting parameters.*")

# ─────────────────────── Utils de préparation ───────────────────────
def filter_exog_by_corr(y: pd.Series, exog_df: pd.DataFrame, threshold=0.05) -> pd.DataFrame:
    if exog_df is None or exog_df.empty:
        return exog_df
    corr_values = exog_df.corrwith(y).abs()
    keep = corr_values[corr_values > threshold].index
    return exog_df[keep]

def apply_lag_if_correlated(df: pd.DataFrame, target="prix_m2_vente", threshold=0.3) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    lagged_cols = []
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        if s.isna().any():
            continue
        if s.nunique(dropna=True) < 2:
            continue
        corr = s.corr(df[target])
        if pd.notna(corr) and abs(corr) >= threshold:
            df[col] = s.shift(1)
            lagged_cols.append(col)
    if lagged_cols:
        df = df.dropna(subset=lagged_cols + [target])
    return df, lagged_cols

def adf_significant(series: pd.Series, cutoff=0.05) -> bool:
    series = series.dropna()
    if len(series) < 15:
        return False
    p_value = adfuller(series, autolag='AIC')[1]
    return p_value <= cutoff

def get_diff_order(series: pd.Series, seasonal=False, s=12, max_d=2, cutoff=0.05) -> int:
    for d in range(0, max_d + 1):
        y_diff = series.copy()
        for _ in range(d):
            y_diff = y_diff.diff(s if seasonal else 1)
        if adf_significant(y_diff, cutoff=cutoff):
            return d
    return max_d

def get_valid_seasonal_orders(D: int, s=12, max_P=2, max_Q=2):
    """Si s<=1 → pas de saisonnalité."""
    if s is None or s <= 1:
        return [(0, 0, 0, 0)]
    return [(P, D, Q, s) for P in range(max_P + 1) for Q in range(max_Q + 1)]

def _to_month_start(idx):
    return pd.to_datetime(idx, errors="coerce").to_period("M").to_timestamp()

def prepare_monthly_index(y: pd.Series,
                          exog: Optional[pd.DataFrame] = None,
                          min_frac_non_nan: float = 0.9):
    """
    Retourne y_m (freq 'MS'), exog_m (standardisée) et scaler.
    - Aligne au 1er de chaque mois
    - Recrée l'index mensuel complet
    - Ffill pour trous
    """
    y = y.copy()
    y.index = _to_month_start(y.index)
    y = y.sort_index()

    non_nan_mask = y.notna()
    if non_nan_mask.mean() < min_frac_non_nan:
        first_valid = non_nan_mask[non_nan_mask].index.min()
        y = y.loc[first_valid:]

    full_idx = pd.date_range(y.index.min(), y.index.max(), freq="MS")
    y = y.reindex(full_idx).ffill()

    scaler = None
    ex_scaled = None
    if exog is not None and not exog.empty:
        ex = exog.copy()
        ex.index = _to_month_start(ex.index)
        ex = ex.reindex(full_idx).ffill()
        scaler = StandardScaler()
        ex_scaled = pd.DataFrame(
            scaler.fit_transform(ex.values),
            index=ex.index, columns=ex.columns
        )

    return y.asfreq("MS"), (ex_scaled.asfreq("MS") if ex_scaled is not None else None), scaler, full_idx

# ─────────────────── Fit robuste avec fallback ───────────────────
def fit_sarimax_robust(y, exog, order, seasonal_order, trend="c", maxiter=100, start_params=None):
#initialement maxiter à 800 mais ca mouline
    # concentrate_scale=True aide souvent la convergence ; simple_differencing accélère
    model = sm.tsa.SARIMAX(
        y, exog=exog, order=order, seasonal_order=seasonal_order, trend=trend,
        enforce_stationarity=False, enforce_invertibility=False,
        simple_differencing=True, concentrate_scale=True
    )

    def _quiet_fit(method, iters, **kw):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge.*")
            warnings.filterwarnings("ignore", message="Too few observations to estimate starting parameters.*")
            return model.fit(method=method, maxiter=iters, disp=False, **kw)

    # 1) L-BFGS
    try:
        res = _quiet_fit("lbfgs", maxiter, pgtol=1e-5)
        if bool(res.mle_retvals.get("converged", 0)):
            return res, "lbfgs"
    except Exception:
        pass

    # 2) BFGS
    try:
        res = _quiet_fit("bfgs", maxiter // 2, gtol=1e-4)
        if bool(res.mle_retvals.get("converged", 0)):
            return res, "bfgs"
    except Exception:
        pass

    # 3) Powell → L-BFGS
    try:
        res_p = _quiet_fit("powell", maxiter // 3, xtol=1e-3)
        res = _quiet_fit("lbfgs", maxiter // 2, pgtol=1e-4, start_params=res_p.params)
        if bool(res.mle_retvals.get("converged", 0)):
            return res, "powell→lbfgs"
    except Exception:
        pass

    # 4) Nelder-Mead
    warnings.simplefilter("ignore", ConvergenceWarning)
    res = _quiet_fit("nm", maxiter // 2, xtol=1e-3)
    return res, "nm"

# ─────────────────── Évaluation du modèle ───────────────────
def evaluate_model(model_result) -> Dict[str, float]:
    resid = pd.Series(model_result.resid).dropna()
    # lag dynamique : 1..12, ≈20% des points
    lag = int(min(12, max(1, len(resid) // 5)))
    try:
        ljung_pval = acorr_ljungbox(resid, lags=[lag], return_df=True)['lb_pvalue'].iloc[0]
    except Exception:
        ljung_pval = np.nan
    try:
        normal_pval = normal_ad(resid)[1]
    except Exception:
        normal_pval = np.nan

    return {
        'aic': float(getattr(model_result, "aic", np.nan)),
        'bic': float(getattr(model_result, "bic", np.nan)),
        'llf': float(getattr(model_result, "llf", np.nan)),
        'ljung_pvalue': float(ljung_pval) if np.isfinite(ljung_pval) else -1.0,
        'normal_pvalue': float(normal_pval) if np.isfinite(normal_pval) else -1.0
    }

def all_exog_significant(model_result, exog_vars: List[str]) -> bool:
    p = model_result.pvalues
    for var in exog_vars:
        if var in p.index and p[var] > 0.05:
            return False
    return True

# ─────────────────── Grille SARIMAX bornée ───────────────────
def get_exog_subsets(columns: List[str], max_comb_size: int) -> List[List[str]]:
    if not columns:
        return [[]]
    cols = list(columns)
    subs = []
    for i in range(1, min(max_comb_size, len(cols)) + 1):
        subs.extend(itertools.combinations(cols, i))
    return [list(s) for s in subs] or [[]]

def try_sarimax_grid(y: pd.Series,
                     exog_df: pd.DataFrame,
                     d: int, D: int, s: int,
                     max_p: int, max_q: int,
                     max_exog_comb_size: int,
                     max_models: Optional[int] = None,
                     stop_on_first_valid: bool = False):
    best_model = None
    best_metrics = None
    best_params = None

    if exog_df is not None and not exog_df.empty:
        exog_df = exog_df.select_dtypes(include=[np.number]).reindex(index=y.index)
        exog_df = exog_df.ffill().bfill()
    else:
        exog_df = pd.DataFrame(index=y.index)

    exog_subsets = get_exog_subsets(list(exog_df.columns), max_exog_comb_size)
    trend_used = "n" if d > 0 else "c"
    tested = 0

    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        for (P, D_seas, Q, s_val) in get_valid_seasonal_orders(D, s, max_P=max_p, max_Q=max_q):
            for exog_subset in exog_subsets:
                # Garde-fou : si trop peu d'observations pour cette complexité, on skip
                n = len(y)
                order_complexity = (p + q) + (P + Q)
                exog_k = len(exog_subset) if exog_subset else 0
                if n < max(30, 8 + 4 * order_complexity + exog_k):
                    continue

                if max_models is not None and tested >= max_models:
                    return best_model, best_metrics, best_params
                tested += 1
                try:
                    exog = exog_df[exog_subset] if exog_subset else None
                    result, used_opt = fit_sarimax_robust(
                        y, exog, order=(p, d, q), seasonal_order=(P, D_seas, Q, s_val), trend=trend_used
                    )
                    converged = bool(result.mle_retvals.get("converged", 0))
                    if not converged:
                        with mlflow.start_run(nested=True):
                            mlflow.log_metric("converged", 0.0)
                            mlflow.log_param("optimizer_used", used_opt)
                            mlflow.set_tag("tested_model", True)
                        continue

                    metrics = evaluate_model(result)
                    all_signif = all_exog_significant(result, exog_subset)

                    with mlflow.start_run(nested=True):
                        mlflow.log_param("order", (p, d, q))
                        mlflow.log_param("seasonal_order", (P, D_seas, Q, s_val))
                        mlflow.log_param("exog_vars", exog_subset)
                        mlflow.log_param("optimizer_used", used_opt)
                        mlflow.log_param("all_exog_significant", all_signif)
                        mlflow.log_metric("converged", 1.0)
                        for k, v in metrics.items():
                            mlflow.log_metric(k, v)
                        mlflow.set_tag("tested_model", True)

                    if not all_signif:
                        continue

                    if (metrics['ljung_pvalue'] > 0.05 and
                        metrics['normal_pvalue'] > 0.05 and
                        (best_model is None or metrics['aic'] < best_metrics['aic'])):
                        best_model, best_metrics, best_params = result, metrics, {
                            "order": (p, d, q),
                            "seasonal_order": (P, D_seas, Q, s_val),
                            "exog_vars": exog_subset,
                            "optimizer_used": used_opt
                        }
                        if stop_on_first_valid:
                            return best_model, best_metrics, best_params
                except Exception as e:
                    mlflow.set_tag("grid_error", str(e))
                    continue

    return best_model, best_metrics, best_params

# ─────────────────── Fallback Prophet ───────────────────
def fallback_prophet(df_cluster, cluster_id, output_folder: str, suffix: str = ""):
    """
    Accepte soit:
      - une Series indexée par dates (cible 'prix_m2_vente'),
      - un DataFrame avec une colonne 'prix_m2_vente' et un index date.
    Construit Prophet(df) = {'ds': dates, 'y': valeurs}.
    """
    print(f"[Cluster {cluster_id}] ❌ Aucun SARIMAX valide, fallback vers Prophet.")

    # Extraire y (cible) + index temporel, sans hypothèse de nom d'index
    if isinstance(df_cluster, pd.Series):
        y = pd.to_numeric(df_cluster, errors="coerce")
        idx = df_cluster.index
    elif isinstance(df_cluster, pd.DataFrame):
        if "prix_m2_vente" not in df_cluster.columns:
            raise ValueError("fallback_prophet: colonne 'prix_m2_vente' absente du DataFrame.")
        y = pd.to_numeric(df_cluster["prix_m2_vente"], errors="coerce")
        idx = df_cluster.index
    else:
        raise TypeError("fallback_prophet: df_cluster doit être une Series ou un DataFrame.")

    ds = pd.to_datetime(pd.Index(idx), errors="coerce")
    prophet_df = pd.DataFrame({"ds": ds, "y": y.values})
    prophet_df = prophet_df.dropna(subset=["ds", "y"]).sort_values("ds")
    prophet_df = prophet_df.drop_duplicates(subset="ds", keep="last").reset_index(drop=True)

    if prophet_df.empty:
        raise ValueError("fallback_prophet: aucune donnée valide (ds,y) pour Prophet après nettoyage.")

    model = Prophet()
    model.fit(prophet_df)

    os.makedirs(output_folder, exist_ok=True)
    path_main = os.path.join(output_folder, f"cluster_{cluster_id}_sarimax.pkl")
    joblib.dump(model, path_main)
    mlflow.log_artifact(path_main)

    if suffix:
        path_suffix = os.path.join(output_folder, f"cluster_{cluster_id}_prophet{suffix}.pkl")
        joblib.dump(model, path_suffix)
        mlflow.log_artifact(path_suffix)

    mlflow.set_tag("fallback", "prophet")

# ─────────────────── IO helpers ───────────────────
def load_periodique_concat(input_folder: str) -> pd.DataFrame:
    candidates = []
    for base in ["train_periodique_q12", "test_periodique_q12"]:
        files = sorted([f for f in os.listdir(input_folder) if f.startswith(base) and f.endswith(".csv")])
        candidates.extend(files)

    if not candidates:
        raise FileNotFoundError(
            f"Aucun fichier '*periodique_q12*.csv' trouvé dans {input_folder}. "
            "Assure-toi d'avoir exécuté 'splitst' et que les exports existent."
        )

    dfs = []
    for f in candidates:
        df = pd.read_csv(os.path.join(input_folder, f), sep=";", parse_dates=["date"])
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True).sort_values(["cluster", "date"])
    return full

# ─────────────────── Main par cluster ───────────────────
def train_all_clusters(input_folder: str, output_folder: str, suffix: str = "",
                       save_split: bool = False,
                       corr_threshold: float = 0.05,
                       lag_corr_threshold: float = 0.3,
                       max_p: int = 2, max_q: int = 2,
                       top_k_exog: int = 8,
                       max_exog_comb_size: int = 3,
                       max_models: Optional[int] = None,
                       sample_last: Optional[int] = None,
                       downsample_every: int = 1,
                       stop_on_first_valid: bool = False):
    os.makedirs(output_folder, exist_ok=True)
    mlflow.set_experiment("ST-SARIMAX-AutoSearch")

    # Si un run est actif (ex. crash précédent), on le ferme proprement
    if mlflow.active_run() is not None:
        mlflow.end_run()

    full = load_periodique_concat(input_folder)

    required = {"date", "cluster", "prix_m2_vente"}
    missing = required - set(full.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans les fichiers periodiques: {missing}")

    # Run parent pour la boucle clusters (pour éviter le conflit de runs)
    with mlflow.start_run(run_name="sarimax_all_clusters"):
        for cluster_id in sorted(full["cluster"].dropna().unique()):
            dfc = full[full["cluster"] == cluster_id].copy()
            dfc = dfc.set_index("date").sort_index()

            exog_vars = [c for c in dfc.columns if c not in ["prix_m2_vente", "cluster"]]
            exog_df_raw = dfc[exog_vars].select_dtypes(include=[np.number])

            # Lags conditionnels
            dfc_lagged, lagged_cols = apply_lag_if_correlated(
                dfc[["prix_m2_vente"] + list(exog_df_raw.columns)],
                target="prix_m2_vente", threshold=lag_corr_threshold
            )
            y_lag = dfc_lagged["prix_m2_vente"]
            exog_lag = dfc_lagged.drop(columns=["prix_m2_vente"])

            # Fréquence mensuelle + standardisation exog
            y_m, X_m, scaler_exog, full_idx = prepare_monthly_index(y_lag, exog_lag)
            frac_filled = float(y_m.notna().mean())

            # ---- Sampling (test rapide) ----
            n_before = len(y_m)
            if sample_last is not None and sample_last > 0:
                y_m = y_m.iloc[-sample_last:]
            if downsample_every and downsample_every > 1:
                y_m = y_m.iloc[::downsample_every]

            # réaligne exogènes sur le nouvel index échantillonné
            y_m.index.name = "date"
            if X_m is not None and not X_m.empty:
                X_m = X_m.reindex(y_m.index).ffill()
                X_m.index.name = "date"

            # Saison : n trop petit → pas de saisonnalité
            s = 12 if len(y_m) >= 24 else 0

            print(f"\n🌀 [Cluster {cluster_id}] Début entraînement — n={len(y_m)}, "
                  f"seasonal_s={s}, exog={list(X_m.columns) if X_m is not None else []}")

            # Ordres ADF (sur non-saisonnier si s=0)
            d = get_diff_order(y_m, seasonal=False)
            D = get_diff_order(y_m, seasonal=True, s=12) if s >= 2 else 0

            # Run imbriqué par cluster
            with mlflow.start_run(run_name=f"cluster_{cluster_id}", nested=True):
                # Logs sampling & ADF
                mlflow.log_param("sample_last", sample_last if sample_last else 0)
                mlflow.log_param("downsample_every", downsample_every)
                mlflow.log_metric("n_points_before_sample", float(n_before))
                mlflow.log_metric("n_points_after_sample", float(len(y_m)))

                mlflow.log_param("ADF_d", d)
                mlflow.log_param("ADF_D", D)
                mlflow.log_param("ADF_s", s)
                mlflow.log_param("lagged_exog", lagged_cols)
                mlflow.log_param("exog_initial", exog_vars)
                mlflow.log_metric("frac_month_non_nan_after_align", frac_filled)

                # Sauvegarde scaler
                if scaler_exog is not None:
                    scaler_path = os.path.join(output_folder, f"cluster_{cluster_id}_exog_scaler.pkl")
                    joblib.dump(scaler_exog, scaler_path)
                    mlflow.log_artifact(scaler_path)

                # Filtre corrélation faible + top-k pour vitesse
                if X_m is None or X_m.empty:
                    mlflow.set_tag("reason", "no_exog_available")
                    fallback_prophet(y_m.to_frame("prix_m2_vente"), cluster_id, output_folder, suffix=suffix)
                    continue

                corrs = X_m.corrwith(y_m).abs().sort_values(ascending=False)
                top_cols = corrs.index[:min(top_k_exog, len(corrs))]
                X_m2 = X_m[top_cols]
                X_m2 = filter_exog_by_corr(y_m, X_m2, threshold=corr_threshold)

                if X_m2 is None or X_m2.empty:
                    mlflow.set_tag("reason", "no_exog_remaining_after_corrfilter")
                    fallback_prophet(pd.concat([y_m.rename("prix_m2_vente"), X_m], axis=1) if X_m is not None else
                                     y_m.to_frame("prix_m2_vente"),
                                     cluster_id, output_folder, suffix=suffix)
                    continue

                # Si n est vraiment petit, rétrécir automatiquement la grille
                local_max_p = 1 if len(y_m) < 24 else max_p
                local_max_q = 1 if len(y_m) < 24 else max_q
                local_max_exog_comb = min(max_exog_comb_size, max(1, len(X_m2.columns)))

                best_model, best_metrics, best_params = try_sarimax_grid(
                    y=y_m, exog_df=X_m2, d=d, D=D, s=s,
                    max_p=local_max_p, max_q=local_max_q,
                    max_exog_comb_size=local_max_exog_comb,
                    max_models=max_models,
                    stop_on_first_valid=False  # gardons la recherche bornée
                )

                if best_model:
                    path_main = os.path.join(output_folder, f"cluster_{cluster_id}_sarimax.pkl")
                    joblib.dump(best_model, path_main)
                    mlflow.log_artifact(path_main)

                    if suffix:
                        path_suffix = os.path.join(output_folder, f"cluster_{cluster_id}_sarimax{suffix}.pkl")
                        joblib.dump(best_model, path_suffix)
                        mlflow.log_artifact(path_suffix)

                    for k, v in best_metrics.items():
                        mlflow.log_metric(k, v)
                    for k, v in best_params.items():
                        if k != "exog_vars":
                            mlflow.log_param(k, v)
                    mlflow.log_param("exog_vars", best_params.get("exog_vars", []))

                    print(f"✅ [Cluster {cluster_id}] SARIMAX validé (opt={best_params.get('optimizer_used')}) et sauvegardé.")
                else:
                    mlflow.set_tag("reason", "no_valid_sarimax")
                    fallback_prophet(pd.concat([y_m.rename("prix_m2_vente"), X_m], axis=1) if X_m is not None else
                                     y_m.to_frame("prix_m2_vente"),
                                     cluster_id, output_folder, suffix=suffix)

# ─────────────────── CLI ───────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # alias compatibles DVC/Click
    parser.add_argument("--input-folder", "--encoded-folder",
                        dest="input_folder", type=str, required=True,
                        help="Dossier d'entrée (alias: --encoded-folder)")
    parser.add_argument("--output-folder", "--output",
                        dest="output_folder", type=str, required=True,
                        help="Dossier de sortie (alias: --output)")
    parser.add_argument("--suffix", type=str, default="", help="Suffixe à ajouter (copie supplémentaire)")
    parser.add_argument("--save-split", action="store_true", help="Sauvegarder les fichiers train/test par cluster")
    # bornes de la grille (accélération)
    parser.add_argument("--corr-threshold", type=float, default=0.05,
                        help="Seuil minimal d'|corr| pour garder une exog (ex: 0.2 ou 0.3).")
    parser.add_argument("--max-p", type=int, default=2, help="p maximum (AR)")
    parser.add_argument("--max-q", type=int, default=2, help="q maximum (MA)")
    parser.add_argument("--top-k-exog", type=int, default=8, help="Top-k exogènes par corrélation à garder")
    parser.add_argument("--max-exog-comb-size", type=int, default=3, help="Taille max des combinaisons d'exogènes")
    parser.add_argument("--max-models", type=int, default=None, help="Nombre max de modèles testés (early stop)")
    # mode test - sampling
    parser.add_argument("--sample-last", type=int, default=None,
                        help="Ne garder que les N derniers mois par cluster (ex: 24).")
    parser.add_argument("--downsample-every", type=int, default=1,
                        help="Garder un point sur K (ex: 2 ⇒ un mois sur deux).")
    parser.add_argument("--stop-on-first-valid", action="store_true",
                        help="Arrêter la recherche dès qu'un modèle valide est trouvé (si utilisé).")

    args = parser.parse_args()
    train_all_clusters(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        suffix=args.suffix,
        save_split=args.save_split,
        max_p=args.max_p,
        max_q=args.max_q,
        top_k_exog=args.top_k_exog,
        max_exog_comb_size=args.max_exog_comb_size,
        max_models=args.max_models,
        sample_last=args.sample_last,
        downsample_every=args.downsample_every,
        stop_on_first_valid=args.stop_on_first_valid,
        corr_threshold=args.corr_threshold,
    )

