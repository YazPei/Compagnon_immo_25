# path: mlops/5_clustering/Clustering.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, traceback
from pathlib import Path
from typing import List
import warnings; warnings.filterwarnings("ignore")

import click, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression

# Optional deps (no-op si absent)
try:
    import mlflow  # type: ignore
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False
    class _MLFlowNoOp:
        def set_tracking_uri(self, *a, **k): ...
        def set_experiment(self, *a, **k): ...
        def start_run(self, *a, **k): ...
        def end_run(self): ...
        def log_artifact(self, *a, **k): ...
    mlflow = _MLFlowNoOp()  # type: ignore

try:
    import geopandas as gpd  # type: ignore
    HAS_GPD = True
except Exception:
    HAS_GPD = False

# ───────── helpers ─────────
def _diag():
    print("=== [clustering] diagnostics ===")
    print("python_exe:", sys.executable)
    print("python_ver:", sys.version.split()[0])
    print("cwd:", os.getcwd())
    print("HAS_MLFLOW:", HAS_MLFLOW, "HAS_GPD:", HAS_GPD)
    print("SAFE_MODE:", os.getenv("CLUSTERING_SAFE", "0"))
    print("MLFLOW_TRACKING_URI:", os.environ.get("MLFLOW_TRACKING_URI"))

def _setup_mlflow(exp_name: str = "Clustering Données Immo") -> None:
    if not HAS_MLFLOW: return
    uri = os.getenv("MLFLOW_TRACKING_URI")
    try:
        if uri:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(exp_name)
        else:
            raise RuntimeError("MLFLOW_TRACKING_URI non défini")
    except Exception:
        local_dir = Path("mlruns").resolve(); local_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file://{local_dir}")
        mlflow.set_experiment(exp_name + " (offline)")

def _ensure_dirs(*paths: Path) -> None:
    for p in paths: p.mkdir(parents=True, exist_ok=True)

def _load_csv(file_path: Path, parse_dates: List[str] | None = None) -> pd.DataFrame:
    if not file_path.exists(): raise FileNotFoundError(f"Introuvable: {file_path}")
    return pd.read_csv(file_path, sep=";", parse_dates=parse_dates, low_memory=False)

def _derive_code_postal(df: pd.DataFrame) -> pd.Series:
    if "codePostal" in df.columns:
        return df["codePostal"].astype(str).str.replace(r"\.0$", "", regex=True)
    return pd.Series(["inconnu"] * len(df), index=df.index)

def _code_postal_from_geo(df: pd.DataFrame, geo_file: Path) -> pd.Series:
    if not HAS_GPD or not geo_file.exists(): return _derive_code_postal(df)
    need = {"mapCoordonneesLatitude", "mapCoordonneesLongitude"}
    if not need.issubset(df.columns): return _derive_code_postal(df)
    polys = gpd.read_file(geo_file)[["codePostal","geometry"]].to_crs(epsg=4326)
    idx = df["mapCoordonneesLatitude"].notna() & df["mapCoordonneesLongitude"].notna()
    pts = gpd.GeoDataFrame(
        df.loc[idx].copy(),
        geometry=gpd.points_from_xy(df.loc[idx,"mapCoordonneesLongitude"], df.loc[idx,"mapCoordonneesLatitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, polys, how="left", predicate="within")
    cp = pd.Series("inconnu", index=df.index)
    cp.loc[idx] = joined["codePostal"].astype(str).str.replace(r"\.0$","",regex=True).reindex(df.loc[idx].index).fillna("inconnu")
    return cp

def _plot_elbow(wcss: list[float], ks: list[int], out_png: Path) -> None:
    plt.figure(figsize=(6,4)); plt.plot(ks, wcss, marker="o")
    plt.title("Elbow plot (KMeans)"); plt.xlabel("k"); plt.ylabel("WCSS"); plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close()

def _cp_regroup(cp: str, frequents: set[str]) -> str:
    s = str(cp)
    if s in frequents: return s
    if s.startswith("97") and len(s)>=3: return s[:3]
    if s.isdigit() and len(s)==5: return s[:2]
    return "inconnu"

def _cp_final(zone: str) -> str:
    s = str(zone)
    if s.isdigit() and len(s)==5: return s
    if s.isdigit() and len(s)==2: return s+"000"
    if s.startswith("97") and len(s)==3: return s+"00"
    return "inconnu"

class _NullCtx:
    def __enter__(self): return self
    def __exit__(self,*a): return False

def safe_emit(df_full: pd.DataFrame, out_cluster_csv: Path, out_st_csv: Path) -> None:
    # why: garantir des fichiers valides pour DVC/étapes suivantes
    print("[SAFE] Emission des sorties minimales…")
    _ensure_dirs(out_cluster_csv.parent, out_st_csv.parent, Path("mlflow_outputs"), Path("data"))
    out = df_full.copy()
    out["cluster"] = pd.Series([""]*len(out), dtype="string")
    out["cluster_label"] = "inconnu"
    out.drop(columns=["split"], errors="ignore").to_csv(out_st_csv, sep=";", index=False)
    out.to_csv(out_cluster_csv, sep=";", index=False)
    fig = plt.figure(figsize=(4,3)); plt.text(0.5,0.5,"no elbow (safe)", ha="center", va="center"); plt.axis("off")
    plt.savefig("mlflow_outputs/elbow_plot.png", dpi=90, bbox_inches="tight"); plt.close(fig)
    Path("mlflow_outputs/cluster_input.csv").write_text("codePostal_recons\n", encoding="utf-8")
    out.drop(columns=["split"], errors="ignore").to_csv(Path("data/df_sales_clean_ST.csv"), sep=";", index=False)
    out.to_csv(Path("data/df_cluster.csv"), sep=";", index=False)
    print(f"[SAFE] OK → {out_cluster_csv}, {out_st_csv}, mlflow_outputs/*")

# ───────── pipeline ─────────
def run_clustering_pipeline(input_path: str, output_path: str, min_cp_freq: int = 3, min_samples: int = 100) -> None:
    _diag(); _setup_mlflow()
    SAFE = os.getenv("CLUSTERING_SAFE","0") == "1"

    in_dir = Path(input_path)
    train_file = in_dir / "df_sales_clean_train.csv"
    test_file  = in_dir / "df_sales_clean_test.csv"
    geo_file   = in_dir / "contours-codes-postaux.geojson"  # optionnel

    missing = [p for p in [train_file, test_file] if not p.exists()]
    if missing: raise FileNotFoundError(f"Fichiers manquants: {', '.join(map(str, missing))}")

    out_path = Path(output_path)
    out_dir = out_path.parent if out_path.suffix.lower()==".csv" else out_path
    out_cluster_csv = out_path if out_path.suffix.lower()==".csv" else (out_dir / "df_cluster.csv")
    out_st_csv = out_dir / "df_sales_clean_ST.csv"
    _ensure_dirs(out_dir, Path("mlflow_outputs"), Path("exports"), Path("data"))

    ctx = mlflow.start_run(run_name="clustering_macro_kpi") if HAS_MLFLOW else _NullCtx()
    with ctx:
        # Load
        train = _load_csv(train_file, parse_dates=["date"]); train["split"]="train"
        test  = _load_csv(test_file , parse_dates=["date"]); test["split"]="test"
        print(f"[INFO] train shape: {train.shape}, test shape: {test.shape}")
        df = pd.concat([train,test], ignore_index=True)

        # Code postal
        if not (HAS_GPD and geo_file.exists()):
            print(f"[INFO] GeoJSON absent ({geo_file}) ou geopandas indisponible, fallback non-spatial.")
        df["codePostal"] = _code_postal_from_geo(df, geo_file)

        # Zones mixtes (seuil paramétrable)
        cp_counts = df[df["split"]=="train"]["codePostal"].value_counts()
        cp_frequents = set(cp_counts[cp_counts>=min_cp_freq].index)
        print(f"[INFO] cp_frequents (>= {min_cp_freq}): {len(cp_frequents)}")
        df["zone_mixte"] = df["codePostal"].astype(str).apply(lambda x: _cp_regroup(x, cp_frequents))

        # Agrégations
        try:
            tr = df[df["split"]=="train"].copy().dropna(subset=["date"])
            tr["date"] = pd.to_datetime(tr["date"], errors="coerce")
            tr["Year"] = tr["date"].dt.year.astype(int)
            tr["Month"]= tr["date"].dt.month.astype(int)
            agg = (tr.groupby(["Year","Month","zone_mixte"])
                     .agg(prix_m2_vente=("prix_m2_vente","mean"))
                     .reset_index())
            print(f"[INFO] agg rows: {len(agg)}")
            if agg.empty: raise ValueError("Aucune agrégation produite.")
            agg["date"] = pd.to_datetime(dict(year=agg["Year"],month=agg["Month"],day=1))
            agg["codePostal_recons"] = agg["zone_mixte"].apply(_cp_final)
            agg = agg.sort_values(["codePostal_recons","date"])
            agg["ym"] = agg["Year"]*12 + agg["Month"]
            agg["t"]  = agg.groupby("codePostal_recons")["ym"].transform(lambda x: x - x.min())
            agg["log_prix"] = np.log(agg["prix_m2_vente"])
        except Exception as e:
            if SAFE:
                print(f"[WARN] Agrégations impossibles: {e}")
                return safe_emit(df, out_cluster_csv, out_st_csv)
            raise

        # TCAM
        def tcam(g: pd.DataFrame) -> float:
            g = g.dropna(subset=["log_prix","t"])
            if len(g)<2: return np.nan
            coef = LinearRegression().fit(g[["t"]].values, g["log_prix"].values).coef_[0]
            return (np.exp(coef)-1)*100*12

        tcam_df = agg.groupby("codePostal_recons").apply(tcam).reset_index(name="tc_am_reg")

        # Features brutes
        cluster_input_raw = (
            agg.rename(columns={"prix_m2_vente":"prix_m2_mean"})
               .groupby("codePostal_recons")
               .agg(prix_m2_mean=("prix_m2_mean","mean"),
                    prix_m2_std =("prix_m2_mean","std"),
                    prix_m2_max =("prix_m2_mean","max"),
                    prix_m2_min =("prix_m2_mean","min"))
               .reset_index()
        ).merge(tcam_df, on="codePostal_recons", how="left")
        cluster_input_raw["prix_m2_cv"] = cluster_input_raw["prix_m2_std"]/cluster_input_raw["prix_m2_mean"]
        cluster_input_raw.to_csv("mlflow_outputs/cluster_input_raw.csv", index=False, sep=";")
        if HAS_MLFLOW: mlflow.log_artifact("mlflow_outputs/cluster_input_raw.csv")

        # Stats NaN
        feats = ["prix_m2_std","prix_m2_max","prix_m2_min","tc_am_reg","prix_m2_cv"]
        nan_pct = cluster_input_raw[feats].isna().mean().sort_values(ascending=False)
        print("[INFO] NaN ratio per feature:\n", nan_pct.to_string())

        # Nettoyage → X
        X = cluster_input_raw[feats].replace([np.inf,-np.inf], np.nan)
        before_drop = len(X)
        X = X.dropna()
        after_drop = len(X)
        print(f"[INFO] X rows: before_drop={before_drop}, after_drop={after_drop}")

        cluster_input_raw.to_csv("mlflow_outputs/cluster_input.csv", index=False, sep=";")
        if HAS_MLFLOW: mlflow.log_artifact("mlflow_outputs/cluster_input.csv")

        if after_drop < max(2, min_samples):
            msg = f"Pas assez d’échantillons pour KMeans: {after_drop} < min_samples={min_samples}"
            if SAFE:
                print(f"[WARN] {msg}")
                return safe_emit(df, out_cluster_csv, out_st_csv)
            raise ValueError(msg)

        # Clustering
        ks = list(range(2, min(9, after_drop)))  # ne pas dépasser n_samples
        Xs = StandardScaler().fit_transform(X.values)
        wcss, sils, models = [], [], []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(Xs)
            models.append(km); wcss.append(km.inertia_)
            try: sils.append(silhouette_score(Xs, labels))
            except Exception: sils.append(np.nan)

        _plot_elbow(wcss, ks, Path("mlflow_outputs/elbow_plot.png"))
        if HAS_MLFLOW: mlflow.log_artifact("mlflow_outputs/elbow_plot.png")

        k_best = ks[int(np.nanargmax(sils))] if np.isfinite(sils).any() else 4
        if k_best >= after_drop: k_best = max(2, min(after_drop-1, 4))  # garde-fou
        km = models[ks.index(k_best)]
        labels = km.predict(Xs)

        # Labels
        cluster_input = cluster_input_raw.copy()
        cluster_input.loc[X.index, "cluster"] = labels.astype(int)
        order = (cluster_input.dropna(subset=["cluster"])
                              .groupby("cluster")["prix_m2_mean"]
                              .mean().sort_values().index.tolist())
        names = ["Zones rurales/petites villes","Centres urbains établis","Banlieues mixtes","Zones tendues/spéculatives"]
        name_map = {c: names[i % len(names)] for i, c in enumerate(order)}
        cluster_input["cluster_label"] = cluster_input["cluster"].map(name_map)

        # Merge → full
        df["codePostal_recons"] = df["zone_mixte"].apply(_cp_final)
        lab = cluster_input[["codePostal_recons","cluster","cluster_label"]].drop_duplicates()
        out_full = df.merge(lab, on="codePostal_recons", how="left")
        out_full["cluster"] = out_full["cluster"].astype("Int64")
        out_full["cluster_label"] = out_full["cluster_label"].fillna("inconnu")

        # Exports
        out_full.drop(columns=["split"], errors="ignore").to_csv(out_st_csv, sep=";", index=False)
        out_full.to_csv(out_cluster_csv, sep=";", index=False)
        if HAS_MLFLOW:
            mlflow.log_artifact(str(out_st_csv))
            mlflow.log_artifact(str(out_cluster_csv))
        out_full.drop(columns=["split"], errors="ignore").to_csv(Path("data/df_sales_clean_ST.csv"), sep=";", index=False)
        out_full.to_csv(Path("data/df_cluster.csv"), sep=";", index=False)

        print(f"✅ Clustering OK → {out_cluster_csv}")
        print("ℹ️ geopandas :", "OK" if HAS_GPD else "non installé (fallback appliqué)")

@click.command()
@click.option("--input-path", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--output-path", type=click.Path(), required=True)
@click.option("--min-cp-freq", type=int, default=3, help="Seuil CP fréquents (train) avant regroupement (défaut 3).")
@click.option("--min-samples", type=int, default=100, help="Taille minimale de X après dropna pour KMeans (défaut 100).")
def cli(input_path: str, output_path: str, min_cp_freq: int, min_samples: int):
    try:
        run_preprocessing_pipeline = None  # silence l7 tools linters
        run_clustering_pipeline(input_path, output_path, min_cp_freq=min_cp_freq, min_samples=min_samples)
    except Exception as e:
        print("[FATAL] clustering failed:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    cli()

