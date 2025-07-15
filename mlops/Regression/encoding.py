import os
import click
import numpy as np
import pandas as pd
import mlflow


@click.command()
@click.option(
    "--data-path",
    prompt="Chemin vers df_cluster.csv",
    help="Fichier CSV complet avec colonne split",
)
@click.option(
    "--output", prompt="Dossier de sortie",
    help="Où sauvegarder les fichiers encodés"
)
@click.option(
    "--target",
    default="prix_m2_vente",
    show_default=True,
    help="Nom de la colonne cible",
)
def encode_data(data_path, output, target):
    mlflow.set_experiment("regression_pipeline")
    try:
        with mlflow.start_run(run_name="encoding"):
            print(f"Chargement depuis {data_path}")
            df = pd.read_csv(
                data_path, sep=";", parse_dates=["date"],
                dtype={target: np.float32}
            )

            for col in ["split", target, "date"]:
                if col not in df.columns:
                    raise ValueError(
                        f"La colonne '{col}' est manquante dans le fichier."
                    )

            if not set(["train", "test"]).issubset(df["split"].unique()):
                raise ValueError(
                    "train et test doivent exister dans split."
                )

            df = df.dropna(subset=[target])
            df["date"] = pd.to_datetime(df["date"])

            train = df[df["split"] == "train"]
            test = df[df["split"] == "test"]

            cat_cols = train.select_dtypes(include="object").columns.drop(
                ["split", "date"], errors="ignore"
            )
            if len(cat_cols) > 0:
                train = pd.get_dummies(train, columns=cat_cols, drop_first=True)
                test = pd.get_dummies(test, columns=cat_cols, drop_first=True)
                train, test = train.align(test, join="left",
                                          axis=1, fill_value=0)

            X_train = train.drop(columns=[target])
            y_train = train[[target]].astype(np.float32)
            X_test = test.drop(columns=[target])
            y_test = test[[target]].astype(np.float32)

            os.makedirs(output, exist_ok=True)
            X_train_path = os.path.join(output, "X_train.csv")
            y_train_path = os.path.join(output, "y_train.csv")
            X_test_path = os.path.join(output, "X_test.csv")
            y_test_path = os.path.join(output, "y_test.csv")

            X_train.to_csv(X_train_path, sep=";", index=False, float_format="%.2f")
            y_train.to_csv(y_train_path, sep=";", index=False, float_format="%.2f")
            X_test.to_csv(X_test_path, sep=";", index=False, float_format="%.2f")
            y_test.to_csv(y_test_path, sep=";", index=False, float_format="%.2f")

            mlflow.log_artifact(X_train_path)
            mlflow.log_artifact(y_train_path)
            mlflow.log_artifact(X_test_path)
            mlflow.log_artifact(y_test_path)
            mlflow.log_param("data_path", data_path)
            mlflow.log_param("target", target)
            mlflow.log_metric("n_train", len(X_train))
            mlflow.log_metric("n_test", len(X_test))
            mlflow.log_metric("n_features", X_train.shape[1])

            print("Encodage terminé et sauvegardé.")
    except Exception as e:
        print(f"Erreur : {e}")


if __name__ == "__main__":
    encode_data()
