import os
import click
import numpy as np
import pandas as pd
import mlflow
from sklearn.preprocessing import OneHotEncoder


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
@click.option(
    "--max-modalities",
    default=10,
    show_default=True,
    help="Nombre max de modalités pour appliquer le OneHotEncoding",
)
def encode_data(data_path, output, target, max_modalities):
    """
    OneHotEncode uniquement les variables catégorielles avec peu de modalités.
    """
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

            df = df.dropna(subset=[target])
            df["date"] = pd.to_datetime(df["date"])

            train = df[df["split"] == "train"].copy()
            test = df[df["split"] == "test"].copy()

            cat_cols = train.select_dtypes(include="object").columns.drop(
                ["split", "date"], errors="ignore"
            )
            few_modalities = [
                col for col in cat_cols
                if train[col].nunique() <= max_modalities
            ]

            encoder = OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse=False
            )
            if few_modalities:
                encoder.fit(train[few_modalities])
                train_encoded = encoder.transform(train[few_modalities])
                test_encoded = encoder.transform(test[few_modalities])

                encoded_cols = encoder.get_feature_names_out(few_modalities)
                train_encoded_df = pd.DataFrame(
                    train_encoded, columns=encoded_cols, index=train.index
                )
                test_encoded_df = pd.DataFrame(
                    test_encoded, columns=encoded_cols, index=test.index
                )

                train = train.drop(columns=few_modalities)
                test = test.drop(columns=few_modalities)
                train = pd.concat([train, train_encoded_df], axis=1)
                test = pd.concat([test, test_encoded_df], axis=1)

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
            mlflow.log_param("encoded_cols", list(few_modalities))
            mlflow.log_metric("n_train", len(X_train))
            mlflow.log_metric("n_test", len(X_test))
            mlflow.log_metric("n_features", X_train.shape[1])

            print("Encodage terminé et sauvegardé.")
    except Exception as e:
        print(f"Erreur : {e}")


if __name__ == "__main__":
    encode_data()
