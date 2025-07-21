#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click

from load_split import split_data
from seasonal_decomp import run_decomposition
from sarimax_train import train_sarimax_models
from metrics import evaluate_models

@click.group()
def cli():
    """Pipeline CLI pour la modélisation temporelle des prix immobiliers."""
    pass

@cli.command()
@click.option('--input-path', prompt='Chemin vers df_sales_clean_ST.csv', type=click.Path(exists=True))
@click.option('--output-folder', default='./exports/st/', show_default=True)
def split_data_cmd(input_path, output_folder):
    """Découpe des données par cluster et préparation des séries temporelles."""
    split_data(input_path, output_folder)

@cli.command()
@click.option('--input-folder', prompt='Dossier contenant les séries clusterisées', type=click.Path(exists=True))
@click.option('--output-folder', default='./exports/st/', show_default=True)
def decompose(input_folder, output_folder):
    """Décomposition saisonnière additive & multiplicative."""
    run_decomposition(input_folder, output_folder)

@cli.command()
@click.option('--input-folder', prompt='Dossier contenant les séries clusterisées', type=click.Path(exists=True))
@click.option('--output-folder', default='./exports/st/', show_default=True)
def train(input_folder, output_folder):
    """Entraînement des modèles SARIMAX."""
    train_sarimax_models(input_folder, output_folder)

@cli.command()
@click.option('--model-folder', prompt='Dossier contenant les résultats SARIMAX', type=click.Path(exists=True))
def evaluate(model_folder):
    """Évaluation globale des performances SARIMAX."""
    evaluate_models(model_folder)

if __name__ == '__main__':
    cli()

