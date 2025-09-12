#!/usr/bin/env python
# coding: utf-8

import click
from encoding import encode_data
from train_lgbm import train_lgbm_model
from train_xgb import train_xgb_model
from analyse import analyse_model

@click.group()
def cli():
    """Pipeline CLI pour la modélisation régressive immobilière."""
    pass

@cli.command()
@click.option('--data-path', prompt='Chemin vers df_cluster.csv', help='Chemin vers df_cluster.csv')
@click.option('--output', default='./encoded', prompt='Dossier de sortie encodé', help='Dossier de sortie pour les jeux encodés')
def encode(data_path, output):
    encode_data(data_path, output)

@cli.command()
@click.option('--encoded-folder', default='./encoded', prompt='Dossier des données encodées', help='Dossier contenant les données encodées')
def train_lgbm(encoded_folder):
    train_lgbm_model(encoded_folder)

@cli.command()
@click.option('--encoded-folder', default='./encoded', prompt='Dossier des données encodées', help='Dossier contenant les données encodées')
@click.option('--use-gpu', is_flag=True, help='Utiliser GPU pour XGBoost')
def train_xgb(encoded_folder, use_gpu):
    train_xgb_model(encoded_folder, use_gpu)

@cli.command()
@click.option('--encoded-folder', default='./encoded', prompt='Dossier des données encodées', help='Dossier contenant X_train/X_test et le modèle')
@click.option('--model', type=click.Choice(['lightgbm', 'xgboost']), prompt='Modèle à analyser', help='Choix du modèle')
def analyse(encoded_folder, model):
    analyse_model(encoded_folder, model)

if __name__ == '__main__':
    cli()

