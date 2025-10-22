#!/usr/bin/env bash
set -e
python -c "import mlflow_connect as m; m.configure_mlflow(); print('Env OK')"
python train_example.py
exec "$@"
