# utils/lineage.py
import hashlib, json, mlflow, subprocess, os, pathlib as p
def file_sha256(path): 
    h=hashlib.sha256(); 
    with open(path,'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def log_lineage(dataset_path, schema_path=None):
    ds_hash = file_sha256(dataset_path) if p.Path(dataset_path).exists() else ""
    mlflow.set_tag("dataset_path", dataset_path)
    mlflow.set_tag("dataset_sha256", ds_hash)
    if schema_path and p.Path(schema_path).exists():
        mlflow.set_tag("schema_path", schema_path)
        mlflow.set_tag("schema_sha256", file_sha256(schema_path))
    # DVC commit / rev
    try:
        rev = subprocess.check_output(["dvc","status","-c"], text=True)
        mlflow.set_tag("dvc_status", rev.strip())
    except Exception:
        pass
