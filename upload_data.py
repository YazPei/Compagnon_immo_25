# /home/vboxuser/Compagnon_new/Compagnon_immo_25/upload_data.py
#!/usr/bin/env python3
"""
Uploader polyvalent:
- --dvc *.csv.dvc : matérialise via DVC puis place les CSV dans DEST.
- --src PATHS...  : déplace/copier des fichiers locaux vers DEST.

Exemples:
  python3 upload_data.py --dvc merged_sales_data.csv.dvc --dest data --overwrite --verbose
  python3 upload_data.py --dvc "**/*.csv.dvc" --dest data
  python3 upload_data.py --src data_raw/*.csv --dest data --overwrite
"""

from __future__ import annotations
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

# ------------------------ Utils ------------------------

def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)

def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def move_or_copy(src: Path, dst: Path, copy: bool, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if overwrite:
            if dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        else:
            raise FileExistsError(f"Destination existe déjà: {dst}")
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))

# ------------------------ SRC mode ------------------------

def iter_source_files(src_items: List[str]) -> Iterable[Path]:
    for item in src_items:
        p = Path(item).expanduser()
        # Glob
        if any(ch in item for ch in "*?[]"):
            for m in Path().glob(item):
                if m.is_file():
                    yield m.resolve()
                elif m.is_dir():
                    yield from (f.resolve() for f in m.rglob("*") if f.is_file())
            continue
        if p.is_file():
            yield p.resolve()
        elif p.is_dir():
            yield from (f.resolve() for f in p.rglob("*") if f.is_file())
        # else : ignore silencieux

def run_src_mode(args: argparse.Namespace) -> int:
    files = list(dict.fromkeys(iter_source_files(args.src)))
    if not files:
        eprint("Aucun fichier trouvé d'après --src.")
        return 2
    dest = Path(args.dest).expanduser().resolve()
    ok = 0
    for src in files:
        dst = dest / (src.relative_to(Path(args.base_root)).as_posix() if args.preserve and src.is_relative_to(Path(args.base_root)) else src.name)
        try:
            move_or_copy(src, dst, copy=args.copy, overwrite=args.overwrite)
            if args.hash:
                if args.verbose:
                    print(f"[HASH] {dst} sha256={sha256(dst)}")
            print(f"[OK] {src} → {dst}")
            ok += 1
        except Exception as e:
            eprint(f"[ERR] {src}: {e}")
    print(f"\nRésumé: {ok}/{len(files)} fichiers → {dest}")
    return 0 if ok == len(files) else 3

# ------------------------ DVC mode ------------------------

def check_dvc_available() -> None:
    try:
        subprocess.run(["dvc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        sys.exit("ERREUR: DVC absent. Installe-le :\n  pip install dvc  # ou dvc[s3], dvc[gdrive]\nDétail: " + str(e))

def check_in_dvc_repo() -> None:
    if not Path(".dvc").exists():
        sys.exit("ERREUR: pas de répertoire .dvc ici. Va à la racine du dépôt DVC.")

def load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception:
        sys.exit("ERREUR: PyYAML manquant. Installe :\n  pip install pyyaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def outs_path_from_dvc_file(dvc_file: Path) -> Path:
    data = load_yaml(dvc_file)
    outs = data.get("outs") or []
    if not outs or not outs[0].get("path"):
        raise RuntimeError(f"Impossible de lire 'outs[0].path' dans {dvc_file}")
    return (dvc_file.parent / outs[0]["path"]).resolve()

def dvc_pull_target(dvc_file: Path, verbose: bool) -> None:
    cmd = ["dvc", "pull", str(dvc_file)]
    res = subprocess.run(cmd, cwd=str(dvc_file.parent), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if verbose:
        print(f"[DVC] {' '.join(cmd)}\n{res.stdout}")
    if res.returncode != 0:
        raise RuntimeError(f"Echec dvc pull:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

def expand_dvc_globs(patterns: List[str]) -> List[Path]:
    found: List[Path] = []
    for pat in patterns:
        for p in Path().glob(pat):
            if p.is_file() and (p.name.endswith(".csv.dvc") or p.suffix == ".dvc"):
                found.append(p.resolve())
    # dédoublonner en conservant l'ordre
    seen, out = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def run_dvc_mode(args: argparse.Namespace) -> int:
    check_dvc_available()
    check_in_dvc_repo()
    dvc_files = expand_dvc_globs(args.dvc)
    if not dvc_files:
        eprint("Aucun fichier .csv.dvc trouvé d'après --dvc.")
        return 2
    dest = Path(args.dest).expanduser().resolve()
    ok = 0
    for dvc_file in dvc_files:
        try:
            if args.verbose:
                print(f"[INFO] {dvc_file}")
            dvc_pull_target(dvc_file, args.verbose)
            csv_path = outs_path_from_dvc_file(dvc_file)
            if not csv_path.exists():
                raise FileNotFoundError(f"Après dvc pull, introuvable: {csv_path}")
            dst = dest / csv_path.name
            move_or_copy(csv_path, dst, copy=args.copy, overwrite=args.overwrite)
            if args.hash:
                if args.verbose:
                    print(f"[HASH] {dst} sha256={sha256(dst)}")
            print(f"[OK] {csv_path} → {dst}")
            ok += 1
        except Exception as e:
            eprint(f"[ERR] {dvc_file}: {e}")
    print(f"\nRésumé: {ok}/{len(dvc_files)} CSV placés dans {dest}")
    return 0 if ok == len(dvc_files) else 3

# ------------------------ CLI ------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Uploader/mover vers ./data (support DVC et sources locales).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dvc", nargs="+", help="Chemins/globs de fichiers .csv.dvc (DVC mode).")
    g.add_argument("--src", nargs="+", help="Fichiers/dossiers/globs (mode local).")
    p.add_argument("--dest", default="data", help="Dossier destination (défaut: data).")
    p.add_argument("--copy", action="store_true", help="Copier au lieu de déplacer.")
    p.add_argument("--overwrite", action="store_true", help="Écraser si le fichier existe.")
    p.add_argument("--preserve", action="store_true", help="[SRC] Préserver l'arborescence relative (avec --base-root).")
    p.add_argument("--base-root", default=".", help="[SRC] Racine utilisée avec --preserve (défaut: .).")
    p.add_argument("--hash", action="store_true", help="Afficher sha256 des fichiers écrits.")
    p.add_argument("--verbose", action="store_true", help="Logs détaillés.")
    return p

def main(argv: List[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.dvc:
        return run_dvc_mode(args)
    return run_src_mode(args)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

