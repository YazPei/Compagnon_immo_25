import asyncio
import os
import subprocess
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from ..auth.api_key import verify_api_key

router = APIRouter(prefix="/api/v1/deployment", tags=["deployment"])


class DeploymentRequest(BaseModel):
    environment: str = "staging"
    force: bool = False
    run_tests: bool = True


class DeploymentResponse(BaseModel):
    status: str
    deployment_id: str
    message: str
    logs_url: Optional[str] = None


async def run_deployment_script(environment: str, run_tests: bool = True):
    """Exécute le script de déploiement en arrière-plan."""
    try:
        # Utiliser le chemin relatif au projet
        script_path = os.path.join(os.getcwd(), "scripts", "deploy.sh")

        if not os.path.exists(script_path):
            print(f"❌ Script de déploiement non trouvé: {script_path}")
            return

        process = await asyncio.create_subprocess_exec(
            "bash",
            script_path,
            environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd(),
            env=dict(
                os.environ,
                **{
                    "DAGSHUB_USERNAME": os.getenv("DAGSHUB_USERNAME"),
                    "DAGSHUB_TOKEN": os.getenv("DAGSHUB_TOKEN"),
                    "ENVIRONMENT": environment,
                },
            ),
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            print(f"✅ Déploiement {environment} réussi")
            print(stdout.decode())

            if run_tests:
                # Exécuter les tests avec le chemin correct
                test_script = os.path.join(os.getcwd(), "tests", "test_deployment.py")

                if os.path.exists(test_script):
                    test_process = await asyncio.create_subprocess_exec(
                        "python",
                        test_script,
                        "--environment",
                        environment,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=os.getcwd(),
                    )

                    test_stdout, test_stderr = await test_process.communicate()

                    if test_process.returncode == 0:
                        print("✅ Tests de déploiement réussis")
                        print(test_stdout.decode())
                    else:
                        print(f"❌ Tests échoués: {test_stderr.decode()}")
                else:
                    print("⚠️ Script de test non trouvé, tests ignorés")
        else:
            print(f"❌ Déploiement échoué: {stderr.decode()}")

    except Exception as e:
        print(f"❌ Erreur lors du déploiement: {str(e)}")


@router.post("/deploy", response_model=DeploymentResponse)
async def trigger_deployment(
    request: DeploymentRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Déclenche un déploiement automatisé."""

    if request.environment not in ["staging", "production", "development"]:
        raise HTTPException(
            status_code=400,
            detail="Environment must be 'staging', 'production' or 'development'",
        )

    # Générer un ID unique pour le déploiement
    import uuid

    deployment_id = str(uuid.uuid4())

    # Vérifier que les scripts existent
    script_path = os.path.join(os.getcwd(), "scripts", "deploy.sh")
    if not os.path.exists(script_path):
        raise HTTPException(
            status_code=500, detail=f"Script de déploiement non trouvé: {script_path}"
        )

    # Lancer le déploiement en arrière-plan
    background_tasks.add_task(
        run_deployment_script, request.environment, request.run_tests
    )

    return DeploymentResponse(
        status="initiated",
        deployment_id=deployment_id,
        message=f"Déploiement {request.environment} initié",
        logs_url=f"/api/v1/deployment/{deployment_id}/logs",
    )


@router.get("/status")
async def deployment_status(api_key: str = Depends(verify_api_key)):
    """Retourne le statut du déploiement actuel."""

    try:
        # Vérifier si l'application fonctionne
        app_status = "running"

        # Vérifier les modèles DVC
        try:
            dvc_status = subprocess.run(
                ["dvc", "status"], capture_output=True, text=True, cwd=os.getcwd()
            )

            models_synced = "up to date" in dvc_status.stdout.lower()

        except Exception:
            models_synced = False

        return {
            "status": app_status,
            "models_synced": models_synced,
            "environment": os.getenv("ENVIRONMENT", "development"),
            "project_path": os.getcwd(),
            "dagshub_configured": bool(
                os.getenv("DAGSHUB_USERNAME") and os.getenv("DAGSHUB_TOKEN")
            ),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/rollback")
async def rollback_deployment(
    background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)
):
    """Effectue un rollback du déploiement."""

    async def run_rollback():
        try:
            # Utiliser le chemin relatif au projet
            rollback_script = os.path.join(os.getcwd(), "scripts", "rollback.sh")

            if not os.path.exists(rollback_script):
                print(f"❌ Script de rollback non trouvé: {rollback_script}")
                # Rollback simple avec git
                git_process = await asyncio.create_subprocess_exec(
                    "git",
                    "checkout",
                    "HEAD~1",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=os.getcwd(),
                )

                stdout, stderr = await git_process.communicate()

                if git_process.returncode == 0:
                    print("✅ Rollback Git réussi")
                else:
                    print(f"❌ Rollback Git échoué: {stderr.decode()}")
                return

            process = await asyncio.create_subprocess_exec(
                "bash",
                rollback_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd(),
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                print("✅ Rollback réussi")
                print(stdout.decode())
            else:
                print(f"❌ Rollback échoué: {stderr.decode()}")

        except Exception as e:
            print(f"❌ Erreur lors du rollback: {str(e)}")

    background_tasks.add_task(run_rollback)

    return {"status": "rollback_initiated", "message": "Rollback en cours"}


@router.get("/models/sync")
async def sync_models(api_key: str = Depends(verify_api_key)):
    """Synchronise les modèles avec DagsHub."""

    try:
        # Vérifier la configuration DVC
        if not (os.getenv("DAGSHUB_USERNAME") and os.getenv("DAGSHUB_TOKEN")):
            raise HTTPException(
                status_code=500,
                detail="Configuration DagsHub manquante (DAGSHUB_USERNAME, DAGSHUB_TOKEN)",
            )

        # Synchroniser avec DVC
        result = subprocess.run(
            ["dvc", "pull"], capture_output=True, text=True, cwd=os.getcwd()
        )

        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Modèles synchronisés avec succès",
                "output": result.stdout,
            }
        else:
            return {
                "status": "error",
                "message": "Erreur lors de la synchronisation",
                "error": result.stderr,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
