# Demo video — shot list (about 5 minutes)

The assignment asks for "a short video recording of the overall pipeline". Record your screen with
**Windows + G** (Xbox Game Bar) or OBS, with a microphone if you want to narrate. Rehearse once first.

**Before recording** (one-time prep, so nothing slow happens on camera):

1. Open **two terminals**: a Windows PowerShell in `C:\Personal Stuff\MLOPS` and a WSL window in `/mnt/c/Personal Stuff/MLOPS`.
2. In WSL, make sure the cluster is up: `docker ps | grep heart-mlops` (if empty, run `bash scripts/k8s_deploy.sh` and `bash scripts/k8s_monitoring.sh` first).
3. Open browser tabs: the GitHub repo, its **Actions** tab, http://localhost:8080/docs, http://grafana.localhost:8080, http://prometheus.localhost:8080/targets.
4. Close anything private (mail, chat, other repos' tabs).

| # | Time | Show | Say (suggested) |
|---|---|---|---|
| 1 | 0:00 | `docs/architecture.png` (or page 3 of the report) | "This is the pipeline: data and ML, CI/CD, serving on Kubernetes, and monitoring." |
| 2 | 0:30 | GitHub repo page: folders `src/`, `test/`, `k8s/`, `scripts/`, `.github/workflows/` | "Everything is in one repository: code, tests, Dockerfile, manifests and the CI workflow." |
| 3 | 1:00 | PowerShell: `python -m src.models.train` (let it print the comparison table), then `mlflow ui --backend-store-uri sqlite:///mlflow.db` and the runs table in the browser | "Three models are tuned with cross-validation. Every run is tracked in MLflow with parameters, metrics and plots." |
| 4 | 1:45 | PowerShell: `python -m pytest -q` (shows 87 passed) and `ruff check .` | "87 tests and lint run locally and in CI." |
| 5 | 2:15 | Browser: GitHub **Actions** → latest run → the five green jobs and the artifacts list | "On every push CI lints, tests, builds and tests the Docker image, deploys to a Kubernetes cluster and trains the model. It fails on any error." |
| 6 | 3:00 | WSL: `bash scripts/docker_smoke_test.sh` (or show its saved output if short on time) | "The model is served by a FastAPI container. The smoke test builds it, runs it, and checks the API." |
| 7 | 3:30 | WSL: `kubectl -n heart-mlops get pods,svc,ingress`; browser: http://localhost:8080/docs → **POST /predict** → **Try it out** → **Execute** | "Two replicas run behind a Service and an Ingress. A prediction comes back with the confidence." |
| 8 | 4:15 | Browser: Grafana dashboard (run `bash scripts/generate_traffic.sh 100` in WSL so the panels move), then the Prometheus **Targets** page | "Prometheus scrapes both pods and Grafana shows request rate, latency, predictions by class and per-pod load." |
| 9 | 4:50 | Back to the architecture diagram | "From raw data to a monitored, tested, containerised service, fully automated." |

**Tips**

- Do not read long output aloud; let the screen speak and keep to the "Say" line.
- If a step is slow (train, image build), cut to the saved screenshot or recorded result rather than waiting on camera.
- Keep the final file under the size limit of your submission portal (about 5 minutes at 720p is typically 30–80 MB).
