# Vertex AI Workbench: Train, Build the Serving Image, and Push Artifacts

A single-instance runbook: spin up the Workbench VM from [`gcp_training.md`](gcp_training.md) Option A,
train the model, push the checkpoint to GCS, and build/test the serving image from
[`gcp_deployment.md`](gcp_deployment.md) — all before stopping/deleting the instance. Per
[`ADR-002`](../adr/ADR-002-separate-vms-for-training-vs-deployment.md), this VM is a temporary
train-and-build bench, not where the model ends up served — the built image is deployed to Cloud Run
separately, decoupled from this instance's lifecycle.

**Prerequisites this doc assumes are done before Section 7** (not yet in the repo as of this writing — see
`gcp_deployment.md` steps 2–3 for the design): `serve/main.py` (the FastAPI app) and a `Dockerfile` exist in
the repo. Section 7 reproduces the `Dockerfile` from `gcp_deployment.md` so this doc is self-contained for
the build/push steps; the serving code itself still needs to be written per that doc if it doesn't exist
yet.

---

## 1. Enable the required APIs (one-time per project)
```bash
gcloud services enable notebooks.googleapis.com compute.googleapis.com aiplatform.googleapis.com
```

## 2. Create the Workbench instance
```bash
gcloud workbench instances create fraud-gnn-workbench \
  --location=us-central1-a \
  --machine-type=n2-highmem-4 \
  --data-disk-size=100 \
  --data-disk-type=PD_SSD
```
- `n2-highmem-4` = 4 vCPU / 32 GB RAM — should comfortably clear preprocessing OOMs. Bump to
  `n2-highmem-8` (64 GB) if needed.
- `--data-disk-size=100` (GB) holds the repo, raw CSVs, processed artifacts, and Docker image layers.
- Exact flag names can drift between `gcloud` versions — run `gcloud workbench instances create --help`
  if a flag is rejected.

**Or via Console**: Vertex AI → Workbench → Instances → **Create New** → pick machine type and disk size.

## 3. Open JupyterLab
```bash
gcloud workbench instances describe fraud-gnn-workbench \
  --location=us-central1-a --format="value(proxyUri)"
```
Open the returned URL — routed through GCP's IAM-authenticated proxy, no firewall rule or SSH tunnel
needed. (Or Console: Workbench → Instances → "Open JupyterLab".)

## 4. Get the repo and data onto the instance
From a terminal inside JupyterLab (File → New → Terminal):
```bash
git clone https://github.com/srimugunthan/GNN-Fraud-Detection.git
cd GNN-Fraud-Detection
pip install -r requirements.txt
pip install torch torch-geometric   # for notebook 04
```
Data:
```bash
pip install kaggle
# set credentials via getpass or upload kaggle.json first — see README for auth details, don't hardcode a token in a cell
python download_dataset.py --unzip
```

## 5. Train the model
Run [`notebooks/04_gcn_model.ipynb`](../notebooks/04_gcn_model.ipynb) (or `gnn_fraud_detection.py`) through
to completion so a checkpoint exists on disk — the rest of this doc assumes `best_gnn_model.pt` (or the
notebook's `results/best_gcn_model.pt`) now exists.

## 6. Push the checkpoint + encoders to GCS
Per `gcp_deployment.md` step 1 — the serving code needs the weights *and* the fitted feature
encoders/scalers, final `in_channels`, and the `Config`/hyperparameter values used to reconstruct the model
before loading `state_dict`. Bundle whatever your training run produced and upload it:
```bash
gsutil mb -l us-central1 gs://<PROJECT_ID>-fraud-gnn 2>/dev/null || true   # create bucket if it doesn't exist yet

gsutil cp best_gnn_model.pt gs://<PROJECT_ID>-fraud-gnn/models/v1/
gsutil cp results/*.json gs://<PROJECT_ID>-fraud-gnn/models/v1/            # encoders/scalers/config, if saved as json/pkl
# or, for a whole directory of artifacts:
gsutil -m cp -r artifacts/ gs://<PROJECT_ID>-fraud-gnn/models/v1/
```
Verify:
```bash
gsutil ls -l gs://<PROJECT_ID>-fraud-gnn/models/v1/
```
This GCS path is the hand-off point between training and serving — `MODEL_URI` in the Cloud Run deploy
step (Section 9 below, and `gcp_deployment.md` step 5) points here.

## 7. Build the serving image

**Prefer `gcloud builds submit` over a local `docker build`.** It offloads the build to the managed Cloud
Build service, so you don't need Docker installed or root/sudo access on the Workbench instance (Workbench
VMs run under a restricted user by default, which local `docker build` can be fussy about):
```bash
gcloud artifacts repositories create fraud-gnn \
  --repository-format=docker --location=us-central1 2>/dev/null || true   # create repo if it doesn't exist yet

gcloud builds submit --tag us-central1-docker.pkg.dev/<PROJECT_ID>/fraud-gnn/serve:v1
```
This needs `serve/` and a `Dockerfile` in your working directory. The `Dockerfile` from
`gcp_deployment.md`, reproduced here for convenience:
```dockerfile
FROM python:3.11-slim
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install torch-geometric==2.5.3
RUN pip install fastapi uvicorn google-cloud-storage pandas scikit-learn
COPY serve/ /app/serve/
CMD ["uvicorn", "serve.main:app", "--host", "0.0.0.0", "--port", "8080"]
```
Pin `torch`/`torch-geometric` versions carefully — this is the most common place PyG deploys break, since
`torch-scatter`/`torch-sparse` compiled extensions are matched to an exact `torch` version + CPU/CUDA
build.

## 8. Test the image before deploying

Three options, roughly in order of least to most friction — pick based on what you're trying to catch:

**a) Fastest — test the app logic directly, no Docker at all:**
```bash
uvicorn serve.main:app --host 0.0.0.0 --port 8080
# from another terminal on the same instance:
curl localhost:8080/healthz
curl -X POST localhost:8080/predict -H "Content-Type: application/json" -d '{...}'
```
Good for catching bugs in the serving code itself, separately from container/dependency issues.

**b) Closer to production — run the built image locally, needs Docker installed:**
```bash
sudo apt-get update && sudo apt-get install -y docker.io   # check `docker --version` first — may already be present
docker pull us-central1-docker.pkg.dev/<PROJECT_ID>/fraud-gnn/serve:v1
docker run -p 8080:8080 us-central1-docker.pkg.dev/<PROJECT_ID>/fraud-gnn/serve:v1
curl localhost:8080/healthz
```
If `sudo` is restricted on your Workbench instance, this option may not work without extra setup — fall
back to (a) or (c).

**c) Real end-to-end test — deploy straight to Cloud Run and curl the live endpoint:**
See Section 9 below; given how cheap Cloud Run is for light usage, iterating directly there is often less
friction than fighting local Docker permissions.

## 9. Deploy to Cloud Run
```bash
gcloud run deploy fraud-gnn-serve \
  --image us-central1-docker.pkg.dev/<PROJECT_ID>/fraud-gnn/serve:v1 \
  --region us-central1 \
  --memory 2Gi --cpu 2 \
  --set-env-vars MODEL_URI=gs://<PROJECT_ID>-fraud-gnn/models/v1/ \
  --allow-unauthenticated   # or drop this and use IAM/auth for anything real
```
```bash
curl -X POST https://<cloud-run-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 34.0, "card1": 12345, "DeviceInfo": "DVC-771", ...}'
```
This step runs independently of the Workbench instance — see `gcp_deployment.md` for the full deploy
walkthrough, cost breakdown, and ops follow-ups.

## 10. Control cost — stop/start/delete the Workbench instance

Once the checkpoint is in GCS and the image is built and pushed, the Workbench instance's job is done —
nothing about the deployed Cloud Run service depends on it staying up (per ADR-002).

**Stop it** when stepping away — pauses compute billing, keeps the disk (repo, data, any local Docker
image layers) intact:
```bash
gcloud workbench instances stop fraud-gnn-workbench --location=us-central1-a
```

**Start it again** later if you need to retrain or rebuild:
```bash
gcloud workbench instances start fraud-gnn-workbench --location=us-central1-a
gcloud workbench instances describe fraud-gnn-workbench \
  --location=us-central1-a --format="value(proxyUri)"   # re-fetch the JupyterLab URL
```

**Delete it** once you're fully done with this instance (the only way to stop the disk-storage charge too):
```bash
gcloud workbench instances delete fraud-gnn-workbench --location=us-central1-a
```

**Cost estimate**: `n2-highmem-4` ≈ $0.26/hr compute + a modest managed-notebooks surcharge + ~100GB
pd-ssd storage (~$0.17/GB-month while the instance exists, running or stopped). A training + build session
of a few hours ≈ **$2–5**; delete the instance afterward and storage cost goes to zero. The deployed Cloud
Run service's own cost is separate and tracked in `gcp_deployment.md` (~$0.00–0.30/month for light usage).
