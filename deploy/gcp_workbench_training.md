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

## 11. Need a GPU? Create a Fresh Instance Instead of Upgrading This One

⚠️ **Superseded for notebook 5 (full-batch) by an actual OOM finding — read this before choosing a GPU
here.** This section originally recommended a T4/L4 for RGCN purely on compute grounds. Since then, running
notebook 5 on `n2-highmem-4` (32 GB) actually OOM-killed the kernel, which led to a corrected memory
estimate in [`resource_requirements.md`](../resource_requirements.md): **full-batch** RGCN training peaks
around **~25–35 GB**, not the ~6–14 GB originally assumed. A T4 (16 GB VRAM) or L4 (24 GB VRAM) does not have
enough memory for *that specific notebook* — for notebook 5, prefer Section 2's CPU-only setup bumped to
`n2-highmem-8` (§12) or, if it must be a GPU, §11.5's A100 below.

✅ **This section's T4/L4 setup is still exactly right for GCN/GraphSAGE (notebook 4)**, whose ~5–10 GB
full-batch peak comfortably fits a T4, as originally documented — Sections 11.1–11.4 below cover creating
that instance. **It's also right for [`06_rgcn_minibatch_gpu.ipynb`](../notebooks/06_rgcn_minibatch_gpu.ipynb)**,
which trains the same RGCN architecture via `NeighborLoader` mini-batching instead of full-batch — memory
scales with one sampled subgraph (a few thousand nodes) rather than the whole 1.1M-node graph, so the
~25–35 GB problem above doesn't arise there either. Notebook 6 has its own dedicated setup/run steps in
**Section 13**, since it needs one extra install (`pyg-lib`) and has its own things worth checking once
running — this section (11) just gets the GPU instance itself created.

**Don't try to retrofit `fraud-gnn-workbench` in place.** It was created as `n2-highmem-4`, and the **N2
series doesn't support attaching GPUs at all** (only N1, A2, A3, and G2 do) — "adding a GPU" to it actually
means changing the machine series, not editing an existing config. Combined with this repo's own pattern of
treating the Workbench instance as disposable (Section 10 above, and
[ADR-002](../adr/ADR-002-separate-vms-for-training-vs-deployment.md)'s "stood up, used, torn down"
philosophy), it's cleaner to create a new instance with the GPU baked in from the start than to stop, edit,
and hope the hardware/driver changes land correctly on the old one.

### 11.1 Save the processed artifacts before deleting the old instance

Notebook 3's preprocessing is the real time/memory cost in this pipeline, not training
([`resource_requirements.md`](../resource_requirements.md)) — don't pay it twice. Before tearing down the
old instance, push what it already computed to GCS:
```bash
# on the OLD (fraud-gnn-workbench) instance
gsutil -m cp -r ieee-cis-dataset/processed/ gs://<PROJECT_ID>-fraud-gnn/processed-artifacts/v1/
```

### 11.2 Create the new GPU instance

A single **T4** on an **N1** machine type is the cost-effective default for this model (small — see the
param counts in `resource_requirements.md`; the GPU buys wall-clock, not VRAM headroom). An **L4** (G2
series) is a reasonable step up if T4 quota isn't available in your region:
```bash
gcloud workbench instances create fraud-gnn-workbench-gpu \
  --location=us-central1-a \
  --machine-type=n1-highmem-4 \
  --accelerator-type=NVIDIA_TESLA_T4 \
  --accelerator-core-count=1 \
  --data-disk-size=100 \
  --data-disk-type=PD_SSD
```
**Check GPU quota first** — new/lightly-used projects commonly have 0 quota for `NVIDIA_T4_GPUS` (or the
L4/A100 equivalent) in a given region. IAM & Admin → Quotas, filtered to `us-central1`, request an increase
if needed; this is the step most likely to block the `create` call. If `--accelerator-type`/
`--accelerator-core-count` are rejected by your `gcloud` version, use **Console** instead: Vertex AI →
Workbench → **Create New** → pick the GPU-enabled machine config directly, making sure "Install NVIDIA GPU
driver automatically" is checked so you don't have to install CUDA drivers by hand after boot.

### 11.3 Set up the new instance — repo, GPU-enabled PyTorch, and the saved artifacts

From a terminal inside the new instance's JupyterLab:
```bash
git clone https://github.com/srimugunthan/GNN-Fraud-Detection.git
cd GNN-Fraud-Detection
pip install -r requirements.txt

# CUDA build, not the CPU wheel from Section 4 — match the CUDA version the driver install landed on
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
```
Pull down the processed artifacts saved in Step 11.1 instead of re-running
[`03_preprocessing_gnn_features.ipynb`](../notebooks/03_preprocessing_gnn_features.ipynb):
```bash
gsutil -m cp -r gs://<PROJECT_ID>-fraud-gnn/processed-artifacts/v1/ ieee-cis-dataset/processed/
```
Verify the GPU is actually visible before kicking off a full run:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```
Then run [`04_gcn_model.ipynb`](../notebooks/04_gcn_model.ipynb) as in Section 5 — its ~5–10 GB full-batch
peak fits this instance fine. **Don't** run [`05_rgcn_model.ipynb`](../notebooks/05_rgcn_model.ipynb)'s
full-batch training here — see the warning at the top of Section 11. For
[`06_rgcn_minibatch_gpu.ipynb`](../notebooks/06_rgcn_minibatch_gpu.ipynb), see **Section 13** below for its
own dedicated setup and run steps (one extra install, plus what to check once it's running).

### 11.4 Retire the old instance

Once the new GPU instance is confirmed working, delete the CPU-only one — nothing depends on it staying up
(same reasoning as Section 10):
```bash
gcloud workbench instances delete fraud-gnn-workbench --location=us-central1-a
```

**Cost note**: this changes the Section 10 baseline. `n1-highmem-4` is similar compute cost to
`n2-highmem-4`, but a T4 adds roughly **+$0.35/hr** on top (L4 more, A100 substantially more) while the
instance is running — stop it (`gcloud workbench instances stop fraud-gnn-workbench-gpu --location=us-central1-a`)
between training runs rather than leaving a GPU idle. See Section 13 for notebook 6's own expected
runtime/cost on this instance.

### 11.5 Want *Notebook 5* (Full-Batch) on a GPU Specifically? Use A100, Not T4/L4

This subsection is only relevant if you're sticking with notebook 5's **full-batch** training rather than
switching to notebook 6's mini-batch version above. If you want notebook 5 on a GPU rather than the
`n2-highmem-8` CPU path (Section 12) — e.g. you're about to run many full-batch RGCN training passes and the
per-run time savings are worth it (see the cost/time comparison in
[`resource_requirements.md`](../resource_requirements.md)) — **T4 and L4 are not options here**, per the
warning at the top of this section: their 16–24 GB VRAM is below the ~25–35 GB full-batch RGCN peak. The
only GPU with enough VRAM is an **A100** (`a2-highgpu-1g`: 1× A100 40GB, 12 vCPU, 85 GB RAM — a fixed shape,
not something you assemble from separate machine-type + accelerator flags like Section 11.2's T4 setup):

```bash
gcloud workbench instances create fraud-gnn-workbench-a100 \
  --location=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator-type=NVIDIA_TESLA_A100 \
  --accelerator-core-count=1 \
  --data-disk-size=100 \
  --data-disk-type=PD_SSD
```

**Check A100 quota before running this** — more so than the T4/L4 case. New or lightly-used GCP projects
routinely start at **0 quota for `NVIDIA_A100_GPUS`**, and an increase request can take days with no
guarantee of approval, unlike the smaller GPUs. Budget for that lead time, or fall back to Section 12's
`n2-highmem-8` path (no quota request needed at all) if you need to start training sooner.

Setup from here is identical to Steps 11.1 and 11.3 above: push `ieee-cis-dataset/processed/` to GCS from
the old instance first, then on this new one `git clone` + `pip install -r requirements.txt` +
`pip install torch --index-url https://download.pytorch.org/whl/cu121` + `pip install torch-geometric`,
pull the processed artifacts back down from GCS, verify with `torch.cuda.is_available()`, then run
[`05_rgcn_model.ipynb`](../notebooks/05_rgcn_model.ipynb).

**Cost note**: `a2-highgpu-1g` ≈ **~$3.67/hr** (approximate on-demand `us-central1` — confirm current pricing
before relying on this), roughly 7× `n2-highmem-8`'s ~$0.52/hr. Per the comparison in
`resource_requirements.md`, this is plausibly offset by a much shorter run (~2–12 min vs. ~40–120 min), so
per-run $ cost lands close to a wash — the A100 path mainly pays off if you're training repeatedly, not for
a single run. Stop it (`gcloud workbench instances stop fraud-gnn-workbench-a100 --location=us-central1-a`)
immediately after each run — at this hourly rate, idle time is the expensive mistake to avoid.

## 12. Need More RAM Instead? Resize This Instance In Place

For notebook 5's corrected ~25–35 GB RGCN memory peak ([`resource_requirements.md`](../resource_requirements.md)),
this is the recommended fix — **staying on CPU, just bigger**. Unlike Section 11's GPU case, this is a
same-series resize (`n2-highmem-4` → `n2-highmem-8`, still N2), not a machine-series change, so none of
Section 11's complications apply: no GPU quota to request, no driver to install, and — the big one — **no
need to recreate the instance**. The repo, downloaded dataset, and any `ieee-cis-dataset/processed/`
artifacts already on the data disk carry straight through a resize untouched.

```bash
# 1. Stop it first — required before a machine-type change
gcloud workbench instances stop fraud-gnn-workbench --location=us-central1-a

# 2. Resize
gcloud workbench instances update fraud-gnn-workbench \
  --location=us-central1-a \
  --machine-type=n2-highmem-8
```
If `update` rejects `--machine-type` on your `gcloud` version (same flag-drift caveat as Section 2's
`create`), use **Console** instead: Vertex AI → Workbench → Instances → select `fraud-gnn-workbench`
(stopped) → **Edit** → change Machine type to `n2-highmem-8` → Save.

```bash
# 3. Start it back up
gcloud workbench instances start fraud-gnn-workbench --location=us-central1-a
gcloud workbench instances describe fraud-gnn-workbench \
  --location=us-central1-a --format="value(proxyUri)"   # re-fetch the JupyterLab URL
```

**Verify the resize landed** before re-running the notebook:
```bash
free -h   # should now show ~64 GB total, not ~32 GB
```

Go straight back into [05_rgcn_model.ipynb](../notebooks/05_rgcn_model.ipynb) from there — no re-clone,
no re-download, no re-running notebook 3's preprocessing.

**Escalate to `n2-highmem-16` (128 GB)** with the same stop → update → start sequence if `n2-highmem-8`
still OOMs; check `dmesg | grep -i "killed process"` first to confirm it's actually memory before paying for
another doubling.

**Cost note**: `n2-highmem-8` ≈ double `n2-highmem-4`'s ~$0.26/hr (roughly **~$0.52/hr**) — still far below
any of Section 11's GPU options, consistent with `resource_requirements.md`'s recommendation to prefer RAM
over VRAM for this specific (architecture-unchanged, full-batch) case.

## 13. Running Notebook 6 (Mini-Batch RGCN) on the GPU Instance

**Prerequisite**: the T4/L4 instance from Section 11 (11.1–11.4), with the repo cloned and GPU-enabled
`torch`/`torch-geometric` already installed per §11.3. This section covers what's specific to
[`06_rgcn_minibatch_gpu.ipynb`](../notebooks/06_rgcn_minibatch_gpu.ipynb) on top of that.

### 13.1 Install `pyg-lib` (optional, but worth doing)

Notebook 6 trains via PyG's `NeighborLoader`, which samples each mini-batch's neighborhood on **CPU**
regardless of which device the model itself trains on. Without `pyg-lib`, PyG falls back to a pure-Python
sampler — correct, just slower per batch. Since this CPU-side sampling overhead is the single biggest
unknown in this notebook's time estimate (see 13.3 below), install it before a real run:
```bash
# match the URL to your installed torch + CUDA version (this matches the cu121 build from §11.3)
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

### 13.2 Run the notebook

Open [`06_rgcn_minibatch_gpu.ipynb`](../notebooks/06_rgcn_minibatch_gpu.ipynb) in this instance's JupyterLab
and run it top to bottom. Nothing else instance-specific is needed — the notebook already:
- keeps the full graph (`data`) on CPU and only moves each sampled mini-batch to `DEVICE`,
- picks up `cuda` automatically via `torch.cuda.is_available()` in its first cell (verify this prints `True`
  before letting a full run go — a silent CPU fallback would still work, just slowly, and easy to miss),
- saves results under the `"rgcn_minibatch"` key (separate from notebook 5's `"rgcn"`), so both remain in
  `results/` for the same-notebook comparison in its own Section 7.

### 13.3 What to check once it's running

The training loop prints and saves per-epoch wall-clock time (`epoch_sec`, in `history` and in the metrics
JSON) — this is a **measured** number, unlike the batch-count-based estimate in
[`resource_requirements.md`](../resource_requirements.md) (10–50+ min for 60 epochs, deliberately wide
because `NeighborLoader` sampling overhead couldn't be sized from shapes alone the way full-batch memory
could). Worth feeding the real `epoch_sec` back into that doc once you have it, the same way notebook 5's
actual OOM corrected that doc's memory estimate earlier.

Also sanity-check that mini-batch val AUC lands close to notebook 5's full-batch AUC (Section 8 of notebook
6 discusses this) — a large gap would suggest `NUM_NEIGHBORS = [15, 10, 5]` is too aggressive a cap for this
graph and worth widening, at the cost of larger (but still bounded) per-batch subgraphs.

### 13.4 Cost recap

At `n1-highmem-4` + T4's ~$0.60–0.90/hr (§11.4), even the pessimistic end of the time range above (~1 hour)
puts a full run well under $1 — **stop the instance right after** (`gcloud workbench instances stop
fraud-gnn-workbench-gpu --location=us-central1-a`) rather than leaving a GPU idle between runs, per §11.4's
existing guidance.
