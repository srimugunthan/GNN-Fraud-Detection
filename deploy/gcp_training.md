# Running the Notebooks on a Bigger Machine (GCP)

The raw IEEE-CIS CSVs are ~1.3GB on disk, but pandas working sets for this pipeline (merge → concat
train+test → feature engineering → graph construction) commonly run 3–8x that once everything is loaded
and held in memory at once. If [`notebooks/03_preprocessing_gnn_features.ipynb`](../notebooks/03_preprocessing_gnn_features.ipynb)
or [`notebooks/04_gcn_model.ipynb`](../notebooks/04_gcn_model.ipynb) are getting killed for memory locally
or on a free Colab runtime, the fix is usually just more RAM. This doc covers two ways to get that on GCP.
(See [gcp_deployment.md](gcp_deployment.md) for the separate concern of deploying the *trained* model as an
inference endpoint — this doc is about training/notebook compute, not serving.)

## What Vertex AI Workbench is

**Vertex AI Workbench** is GCP's managed JupyterLab environment. Under the hood it's still a regular
Compute Engine VM that you pick the size of (same machine types, same per-hour pricing as any VM) — the
"managed" part is everything *around* the VM:

- JupyterLab, Python, and common ML libraries come **preinstalled** on the boot image — no manual
  `apt-get`/`pip install jupyter` setup step.
- Browser access to the notebook is routed through **GCP's own IAM-authenticated proxy**. You don't open a
  firewall rule or maintain an SSH tunnel to reach it — anyone with the right IAM role on the project can
  open it with a URL, and no one else can, without you managing auth yourself.
- It integrates with the rest of Vertex AI (model registry, other GCP data services) if you later want that,
  though nothing here requires it.

The tradeoff versus a plain Compute Engine VM is a small per-hour management surcharge on top of raw VM
pricing, in exchange for skipping the manual environment setup and the SSH-tunnel security bookkeeping.
Both options are below — pick Workbench if you want less setup, pick the raw VM if you want the cheapest
possible option and don't mind a few extra `apt`/`pip` commands and an SSH tunnel.

---

## Option A: Vertex AI Workbench (managed, less setup)

### 1. Enable the required APIs (one-time per project)
```bash
gcloud services enable notebooks.googleapis.com compute.googleapis.com
```

### 2. Create the Workbench instance
```bash
gcloud workbench instances create fraud-gnn-workbench \
  --location=us-central1-a \
  --machine-type=n2-highmem-4 \
  --data-disk-size=100 \
  --data-disk-type=PD_SSD
```
- `n2-highmem-4` = 4 vCPU / 32 GB RAM — should comfortably clear the OOM you're hitting locally/on Colab.
  Bump to `n2-highmem-8` (64 GB) if 32 GB still isn't enough once running.
- `--data-disk-size=100` (GB) holds the repo + raw CSVs + processed artifacts.
- Exact flag names can drift slightly between `gcloud` versions — run
  `gcloud workbench instances create --help` if any flag is rejected, and adjust.

This takes a few minutes to provision (it boots a preconfigured JupyterLab image — no manual Python/Jupyter
install like Option B).

**Or via Console**, if you'd rather click through: Vertex AI → Workbench → Instances → **Create New** →
pick machine type (`n2-highmem-4`) and disk size → Create.

### 3. Open JupyterLab
```bash
gcloud workbench instances describe fraud-gnn-workbench \
  --location=us-central1-a --format="value(proxyUri)"
```
Open the returned URL in your browser — it's routed through GCP's IAM-authenticated proxy, so only your
Google account (with the right IAM role, e.g. `roles/notebooks.admin` or `roles/notebooks.runner`) can
reach it. Nothing to firewall or tunnel manually. (Or from the Console: Workbench → Instances → "Open
JupyterLab" next to your instance.)

### 4. Get the repo and data onto the instance
Open a terminal from inside JupyterLab (File → New → Terminal) and run:
```bash
git clone https://github.com/srimugunthan/GNN-Fraud-Detection.git
cd GNN-Fraud-Detection
pip install -r requirements.txt
pip install torch torch-geometric   # for notebook 04
```
Data — either:
```bash
pip install kaggle
# upload kaggle.json via the JupyterLab file-upload UI first, then:
python download_dataset.py --unzip
```
or upload your local zip straight into the running instance's disk via the JupyterLab UI, or
`gcloud compute scp` to the instance's underlying VM name (Workbench instances are backed by a normal
Compute Engine VM you can also `scp` to directly — find its name in the Console instance details, or via
`gcloud workbench instances describe`).

### 5. Run the notebooks
Navigate to `notebooks/04_gcn_model.ipynb` (or whichever was OOM-ing) in the JupyterLab file browser and
run it — same code, just on a machine with real RAM this time.

### 6. Control cost — stop it when idle, start it again when needed
Workbench instances **do not scale to zero automatically** by default; they bill for compute the whole
time they're running, whether you're actively using JupyterLab or not. Stopping ≠ deleting: the instance,
its disk, and everything on it (your cloned repo, downloaded data, notebook state on disk) are preserved —
you're just pausing the compute bill.

**Stop it** whenever you step away (end of a work session, lunch, overnight):
```bash
gcloud workbench instances stop fraud-gnn-workbench --location=us-central1-a
```
A stopped instance still incurs a small disk-storage charge (see below) but **zero compute charge**.

**Start it again** the next time you want to work:
```bash
gcloud workbench instances start fraud-gnn-workbench --location=us-central1-a
```
Give it a minute to boot, then re-fetch the JupyterLab URL the same way as step 3 (the proxy URL can change
across stop/start cycles):
```bash
gcloud workbench instances describe fraud-gnn-workbench \
  --location=us-central1-a --format="value(proxyUri)"
```
Everything from your last session — repo, data, saved notebooks — is still on disk, so you pick up right
where you left off. (Or via Console: Workbench → Instances → select the instance → **Start**/**Stop**
buttons at the top.)

**Delete it** once you're fully done with the project (this is the only way to stop paying for the disk
too):
```bash
gcloud workbench instances delete fraud-gnn-workbench --location=us-central1-a
```

Optional: set an **idle shutdown** timer at creation time (check
`gcloud workbench instances create --help` for the exact flag/units on your `gcloud` version) so it
auto-stops after N minutes of inactivity — worth doing so a forgotten tab doesn't rack up hours between
manual stops.

**Cost estimate**: `n2-highmem-4` ≈ $0.26/hr compute + a modest managed-notebooks surcharge on top of raw
Compute Engine pricing + ~100GB pd-ssd storage (~$0.17/GB-month while the instance exists, whether running
or stopped). A few hours of active work ≈ **$2–5**; delete the instance afterward and storage cost goes to
zero too.

---

## Option B: Raw Compute Engine VM (cheapest, most control)

### 1. Pick a machine size
| Machine type | vCPUs | RAM | Cost (us-central1, on-demand) |
|---|---|---|---|
| `n2-highmem-4` | 4 | 32 GB | ~$0.26/hr |
| `n2-highmem-8` | 8 | 64 GB | ~$0.52/hr |

Start with `n2-highmem-4`.

### 2. Create the VM
```bash
gcloud config set project <PROJECT_ID>

gcloud compute instances create fraud-gnn-notebook \
  --zone=us-central1-a \
  --machine-type=n2-highmem-4 \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd
```
(Bump `--boot-disk-size` if you're keeping raw CSVs + processed artifacts on-disk too — 100GB is generous
headroom.)

### 3. SSH in and set up the environment
```bash
gcloud compute ssh fraud-gnn-notebook --zone=us-central1-a
```
Then on the VM:
```bash
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git
git clone https://github.com/srimugunthan/GNN-Fraud-Detection.git
cd GNN-Fraud-Detection
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt jupyter
pip install torch torch-geometric   # for notebook 04
```

### 4. Get the data onto the VM
```bash
pip install kaggle
# upload kaggle.json via `gcloud compute scp` first, then:
python download_dataset.py --unzip
```
or, faster if you already have it locally, skip Kaggle entirely and copy your local zip straight up:
```bash
# from your local machine, not the VM:
gcloud compute scp ieee-cis-dataset/ieee-fraud-detection.zip \
  fraud-gnn-notebook:~/GNN-Fraud-Detection/ieee_cis_dataset/ --zone=us-central1-a
```

### 5. Launch Jupyter — without exposing it publicly
```bash
jupyter notebook --no-browser --port=8888
```
Leave that running, then **from your local machine**, open an SSH tunnel:
```bash
gcloud compute ssh fraud-gnn-notebook --zone=us-central1-a -- -L 8888:localhost:8888
```
Now open `http://localhost:8888` in your local browser (using the token Jupyter printed) — traffic goes
over the SSH tunnel, nothing is exposed to the internet. Don't open a firewall rule for port 8888 to
`0.0.0.0/0` — that's the common mistake that gets a Jupyter instance scanned and hit within minutes.

### 6. Stop or delete when done
```bash
gcloud compute instances stop fraud-gnn-notebook --zone=us-central1-a   # keep disk, pause compute billing
# or, once truly done:
gcloud compute instances delete fraud-gnn-notebook --zone=us-central1-a
```
Stopping a highmem VM still bills for the persistent disk (~$0.02–0.04/GB-month for the 100GB pd-ssd, so a
few dollars/month if left stopped) — delete it if you're not coming back soon.

**Cost estimate**: `n2-highmem-4` at ~$0.26/hr, a few hours of EDA + training ≈ **$1–3** if you delete/stop
it afterward.

---

## Option C: Colab Enterprise (Colab's UI, on GCP compute)

**Colab Enterprise** is GCP's answer to "I want Colab's UI, not JupyterLab's, but with my own GCP project's
compute instead of Colab's opaque free/Pro tiers." It's part of Vertex AI: same Colab notebook look and
feel as `colab.research.google.com`, but the VM behind it is one you pick (same machine types as Options A
and B above), billed as standard GCP compute, and gated by GCP IAM instead of a Google account tier.

| | Consumer Colab | Colab Enterprise (this option) | Vertex AI Workbench (Option A) |
|---|---|---|---|
| UI | Colab notebook UI | Same Colab notebook UI | JupyterLab UI |
| Compute | Google-managed, opaque, free/Pro tiers | Your GCP project's VMs — you pick the machine type | Your GCP project's VMs |
| RAM ceiling | Capped by tier (free ~12GB, Pro+ ~50GB) | Whatever machine type you choose | Same |
| Billing | Colab subscription | Standard GCP compute pricing, only while a runtime is active | Standard GCP compute pricing + small managed surcharge |

### 1. Enable the API
```bash
gcloud services enable aiplatform.googleapis.com
```

### 2. Create a Runtime Template
Defines the machine type/disk a runtime will use. Via Console: Vertex AI → Colab Enterprise → Runtime
Templates → **New Template**:
- Machine type: `n2-highmem-4` (32 GB) or `n2-highmem-8` (64 GB) — same sizing as Options A/B
- Disk size: 100 GB
- Idle shutdown: set a timeout (e.g. 60 min) so a forgotten tab doesn't rack up cost

### 3. Create/start a Runtime
From that template — either automatically when you first open a notebook and click "Connect," or
explicitly beforehand via Runtimes → **New Runtime** → select your template.

### 4. Create or upload a notebook
Colab Enterprise → Notebooks → **New Notebook**, or upload one of
[`notebooks/03_preprocessing_gnn_features.ipynb`](../notebooks/03_preprocessing_gnn_features.ipynb) /
[`notebooks/04_gcn_model.ipynb`](../notebooks/04_gcn_model.ipynb) directly. Connect it to your running
runtime (top-right "Connect" dropdown — same gesture as consumer Colab).

### 5. Set up the environment and data — in a code cell
Same as consumer Colab:
```python
!git clone https://github.com/srimugunthan/GNN-Fraud-Detection.git
%cd GNN-Fraud-Detection
!pip install -r requirements.txt torch torch-geometric

!pip install kaggle
import os
os.environ["KAGGLE_USERNAME"] = "..."
os.environ["KAGGLE_KEY"] = "..."
!python download_dataset.py --unzip
```

### 6. Run the notebook

### 7. Stop the runtime when done
Console → Colab Enterprise → Runtimes → Stop, so compute billing stops. Delete the runtime template too
if you're fully done, though templates themselves don't incur cost — only active runtimes do.

**Cost estimate**: same per-hour math as Options A/B (`n2-highmem-4` ≈ $0.26/hr while the runtime is
active) — Colab Enterprise doesn't add the small managed-notebooks surcharge that Workbench does, since
it's a lighter wrapper around the VM.

---

## Which one to pick

| | Vertex AI Workbench | Raw Compute Engine VM | Colab Enterprise |
|---|---|---|---|
| Setup effort | Lower — JupyterLab preinstalled, no SSH tunnel | Higher — manual Python/Jupyter setup, manual SSH tunnel | Lower — Colab UI preconfigured, no SSH tunnel |
| UI | JupyterLab | JupyterLab (self-installed) | Colab notebook UI |
| Access model | GCP IAM-authenticated proxy URL | SSH tunnel you manage yourself | GCP IAM-authenticated, Colab UI |
| Cost | Raw VM price + small managed-notebooks surcharge | Cheapest — raw VM price only | Raw VM price, no extra surcharge |
| Best for | One-off/occasional use, prefer JupyterLab | Repeated use, cost-sensitive, comfortable with SSH | Already used to Colab's UI/workflow |

For a single training run to get past an OOM, all three work — pick based on which UI you'd rather use
(JupyterLab vs. Colab) and how much setup friction you're willing to tolerate for a marginally lower cost
(raw VM).
