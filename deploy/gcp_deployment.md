# Deploying the GNN Model on GCP

Notes on what it would take to deploy `FraudGNN` (from [gnn_fraud_detection.py](gnn_fraud_detection.py)) on
Google Cloud and run inference against it, plus a cost estimate for light (~10 call) usage.

## The part that makes this different from a normal model deploy

Your [`FraudGNN.forward()`](gnn_fraud_detection.py#L371) takes `(x, edge_index)` — a feature matrix *and*
the edges connecting those nodes. A brand-new transaction only becomes scoreable once you've:

1. Looked up which existing transactions share its card/email/device/address
   ([`build_graph_edges`](gnn_fraud_detection.py#L243)),
2. Pulled in a 2–3 hop neighborhood around it (what `NeighborLoader` does at
   [train time](gnn_fraud_detection.py#L443)),
3. Built a little subgraph on the fly, and only then run the forward pass.

So "deploy the model" really means "deploy the model **+ a live graph/feature store**" — that's the piece
the current repo has no equivalent of (it only ever runs on the whole static train+test graph at once,
batch-style, in `main()`).

## Two viable shapes on GCP

| | **Cloud Run** (custom container) | **Vertex AI** (custom container endpoint) |
|---|---|---|
| Effort | Lower — you own a FastAPI app | Higher — Vertex conventions (health/predict routes, model registry) |
| Gets you | A working HTTPS inference endpoint | + autoscaling tuned for ML, model versioning, request/response logging, monitoring/drift tooling |
| Right for | A working demo / portfolio piece, like what's in `demo/` today | If you want this to look like a "real" production MLOps setup |

For a repo at this stage (no saved checkpoint, no Docker, no cloud config yet), start with **Cloud Run**
and only move to Vertex if you specifically want the MLOps surface area on your resume/demo. Steps below
are for that path; Vertex divergence is noted where it matters.

## Steps

**1. Produce something to deploy**
- Run `gnn_fraud_detection.py::main()` (or `notebooks/04_gcn_model.ipynb`) end-to-end so `best_gnn_model.pt`
  actually exists (right now `results/` is empty and no checkpoint is committed).
- Also persist, alongside the weights: the fitted feature encoders/scalers from `engineer_features()`, the
  final `in_channels` count, and `Config` values (`HIDDEN_DIM`, `NUM_LAYERS`, `DROPOUT`) — the serving code
  needs these to reconstruct `FraudGNN(...)` before loading `state_dict`.
- Upload the checkpoint + encoders to a GCS bucket (e.g. `gs://<project>-fraud-gnn/models/v1/`).

**2. Decide how the graph store works at serving time** (the real design decision)

Pick one, roughly in order of effort:
- **Static snapshot (easiest, good for a demo):** freeze the existing IEEE-CIS graph, load it once into the
  container's memory (or Memorystore/Redis) at startup, and "infer" by attaching the incoming transaction to
  its matching neighbors in that frozen graph. Honest about not handling truly new entities, but enough to
  prove the deployment works end-to-end.
- **Precomputed adjacency index in Bigtable/Firestore:** `card1 → [recent txn node ids]`, `device → [...]`,
  etc., refreshed by a batch job. Lookup is fast, still not real-time-write.
- **Full streaming graph store:** transactions land in Bigtable/Spanner as they happen, adjacency updates
  live. This is a real system, not a demo weekend.

For a portfolio deployment, go with the static snapshot and say so explicitly (same honesty already in the
`demo/README.md`).

**3. Write the serving code**

A small FastAPI app, `serve/main.py`:
- `/healthz` — liveness.
- `/predict` — accepts a raw transaction JSON, runs `engineer_features()`-equivalent transforms, looks up
  neighbors from step 2, builds a `torch_geometric.data.Data` subgraph, runs `model(x, edge_index)`, returns
  the sigmoid score for the query node.
- Load model + graph store once at process startup (module-level globals), not per-request.

**4. Containerize — and pin PyG carefully**

This is the most common place these deploys break: `torch-geometric`'s compiled extensions
(`torch-scatter`, `torch-sparse`) are matched to an exact `torch` version + CPU/CUDA build.

```dockerfile
FROM python:3.11-slim
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install torch-geometric==2.5.3
RUN pip install fastapi uvicorn google-cloud-storage pandas scikit-learn
COPY serve/ /app/serve/
CMD ["uvicorn", "serve.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

CPU inference is fine here — this model is small and requests are single-graph, low-QPS; a GPU would be
overkill unless you're batch-scoring.

**5. Push and deploy**

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>

gcloud artifacts repositories create fraud-gnn --repository-format=docker --location=us-central1

gcloud builds submit --tag us-central1-docker.pkg.dev/<PROJECT_ID>/fraud-gnn/serve:v1

gcloud run deploy fraud-gnn-serve \
  --image us-central1-docker.pkg.dev/<PROJECT_ID>/fraud-gnn/serve:v1 \
  --region us-central1 \
  --memory 2Gi --cpu 2 \
  --set-env-vars MODEL_URI=gs://<project>-fraud-gnn/models/v1/ \
  --allow-unauthenticated   # or drop this and use IAM/auth for anything real
```

*(Vertex AI path instead of the last command: `gcloud ai models upload --container-*` +
`gcloud ai endpoints create` + `gcloud ai endpoints deploy-model` — same image, Vertex's routing/health
conventions.)*

**6. Call it**

```bash
curl -X POST https://<cloud-run-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 34.0, "card1": 12345, "DeviceInfo": "DVC-771", ...}'
# → {"fraud_score": 0.86}
```

**7. Ops basics worth adding before calling it "deployed"**
- Cloud Monitoring dashboard + log-based alert on 5xx / latency.
- A `/predict` smoke test in CI before each deploy.
- Version the model in the GCS path (`models/v2/…`) and make `MODEL_URI` a Cloud Run revision env var so
  rollback is just redeploying the old revision.

## Cost estimate — 10 API calls, max

Assuming the Cloud Run path above (`--memory 2Gi --cpu 2`, scale-to-zero).

| Item | Usage for 10 calls | Free tier / month | Cost |
|---|---|---|---|
| vCPU time | ~10 requests × ~15s (generous, incl. cold starts) × 2 vCPU ≈ 300 vCPU-sec | 180,000 vCPU-sec free | **$0.00** |
| Memory time | ~300 GiB-sec | 360,000 GiB-sec free | **$0.00** |
| Requests | 10 | 2,000,000 free | **$0.00** |
| Egress (tiny JSON responses) | a few KB | 1 GiB free | **$0.00** |

10 calls doesn't even round up past the free tier's rounding error — you'd need >100,000x that traffic in
the same month before Cloud Run charges a cent for compute.

**The one cost that isn't zero** is for the container *existing*, independent of how many times you call it:
- Artifact Registry image storage: a PyTorch + torch-geometric CPU image typically lands around 1.5–3 GB.
  First 0.5 GB/month is free, then $0.10/GB/month → roughly **$0.10–$0.30/month**, billed every month the
  image stays in the registry, whether the endpoint is called 10 times or never.
- GCS bucket holding the model checkpoint: a few MB at $0.02/GB/month → fractions of a cent, rounds to
  **$0.00**.

**Total: ~$0.00–$0.30**, and that upper bound is really "cost of storing a Docker image for a month," not
"cost of 10 predictions." Delete the image and Cloud Run service after demoing and it goes to exactly $0.

### Why Vertex AI is a different story

A deployed Vertex AI online-prediction endpoint keeps a minimum of 1 replica running 24/7 — it does **not**
scale to zero. A small CPU machine type (n1-standard-2 class) runs roughly **$0.10–$0.11/hour**, which is
**~$75–$80/month** just for the endpoint to exist, whether it serves 10 requests or zero. For "make 10
calls," Vertex would be ~250x more expensive than Cloud Run for no benefit — this is exactly the case where
the simpler deploy path above is also the financially obvious one.

**Bottom line:** use Cloud Run, tear the service (and ideally the image) down after testing, and this whole
exercise costs pocket change — realistically under $0.30 for a month of leaving the image sitting there, and
near-zero if cleaned up same-day.

Sources: [Cloud Run pricing](https://cloud.google.com/run/pricing) ·
[Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) ·
[Artifact Registry pricing](https://cloud.google.com/artifact-registry/pricing)
