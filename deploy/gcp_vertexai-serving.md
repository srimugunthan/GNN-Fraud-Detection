# Vertex AI Model Registry & Endpoints

Steps to register the trained model in **Vertex AI Model Registry** and serve it from a **Vertex AI
Endpoint**, instead of the plain-GCS + Cloud Run path in [`gcp_deployment.md`](gcp_deployment.md) /
[`gcp_workbench_training.md`](gcp_workbench_training.md). This is the heavier MLOps path those docs
mention only in passing — read the cost callout at the bottom before using it for anything but a demo.

Model Registry gives you a versioned `Model` resource with lineage instead of a bare GCS folder; Endpoints
give you managed autoscaling, request/response logging, and native traffic-splitting across model versions
(e.g. canary rollout) — none of which Cloud Run has out of the box.

---

## Does Vertex actually handle GNN serving, or just "a model"?

Yes and no — worth separating the two claims before following the steps below.

**Yes, in the "any custom container" sense.** Vertex AI Endpoints (custom container mode) is a thin,
generic HTTP wrapper around whatever image you upload. It doesn't inspect or care what architecture is
inside — it just calls your `/predict` route and expects `{"predictions": [...]}` back. It has no problem
running PyTorch Geometric, building a subgraph, and doing a forward pass, because that's exactly what
custom-container mode is for.

**No, it does not solve the actual hard part of GNN serving** — the same problem
[`gcp_deployment.md`](gcp_deployment.md)'s "The part that makes this different from a normal model
deploy" section already flags. Switching from Cloud Run to Vertex doesn't touch any of it:

- **No graph store.** Vertex has no product for adjacency/edges. Vertex Feature Store exists, but it's
  for per-entity tabular feature vectors/embeddings, not "give me this transaction's 2-hop neighborhood."
  You still have to build one of the three options already listed in `gcp_deployment.md` step 2 (static
  snapshot, Bigtable/Firestore adjacency index, or a real streaming graph store) yourself, inside your
  container, regardless of Cloud Run vs. Vertex.
- **No batching support for graphs.** A normal tensor model can sometimes batch multiple requests into
  one matmul for free. A GNN can't be batched that trivially — each item in an `instances: [...]` payload
  needs its own subgraph, and combining them requires `torch_geometric.data.Batch` logic you write
  yourself. Vertex's custom-container mode doesn't do this for you.
- **Model Monitoring won't understand graph drift.** Vertex's built-in feature-skew/drift monitoring
  assumes tabular feature distributions — it has no notion of "the neighborhood structure changed." You'd
  only get monitoring on whatever flat features you explicitly log.
- **Latency budget is still on you.** If subgraph construction involves a live lookup (Bigtable,
  Firestore, etc.) plus a forward pass, that all happens inside your request-handling code before you
  return a response — Vertex doesn't parallelize or cache any of it.

**A pattern worth knowing about:** a common workaround for this mismatch is to run the GNN
**offline/batch** to precompute node embeddings periodically, store those embeddings in Vertex Feature
Store (which *is* good at "look up this entity's vector fast"), and serve a small downstream classifier
(logistic regression/MLP on the embedding) online — sidestepping live subgraph construction entirely, at
the cost of embedding staleness. See
[`ADR-003`](../adr/ADR-003-precompute-gnn-embeddings-in-batch.md) for the tradeoffs of that approach for
this repo specifically.

**Bottom line:** the Cloud Run vs. Vertex Endpoint choice below is about deployment plumbing (autoscaling,
registry, traffic-split). The graph-serving problem is a separate, harder design decision that sits one
layer below both, and neither option solves it for you.

---

## 0. What has to be true first

- The serving image from `gcp_workbench_training.md` Section 7 (`serve/` + `Dockerfile`, built and pushed
  to Artifact Registry) already exists.
- The checkpoint + encoders are already in GCS (`gcp_workbench_training.md` Section 6 /
  `gcp_deployment.md` step 1), e.g. `gs://<PROJECT_ID>-fraud-gnn/models/v1/`.

**`serve/main.py` needs to satisfy Vertex's custom-container contract**, which is stricter than the ad hoc
`/predict` shape used against Cloud Run elsewhere in these docs:
- Request/response bodies are wrapped: Vertex Endpoints call `/predict` with `{"instances": [...]}` and
  expect `{"predictions": [...]}` back — the handler needs to unwrap `instances` and return `predictions`,
  not accept a flat transaction dict directly.
- Health and predict routes are whatever you declare at upload time (`/healthz`, `/predict` are fine —
  see `--container-health-route` / `--container-predict-route` below).
- The container should listen on the port passed via `--container-ports` (8080 below, matching the
  existing `Dockerfile`'s `CMD`).

If `serve/main.py` doesn't exist yet, write it per `gcp_deployment.md` step 3 with the envelope above in
mind before doing anything in this doc.

## 1. Register the model

This is the actual "registration" step — it creates a versioned `Model` resource in the registry:

```bash
gcloud ai models upload \
  --region=us-central1 \
  --display-name=fraud-gnn \
  --container-image-uri=us-central1-docker.pkg.dev/<PROJECT_ID>/fraud-gnn/serve:v1 \
  --artifact-uri=gs://<PROJECT_ID>-fraud-gnn/models/v1/ \
  --container-health-route=/healthz \
  --container-predict-route=/predict \
  --container-ports=8080
```

`--artifact-uri` is the same GCS folder used elsewhere — Vertex mounts it into the container at a known
path (`AIP_STORAGE_URI`) at serving time, so `serve/main.py` should load the checkpoint/encoders from
there rather than fetching them itself.

Verify:
```bash
gcloud ai models list --region=us-central1
```

## 2. Version subsequent retrains under the same model

Pass `--parent-model` so a retrain lands as a new version of the same `Model` instead of a separate one:

```bash
gcloud ai models upload \
  --region=us-central1 \
  --parent-model=<MODEL_ID> \
  --container-image-uri=us-central1-docker.pkg.dev/<PROJECT_ID>/fraud-gnn/serve:v2 \
  --artifact-uri=gs://<PROJECT_ID>-fraud-gnn/models/v2/ \
  --container-health-route=/healthz \
  --container-predict-route=/predict \
  --container-ports=8080
```

## 3. Deploy to an Endpoint

This is the part that actually gets you autoscaling + traffic-splitting:

```bash
gcloud ai endpoints create --region=us-central1 --display-name=fraud-gnn-endpoint

gcloud ai endpoints deploy-model <ENDPOINT_ID> \
  --region=us-central1 \
  --model=<MODEL_ID> \
  --display-name=fraud-gnn-v1 \
  --machine-type=n1-standard-4 \
  --min-replica-count=1 --max-replica-count=3 \
  --traffic-split=0=100
```

For a canary rollout of a new version, `deploy-model` a second model version to the same endpoint and
pass a `--traffic-split` that covers both deployed model IDs (e.g. `<ID1>=90,<ID2>=10`).

Call it:
```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://us-central1-aiplatform.googleapis.com/v1/projects/<PROJECT_ID>/locations/us-central1/endpoints/<ENDPOINT_ID>:predict" \
  -d '{"instances": [{"TransactionAmt": 34.0, "card1": 12345, "DeviceInfo": "DVC-771"}]}'
```

## 4. Batch prediction

No endpoint needed for offline scoring:

```bash
gcloud ai batch-prediction-jobs create \
  --region=us-central1 \
  --model=<MODEL_ID> \
  --job-display-name=fraud-gnn-batch-$(date +%s) \
  --gcs-source=gs://<PROJECT_ID>-fraud-gnn/batch-input/*.jsonl \
  --gcs-destination-prefix=gs://<PROJECT_ID>-fraud-gnn/batch-output/
```

## Cost caveat — read before deploying an endpoint

`--min-replica-count=1` keeps a node running 24/7; it does **not** scale to zero the way Cloud Run does.
Per the cost breakdown in `gcp_deployment.md`'s "Why Vertex AI is a different story" section, a small
CPU machine class runs roughly $0.10–0.11/hour ≈ **$75–80/month** just for the endpoint to exist, whether
it serves 10 requests or zero. Model Registry itself (Section 1–2 above) is not the expensive part —
storing a registered model costs nothing extra beyond the GCS artifact and image storage already priced
in `gcp_deployment.md`; the ongoing cost only shows up once you deploy it to an Endpoint (Section 3). If
you only want the registry/versioning story without the standing compute cost, stop after Section 2 and
keep using Cloud Run (`gcp_deployment.md`) for actual serving.
