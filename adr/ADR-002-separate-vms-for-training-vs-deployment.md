# ADR-002: Use Separate, Decoupled Compute for Training vs. Deployment on GCP

**Status:** Accepted
**Date:** 2026-09-14
**Deciders:** srimugunthan.d@gmail.com

---

## Context

The repo has two GCP-facing docs covering two different needs: [`deploy/gcp_training.md`](../deploy/gcp_training.md)
covers renting a bigger machine (Vertex AI Workbench, a raw Compute Engine VM, or Colab Enterprise — all
backed by an `n2-highmem-4`-class, 32GB-RAM VM) to get past local/Colab-free-tier OOM errors when running
[`notebooks/03_preprocessing_gnn_features.ipynb`](../notebooks/03_preprocessing_gnn_features.ipynb) and
[`notebooks/04_gcn_model.ipynb`](../notebooks/04_gcn_model.ipynb). [`deploy/gcp_deployment.md`](../deploy/gcp_deployment.md)
covers deploying the trained `FraudGNN` model as an inference endpoint, and recommends **Cloud Run**
(serverless containers, scale-to-zero) specifically because it keeps a light-usage (~10 calls) portfolio
deployment at an estimated **~$0.00–$0.30/month**, versus ~$75–80/month for an always-on Vertex AI endpoint.

Since both docs require GCP compute, and the training VM is already provisioned and set up with the repo
cloned and dependencies installed by the time a trained checkpoint exists, the question arose: can that
same VM just be reused as the thing that serves the model, instead of standing up a second, separate piece
of infrastructure?

## Decision

We will **not** reuse the training VM as the deployment target. Training compute
(`deploy/gcp_training.md`) and serving compute (`deploy/gcp_deployment.md`) will remain fully decoupled:
the training VM's job ends once `best_gnn_model.pt` and the fitted feature encoders are uploaded to GCS,
at which point it is stopped/deleted; serving runs separately and entirely on Cloud Run, built via a
managed Cloud Build job (`gcloud builds submit`) that needs no VM of its own to execute from.

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **Reuse the training VM as an always-on server** | Nothing new to provision; environment already has Python/deps installed | Training VM is `n2-highmem-4` (32GB RAM) sized for pandas preprocessing, wildly oversized for a small, low-QPS single-graph inference workload the deployment doc sizes at `--memory 2Gi --cpu 2`; billed 24/7 at ~$0.26/hr ≈ **~$187/month** just to exist, regardless of traffic — worse than the ~$75–80/month Vertex AI "always-on" case the deployment doc already flags as the expensive option to avoid |
| **Decoupled: train on a VM, serve via Cloud Run (chosen)** | Each workload sized and billed for what it actually needs — highmem VM only while actively preprocessing/training, scale-to-zero container for serving; matches the ~$0.00–0.30/month cost story already established in `gcp_deployment.md`; training VM can be torn down immediately after producing a checkpoint, with zero ongoing cost | Two pieces of infrastructure to think about instead of one; artifacts (checkpoint + encoders) must be explicitly handed off via GCS rather than living on the same disk |
| **Reuse the training VM only as a build machine** (install Docker there, build+push the serving image from it, but still deploy to Cloud Run) | Avoids installing Docker anywhere new | Unnecessary — `gcloud builds submit` already offloads the image build to the managed Cloud Build service; it doesn't need Docker, or any particular VM, installed anywhere the command is typed from (training VM, laptop, or Cloud Shell all work identically) |

## Consequences

**Positive:**
- Training compute and serving compute are each billed proportionally to actual usage: the highmem VM
  only while a human is actively running notebooks, Cloud Run only for the ~15 seconds of compute per
  request, scaling to zero between calls.
- The training VM can be stopped/deleted immediately after pushing the checkpoint to GCS with no impact on
  the (separately deployed) serving endpoint — the two lifecycles don't entangle.
- Keeps `gcp_deployment.md`'s ~$0.00–$0.30/month cost estimate for light usage actually true; it would be
  invalidated by leaving a 32GB VM running as the server.

**Negative / Accepted Tradeoffs:**
- Requires an explicit hand-off step (upload checkpoint + encoders to a GCS bucket) between the two docs'
  workflows, rather than everything living on one machine's disk.
- Two separate pieces of infrastructure to provision/track instead of one, though the training VM is
  ephemeral (stood up, used, torn down) while only the Cloud Run service is meant to persist.

## Follow-up Actions

- [ ] Add a short cross-reference note to `deploy/gcp_training.md` pointing at `deploy/gcp_deployment.md`
      for what happens after training, and stating explicitly not to reuse the training VM as the server.
- [ ] Add a matching note to `deploy/gcp_deployment.md` step 1 ("produce something to deploy") clarifying
      the checkpoint hand-off is via GCS, not a shared VM.
