# ADR-003: Precompute GNN Node Embeddings in Batch Instead of Live Subgraph Inference

**Status:** Proposed
**Date:** 2026-09-14
**Deciders:** srimugunthan.d@gmail.com

---

## Context

`FraudGNN` ([`gnn_fraud_detection.py`](../gnn_fraud_detection.py)) is a GCN-style model that scores a
transaction using both its own features and the structure of a graph built from shared
card/email/device/address edges ([`build_graph_edges`](../gnn_fraud_detection.py#L243)). Today the repo
only ever runs it batch-style over the whole static train+test graph at once — there is no code path that
scores one new transaction on its own.

[`deploy/gcp_deployment.md`](../deploy/gcp_deployment.md) step 2 and
[`deploy/gcp_vertexai-serving.md`](../deploy/gcp_vertexai-serving.md) ("Does Vertex actually handle GNN
serving, or just 'a model'?") both document what real-time serving would require: looking up a new
transaction's neighbors, pulling a 2–3 hop neighborhood, building a subgraph on the fly, and running
`model(x, edge_index)` per request. That needs a live graph/adjacency store (Bigtable/Firestore index, or
a full streaming graph store) the repo has no equivalent of, plus per-request subgraph-construction
latency, plus custom multi-request batching (`torch_geometric.data.Batch`) that no managed platform —
Cloud Run or Vertex AI Endpoints included — provides out of the box. None of that infrastructure exists in
this repo yet, and building it is a substantial project in its own right, not a deploy-config choice.

This ADR is a forward-looking design decision made ahead of building any of that: how should the GNN
actually be put behind a serving endpoint at all, before the live-graph-store option is undertaken.

## Decision

We will precompute GNN node embeddings for known entities (cards, devices, emails, accounts, recent
transactions) via a **periodic offline batch job**, persist those embeddings in a fast per-entity lookup
store, and serve a **small downstream classifier** (logistic regression or a small MLP) on the looked-up
embedding(s) plus the incoming transaction's own raw features at request time — instead of doing live
subgraph construction and a full GNN forward pass per request.

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **Live subgraph construction + full GNN forward pass per request** | No staleness — every request sees the current graph; architecturally "purest" GNN serving | Requires building a live graph/adjacency store that doesn't exist yet (Bigtable/Firestore index or streaming graph store); per-request neighbor lookup + subgraph build + forward pass adds real latency; multi-request batching needs bespoke `torch_geometric.data.Batch` logic no platform provides; not supported natively by Cloud Run or Vertex AI Endpoints (both are generic HTTP/container wrappers, not graph-aware) |
| **Precompute + serve embeddings via downstream classifier (chosen)** | Online path is a standard, non-graph-aware serving problem — works unmodified with Cloud Run or Vertex Endpoints; embedding lookup is a simple key→vector read (Feature Store/Bigtable/Redis), no live graph traversal at request time; downstream classifier is cheap to run and batch trivially; decouples GNN retraining cadence from serving uptime | Embeddings are only as fresh as the last batch run — a brand-new card/device/email has no embedding until the next cycle, requiring an explicit cold-start fallback (default embedding vector or raw-features-only scoring); introduces a batch pipeline (scheduling, monitoring, embedding-store writes) that doesn't exist in the repo yet; two models effectively in production (GNN + downstream classifier) instead of one |
| **Hybrid — precomputed embeddings for known entities, on-the-fly partial subgraph lookup only for genuinely new entities** | Covers the cold-start gap the chosen option accepts; better fraud recall on brand-new entities, which is exactly where fraud often concentrates | Needs *both* pieces of infrastructure — the batch embedding pipeline *and* a live (if narrower) subgraph store/lookup for the new-entity case — closer in effort to the live-inference option than the chosen one; meaningfully more complex to build and operate than either alternative alone |

## Consequences

**Positive:**
- Online serving becomes a standard ML-serving problem: a small classifier over a feature vector, which
  Cloud Run or Vertex AI Endpoints (per `deploy/gcp_deployment.md` / `deploy/gcp_vertexai-serving.md`)
  already handle well without any graph-specific serving code.
- Removes per-request GNN inference latency and the need for live multi-request graph batching from the
  request path entirely.
- GNN retraining/recomputation cadence (batch job) is decoupled from serving uptime — a slow or failed
  batch run degrades embedding freshness, not endpoint availability.
- Feature Store (or Bigtable/Redis) is a well-supported, off-the-shelf pattern for the "fast per-entity
  vector lookup" this approach actually needs, unlike graph adjacency, which has no equivalent managed
  primitive.

**Negative / Accepted Tradeoffs:**
- Embedding staleness is accepted: a brand-new card, device, or email has no embedding until the next
  batch cycle runs. This must be handled explicitly (a default/cold-start embedding, or falling back to
  raw-features-only scoring for unseen entities) rather than silently failing or scoring on stale/zeroed
  input.
- Two models are now in production instead of one (the GNN producing embeddings, and the downstream
  classifier consuming them), each needing its own versioning, monitoring, and retraining story.
- A batch pipeline (scheduler, embedding-store writes, freshness monitoring) is new infrastructure this
  repo does not currently have, and is a prerequisite for this decision to be realized, not a detail to
  defer.
- Fraud that concentrates in brand-new entities — often the highest-risk case — is the exact scenario this
  approach serves worst, since those entities are precisely the ones without a precomputed embedding yet.

## Compliance / Risk Notes

- Cold-start transactions effectively fall back to a weaker (non-graph-aware, or default-embedding) model.
  If this system were used for real fraud decisions, that degraded-accuracy segment should be explicitly
  measured and reported, not treated as an edge case — regulators/auditors reviewing a fraud model's
  performance would reasonably ask how new-entity transactions are handled differently.
- Embedding staleness windows (batch job cadence) should be documented and monitored so a stalled batch
  pipeline degrades gracefully (falls back to cold-start handling) rather than silently serving arbitrarily
  old embeddings without anyone noticing.

## Follow-up Actions

- [ ] Decide and document the cold-start fallback behavior (default embedding vector vs. raw-features-only
      scoring) before any implementation starts.
- [ ] Prototype the batch embedding job (schedule, source graph snapshot, entities covered) and pick an
      embedding-store backend (Vertex AI Feature Store vs. Bigtable/Redis) based on cost and latency, per
      the cost tradeoffs already noted in `deploy/gcp_vertexai-serving.md`.
- [ ] Define the downstream classifier's feature set (which entities' embeddings get concatenated per
      transaction, e.g. card + device + email) and retraining cadence relative to the GNN's.
- [ ] Add a cross-reference from `deploy/gcp_vertexai-serving.md`'s embedding-precompute callout to this
      ADR once follow-up work here progresses (already linked from that doc as of this writing).
- [ ] Revisit the hybrid alternative if new-entity fraud recall under the chosen approach proves
      unacceptable in practice.
