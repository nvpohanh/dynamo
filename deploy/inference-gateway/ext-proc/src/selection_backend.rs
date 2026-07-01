// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend abstraction over the runtime-free selection service.
//!
//! Router-only mode can reach the selector two ways:
//!
//! - **HTTP** ([`crate::selector_client`], feature `selector-http`): the EPP is
//!   a thin client of one or more `python -m dynamo.select_service` replicas.
//!   Use this in production where the selector is replicated.
//! - **Embedded** ([`crate::embedded_selector`], feature `selector-embedded`):
//!   the EPP and a runtime-free `SelectionCore` are compiled into a single image
//!   and the EPP calls the selector's Rust API in-process — no HTTP client. Use
//!   this for single-replica evaluation.
//!
//! Both backends speak the same plain wire types defined here (which mirror the
//! documented JSON contract in
//! `docs/components/router/standalone-selection.md`), so the reflector, topology
//! adapter, and router are identical across modes.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// Worker registration payload (`POST /workers`). Only the fields router-only
/// mode populates are included; the selector defaults the rest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkerRegistration {
    pub worker_id: u64,
    pub model_name: String,
    pub endpoint: String,
    pub block_size: u32,
    pub data_parallel_size: u32,
    pub kv_events_endpoints: HashMap<u32, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_kv_blocks: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_num_batched_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stable_routing_id: Option<String>,
}

/// Partial worker update (`PATCH /workers/{id}`). Any `Some` field is applied.
#[derive(Debug, Clone, Default, Serialize)]
pub struct WorkerPatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_events_endpoints: Option<HashMap<u32, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_kv_blocks: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stable_routing_id: Option<String>,
}

/// Select-and-reserve request (`POST /select_and_reserve`). Selection and
/// booking are one operation: the selector picks a worker AND books the
/// request's load against it under `reservation_id`. Prompt fields are sent flat;
/// sending raw `token_ids` lets the selector compute block/sequence hashes.
#[derive(Debug, Clone, Serialize)]
pub struct SelectRequest {
    pub model_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection_id: Option<String>,
    /// Booking key. The EPP uses the gateway request id, so the later
    /// `free_reservation`/`prefill_complete` calls need no extra bookkeeping map.
    /// If `None`, the selector generates one (which the EPP would then have to
    /// read back from the response).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reservation_id: Option<String>,
    pub token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_worker_ids: Option<HashSet<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority_jump: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict_priority: Option<u32>,
}

/// Raw observability overlap summary (matched token counts).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OverlapSummary {
    #[serde(default)]
    pub longest_matched: u32,
    #[serde(default)]
    pub gpu: u32,
    #[serde(default)]
    pub cpu: u32,
    #[serde(default)]
    pub disk: u32,
}

/// Selection result returned by `/select_and_reserve`.
#[derive(Debug, Clone, Deserialize)]
pub struct SelectResponse {
    #[serde(default)]
    pub selection_id: Option<String>,
    /// The booking key the selector recorded (echoes the request's
    /// `reservation_id`, or the selector-generated one when it was omitted).
    #[serde(default)]
    pub reservation_id: Option<String>,
    pub worker_id: u64,
    pub dp_rank: u32,
    pub endpoint: String,
    pub block_size: u32,
    #[serde(default)]
    pub overlap: OverlapSummary,
    #[serde(default)]
    pub effective_prefill_tokens: usize,
}

/// A selection backend: worker-catalog reconciliation, query-only selection, and
/// readiness. Implemented by the HTTP selector fleet and the in-process embedded
/// core.
///
/// The backend owns the "actual" catalog state; the caller only supplies the
/// desired set. This lets the HTTP fleet reconcile each replica independently
/// (and bootstrap replicas that appear or restart) without the caller tracking
/// per-replica state.
#[tonic::async_trait]
pub trait SelectionBackend: Send + Sync + 'static {
    /// Drive the selector catalog toward `desired` (keyed by `worker_id`).
    /// Idempotent: safe to call repeatedly. The HTTP fleet applies the diff to
    /// every live replica; the embedded backend applies it to the in-process
    /// core.
    async fn reconcile(&self, desired: &HashMap<u64, WorkerRegistration>) -> anyhow::Result<()>;

    /// Select a worker for a prompt AND book its load in one operation
    /// (`POST /select_and_reserve`). The returned [`SelectResponse::reservation_id`]
    /// is the booking that must later be released with [`Self::free_reservation`].
    async fn select_and_reserve(&self, req: &SelectRequest) -> anyhow::Result<SelectResponse>;

    /// Release a booking (`DELETE /reservations/{id}`), removing the request from
    /// the selector's slot tracker / active-load accounting. Called when the
    /// gateway signals the response is complete. Idempotent: an unknown
    /// reservation (e.g. a body-less request that never booked) is treated as
    /// success.
    async fn free_reservation(&self, reservation_id: &str) -> anyhow::Result<()>;

    /// Release *prefill* tokens for a booking from a decode worker's load
    /// (`POST /reservations/{id}/prefill_complete`), called when the first token
    /// is generated.
    ///
    /// This is meaningful only for **disaggregated** serving, where prefill and
    /// decode run on different workers and the decode worker's load must drop the
    /// prefill contribution once prefill finishes. In **aggregated** serving
    /// (the only mode supported today) prefill and decode share one worker, so
    /// there is nothing to release and the EPP does not call this — see
    /// [`crate::epp_router::EppRouter::on_prefill_complete`]. Implemented now so
    /// the disaggregated path is ready when it lands.
    async fn prefill_complete(&self, reservation_id: &str) -> anyhow::Result<()>;

    /// Returns `true` once the selector can schedule at least one worker.
    async fn any_ready(&self) -> bool;

    /// Change signal the reconcile loop should also wake on, in addition to pod
    /// changes. The HTTP fleet bumps this when its selector replica set changes;
    /// the embedded backend never fires (its selector is in-process).
    fn subscribe_changes(&self) -> tokio::sync::watch::Receiver<u64>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_registration_serializes_expected_fields() {
        let mut kv = HashMap::new();
        kv.insert(0u32, "tcp://10.0.0.1:5557".to_string());
        let req = WorkerRegistration {
            worker_id: 42,
            model_name: "Qwen/Qwen3-0.6B".to_string(),
            endpoint: "http://10.0.0.1:8000".to_string(),
            block_size: 16,
            data_parallel_size: 1,
            kv_events_endpoints: kv,
            replay_endpoint: None,
            total_kv_blocks: None,
            max_num_batched_tokens: None,
            stable_routing_id: Some("vllm-0".to_string()),
        };
        let v: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["worker_id"], 42);
        assert_eq!(v["block_size"], 16);
        assert_eq!(v["kv_events_endpoints"]["0"], "tcp://10.0.0.1:5557");
        assert_eq!(v["stable_routing_id"], "vllm-0");
        assert!(v.get("replay_endpoint").is_none());
        assert!(v.get("total_kv_blocks").is_none());
    }

    #[test]
    fn select_request_sends_flat_prompt() {
        let req = SelectRequest {
            model_name: "m".to_string(),
            selection_id: Some("s1".to_string()),
            reservation_id: Some("req-42".to_string()),
            token_ids: vec![1, 2, 3],
            allowed_worker_ids: Some(HashSet::from([7u64])),
            priority_jump: None,
            strict_priority: None,
        };
        let v: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["token_ids"], serde_json::json!([1, 2, 3]));
        assert_eq!(v["selection_id"], "s1");
        assert_eq!(v["reservation_id"], "req-42");
        assert_eq!(v["allowed_worker_ids"], serde_json::json!([7]));
        assert!(v.get("priority_jump").is_none());
    }

    #[test]
    fn select_response_deserializes() {
        let body = serde_json::json!({
            "model_name": "m",
            "tenant_id": "default",
            "worker_id": 9,
            "dp_rank": 0,
            "endpoint": "http://10.0.0.1:8000",
            "block_size": 16,
            "overlap": {"longest_matched": 32, "gpu": 16, "cpu": 0, "disk": 0},
            "effective_prefill_tokens": 64
        });
        let resp: SelectResponse = serde_json::from_value(body).unwrap();
        assert_eq!(resp.worker_id, 9);
        assert_eq!(resp.dp_rank, 0);
        assert_eq!(resp.endpoint, "http://10.0.0.1:8000");
        assert_eq!(resp.effective_prefill_tokens, 64);
        assert_eq!(resp.overlap.longest_matched, 32);
    }
}
