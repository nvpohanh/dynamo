// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol types for the token-in/token-out `Generate` API
//! (`POST /inference/v1/generate`).
//!
//! These mirror vLLM's `GenerateRequest` / `GenerateResponse` wire contract
//! (`vllm/entrypoints/scale_out/token_in_token_out/protocol.py`). Only the
//! fields the frontend itself acts on are typed; `sampling_params` is kept
//! opaque (`serde_json::Value`) and every other field — including ones a newer
//! vLLM adds (`features`, `stream_options`, …) — is captured in `vllm_passthrough`
//! via `#[serde(flatten)]` and forwarded to the worker verbatim, so the wire
//! contract is forward-compatible: a new vLLM request field flows through with
//! no change here.

use std::collections::HashMap;

use anyhow::Result;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::protocols::Annotated;
use crate::protocols::common::llm_backend::LLMEngineOutput;

/// Token-in/token-out generation request.
///
/// Only the fields the frontend acts on are typed. `sampling_params` stays
/// opaque so the worker reconstructs vLLM's own `SamplingParams` verbatim; all
/// other fields (`priority`, `cache_salt`, `kv_transfer_params`, `features`, and
/// anything a newer vLLM adds) are captured in [`Self::vllm_passthrough`] via
/// `#[serde(flatten)]` and forwarded untouched — forward-compatible by
/// construction.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerateRequest {
    /// Client-supplied request id, echoed back. The server generates one if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,

    /// Pre-tokenized prompt. Required — this is the KV-routing input.
    pub token_ids: Vec<u32>,

    /// Opaque vLLM `sampling_params`; forwarded verbatim and parsed at the worker.
    pub sampling_params: serde_json::Value,

    /// Model / alias for worker selection. Optional (single-model deployments omit it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Streaming vs. unary response.
    #[serde(default)]
    pub stream: bool,

    /// Every other top-level field — vLLM's own params (`priority`, `cache_salt`,
    /// `kv_transfer_params`, `features`, …) plus anything a newer vLLM adds —
    /// captured verbatim and forwarded to the worker untouched. Keeps the
    /// contract forward-compatible with newer vLLM versions.
    #[serde(flatten)]
    pub vllm_passthrough: serde_json::Map<String, serde_json::Value>,
}

/// A single choice in a `GenerateResponse`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerateResponseChoice {
    pub index: u32,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<u32>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,

    pub finish_reason: Option<String>,
}

/// Token-in/token-out generation response.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerateResponse {
    pub request_id: String,

    pub choices: Vec<GenerateResponseChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_transfer_params: Option<serde_json::Value>,
}

/// Per-index accumulation state while folding a stream of
/// [`LLMEngineOutput`] deltas into a single [`GenerateResponse`].
struct GenerateChoiceAcc {
    index: u32,
    token_ids: Vec<crate::protocols::TokenIdType>,
    finish_reason: Option<String>,
}

/// Folds a stream of [`Annotated<LLMEngineOutput>`] deltas into a single
/// [`GenerateResponse`]. Each chunk carries a delta of newly generated
/// `token_ids` keyed by `index` (default 0).
struct GenerateAggregator {
    request_id: String,
    choices: HashMap<u32, GenerateChoiceAcc>,
    error: Option<String>,
}

impl GenerateAggregator {
    fn new(request_id: String) -> Self {
        Self {
            request_id,
            choices: HashMap::new(),
            error: None,
        }
    }

    async fn apply(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
        request_id: String,
    ) -> Result<GenerateResponse> {
        let aggregator = stream
            .fold(
                GenerateAggregator::new(request_id),
                |mut agg, delta| async move {
                    let delta = match delta.ok() {
                        Ok(delta) => delta,
                        Err(error) => {
                            agg.error = Some(error);
                            return agg;
                        }
                    };

                    if agg.error.is_none()
                        && let Some(output) = delta.data
                    {
                        let index = output.index.unwrap_or(0);
                        let choice = agg.choices.entry(index).or_insert(GenerateChoiceAcc {
                            index,
                            token_ids: Vec::new(),
                            finish_reason: None,
                        });
                        choice.token_ids.extend(output.token_ids);
                        if let Some(finish_reason) = output.finish_reason {
                            choice.finish_reason = Some(finish_reason.to_string());
                        }
                    }
                    agg
                },
            )
            .await;

        if let Some(error) = aggregator.error {
            return Err(anyhow::anyhow!(error));
        }

        let mut choices: Vec<GenerateResponseChoice> = aggregator
            .choices
            .into_values()
            .map(|acc| GenerateResponseChoice {
                index: acc.index,
                token_ids: Some(acc.token_ids),
                logprobs: None,
                finish_reason: acc.finish_reason,
            })
            .collect();
        choices.sort_by_key(|choice| choice.index);

        Ok(GenerateResponse {
            request_id: aggregator.request_id,
            choices,
            prompt_logprobs: None,
            kv_transfer_params: None,
        })
    }
}

impl GenerateResponse {
    /// Aggregate a raw engine stream for the non-streaming endpoint.
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
        request_id: String,
    ) -> Result<GenerateResponse> {
        GenerateAggregator::apply(stream, request_id).await
    }

    /// A complete unary response has at least one terminal choice.
    pub fn is_complete_unary(&self) -> bool {
        !self.choices.is_empty()
            && self
                .choices
                .iter()
                .all(|choice| choice.finish_reason.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn generate_request_deserializes_from_vllm_json() {
        let raw = json!({
            "request_id": "req-123",
            "token_ids": [1, 2, 3, 4],
            "sampling_params": {"temperature": 0.7, "max_tokens": 16},
            "model": "test-model",
            "stream": false,
            "cache_salt": "salt",
            "priority": 0,
            "kv_transfer_params": null
        });
        let req: GenerateRequest = serde_json::from_value(raw).expect("deserialize");
        assert_eq!(req.request_id.as_deref(), Some("req-123"));
        assert_eq!(req.token_ids, vec![1, 2, 3, 4]);
        assert!(!req.stream);
        assert_eq!(req.model.as_deref(), Some("test-model"));
    }

    #[test]
    fn generate_request_captures_unknown_fields() {
        // Untyped + future fields are CAPTURED in `vllm_passthrough` (forwarded
        // verbatim), not dropped — so a field a newer vLLM adds flows through.
        let raw = json!({
            "token_ids": [5, 6],
            "sampling_params": {},
            "priority": 7,
            "future_field": "kept"
        });
        let req: GenerateRequest = serde_json::from_value(raw).expect("deserialize");
        assert_eq!(req.token_ids, vec![5, 6]);
        assert!(!req.stream);
        assert_eq!(req.request_id, None);
        // Not typed on the struct → captured in `vllm_passthrough`.
        assert_eq!(req.vllm_passthrough.get("priority"), Some(&json!(7)));
        assert_eq!(
            req.vllm_passthrough.get("future_field"),
            Some(&json!("kept"))
        );
        // Round-trip: flattened fields serialize back at the top level.
        let back = serde_json::to_value(&req).expect("serialize");
        assert_eq!(back.get("priority"), Some(&json!(7)));
        assert_eq!(back.get("future_field"), Some(&json!("kept")));
    }

    #[test]
    fn generate_response_round_trips_without_usage_key() {
        let resp = GenerateResponse {
            request_id: "req-123".to_string(),
            choices: vec![GenerateResponseChoice {
                index: 0,
                token_ids: Some(vec![10, 11, 12]),
                logprobs: None,
                finish_reason: Some("stop".to_string()),
            }],
            prompt_logprobs: None,
            kv_transfer_params: None,
        };

        let value = serde_json::to_value(&resp).expect("serialize");
        assert!(
            value.get("usage").is_none(),
            "GenerateResponse must not emit a `usage` key"
        );
        assert!(value.get("prompt_logprobs").is_none());
        assert!(value.get("kv_transfer_params").is_none());

        let round: GenerateResponse =
            serde_json::from_value(value).expect("round-trip deserialize");
        assert_eq!(round.request_id, "req-123");
        assert_eq!(round.choices.len(), 1);
        assert_eq!(round.choices[0].token_ids, Some(vec![10, 11, 12]));
        assert_eq!(round.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn generate_response_accumulates_engine_deltas() {
        let chunks = vec![
            LLMEngineOutput {
                token_ids: vec![100],
                index: Some(0),
                ..Default::default()
            },
            LLMEngineOutput {
                token_ids: vec![101],
                index: Some(0),
                finish_reason: Some(crate::protocols::common::FinishReason::Length),
                ..Default::default()
            },
        ];
        let stream = futures::stream::iter(chunks.into_iter().map(Annotated::from_data));

        let response = GenerateResponse::from_annotated_stream(stream, "req-agg".to_string())
            .await
            .expect("aggregate engine deltas");

        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].token_ids, Some(vec![100, 101]));
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("length"));
        assert!(response.is_complete_unary());
    }

    #[tokio::test]
    async fn incomplete_engine_streams_are_not_complete_unary_responses() {
        let empty = futures::stream::iter(Vec::<Annotated<LLMEngineOutput>>::new());
        let empty_response =
            GenerateResponse::from_annotated_stream(empty, "req-empty".to_string())
                .await
                .expect("aggregate empty stream");
        assert!(!empty_response.is_complete_unary());

        let partial = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
            token_ids: vec![100],
            index: Some(0),
            ..Default::default()
        })]);
        let partial_response =
            GenerateResponse::from_annotated_stream(partial, "req-partial".to_string())
                .await
                .expect("aggregate partial stream");
        assert!(!partial_response.is_complete_unary());
    }
}
