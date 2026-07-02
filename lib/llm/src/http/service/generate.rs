// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP handler for the token-in/token-out `Generate` API
//! (`POST /inference/v1/generate`).
//!
//! This is an experimental engine-native endpoint, **disabled by default**;
//! opt in via the `enable_engine_apis` builder flag or the
//! `DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE` env var. When enabled it registers
//! a frontend-native handler that preserves the complete request in an opaque
//! backend envelope. Streaming (`stream=true`) remains unimplemented.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    middleware,
    response::{IntoResponse, Response},
    routing::post,
};
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
use serde::Serialize;
use tracing::Instrument;

use super::disconnect::create_connection_monitor;
use super::metrics::CancellationLabels;
use super::openai::{
    check_model_serving_ready, check_ready, context_from_headers, get_body_limit,
    get_or_create_request_id, smart_json_error_middleware,
};
use super::{RouteDoc, service_v2};
use crate::protocols::common::preprocessor::PreprocessedRequest;
use crate::protocols::common::{SamplingOptions, StopConditions};
use crate::protocols::openai::generate::{GenerateRequest, GenerateResponse};

/// vLLM-style nested error body: `{"error": {"message", "type", "code"}}`.
#[derive(Serialize, Debug)]
struct GenerateError {
    error: GenerateErrorBody,
}

#[derive(Serialize, Debug)]
struct GenerateErrorBody {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: u16,
}

/// Create an Axum [`Router`] for the token-in/token-out `Generate` endpoint.
/// If no path is provided, the default path is `/inference/v1/generate`.
pub fn generate_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/inference/v1/generate".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(handler_generate))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

/// Build a vLLM-style nested-`error` response.
fn generate_error_response(code: StatusCode, error_type: &str, message: String) -> Response {
    (
        code,
        Json(GenerateError {
            error: GenerateErrorBody {
                message,
                error_type: error_type.to_string(),
                code: code.as_u16(),
            },
        }),
    )
        .into_response()
}

/// vLLM's default when `sampling_params.max_tokens` is omitted.
const DEFAULT_MAX_TOKENS: u64 = 16;

/// Project routing controls while retaining the complete client request in
/// `extra_args.vllm_tito`. The backend remains the authority for interpreting
/// every vLLM-specific field.
fn preprocessed_from_generate(
    request: &GenerateRequest,
    model: &str,
) -> anyhow::Result<PreprocessedRequest> {
    let sampling = &request.sampling_params;
    let max_tokens = sampling
        .get("max_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let min_tokens = sampling
        .get("min_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let ignore_eos = sampling
        .get("ignore_eos")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let max_tokens = u32::try_from(max_tokens)
        .map_err(|_| anyhow::anyhow!("sampling_params.max_tokens exceeds u32"))?;
    let min_tokens = u32::try_from(min_tokens)
        .map_err(|_| anyhow::anyhow!("sampling_params.min_tokens exceeds u32"))?;

    PreprocessedRequest::builder()
        .model(model.to_string())
        .token_ids(request.token_ids.clone())
        .stop_conditions(StopConditions {
            max_tokens: Some(max_tokens),
            min_tokens: Some(min_tokens),
            ignore_eos: Some(ignore_eos),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            n: Some(1),
            ..Default::default()
        })
        .output_options(Default::default())
        .routing(Some(crate::protocols::common::preprocessor::RoutingHints {
            expected_output_tokens: Some(max_tokens),
            ..Default::default()
        }))
        .extra_args(Some(serde_json::json!({
            "vllm_tito": serde_json::to_value(request)?,
        })))
        .build()
        .map_err(|error| anyhow::anyhow!("failed to build PreprocessedRequest: {error}"))
}

/// Resolve, route, and dispatch a frontend-native token-in/token-out request.
async fn handler_generate(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<GenerateRequest>,
) -> Response {
    if let Err(response) = check_ready(&state) {
        return response.into_response();
    }

    if request.stream {
        return generate_error_response(
            StatusCode::NOT_IMPLEMENTED,
            "not_implemented",
            "streaming (stream=true) is not implemented for /inference/v1/generate yet".to_string(),
        );
    }

    let model = match &request.model {
        Some(model) => model.clone(),
        None => {
            let models = state.manager().list_generate_models();
            match models.len() {
                1 => models.into_iter().next().unwrap(),
                0 => {
                    return generate_error_response(
                        StatusCode::NOT_FOUND,
                        "not_found",
                        "no generate-capable model is registered".to_string(),
                    );
                }
                _ => {
                    return generate_error_response(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        "multiple models are registered; specify `model` in the request"
                            .to_string(),
                    );
                }
            }
        }
    };

    if let Err(response) = check_model_serving_ready(&state, &model) {
        return response.into_response();
    }

    let (engine, _parsing) = match state.manager().get_generate_engine_with_parsing(&model) {
        Ok(engine) => engine,
        Err(error) => {
            let (status, error_type) = match error {
                crate::discovery::ModelManagerError::ModelUnavailable(_) => {
                    (StatusCode::SERVICE_UNAVAILABLE, "service_unavailable")
                }
                _ => (StatusCode::NOT_FOUND, "not_found"),
            };
            return generate_error_response(status, error_type, error.to_string());
        }
    };

    let preprocessed = match preprocessed_from_generate(&request, &model) {
        Ok(preprocessed) => preprocessed,
        Err(error) => {
            return generate_error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                error.to_string(),
            );
        }
    };

    let request_id = request
        .request_id
        .clone()
        .unwrap_or_else(|| get_or_create_request_id(&headers));
    let context: Context<PreprocessedRequest> =
        match context_from_headers(preprocessed, request_id.clone(), &headers) {
            Ok(context) => context,
            Err(response) => return response.into_response(),
        };
    let engine_context = context.context();
    let cancellation_labels = CancellationLabels {
        model: state.manager().metric_model_for(&model).to_string(),
        endpoint: super::metrics::Endpoint::Generate.to_string(),
        request_type: "unary".to_string(),
    };
    let (mut connection_handle, _stream_handle) = create_connection_monitor(
        engine_context,
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

    let response = match tokio::spawn(
        generate_dispatch(engine, context, request_id, model, state.clone()).in_current_span(),
    )
    .await
    {
        Ok(response) => response,
        Err(error) => generate_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            format!("generate task panicked: {error}"),
        ),
    };

    connection_handle.disarm();
    response
}

async fn generate_dispatch(
    engine: crate::types::openai::generate::GenerateStreamingEngine,
    context: Context<PreprocessedRequest>,
    request_id: String,
    model: String,
    state: Arc<service_v2::State>,
) -> Response {
    let stream = match engine.generate(context).await {
        Ok(stream) => stream,
        Err(error) => {
            if super::metrics::request_was_rejected(error.as_ref()) {
                state
                    .metrics_clone()
                    .inc_rejection(&model, super::metrics::Endpoint::Generate);
                return generate_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "service_unavailable",
                    format!("engine rejected the request: {error:#}"),
                );
            }
            return generate_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                format!("failed to generate: {error:#}"),
            );
        }
    };

    let engine_context = stream.context();
    match GenerateResponse::from_annotated_stream(stream, request_id.clone()).await {
        Ok(response) => {
            if engine_context.is_killed() {
                return generate_error_response(
                    StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
                    "request_cancelled",
                    "request was cancelled".to_string(),
                );
            }
            if !response.is_complete_unary() {
                tracing::error!(%request_id, "generate stream ended without a complete choice");
                return generate_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    format!("generation produced no complete choice for {request_id}"),
                );
            }
            Json(response).into_response()
        }
        Err(error) => {
            tracing::error!(%request_id, %error, "failed to fold generate stream");
            generate_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                format!("failed to fold generate stream for {request_id}"),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::service_v2::{HttpService, VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV};
    use super::*;
    use tokio_util::sync::CancellationToken;

    /// Spin up an `HttpService` bound to an ephemeral port and return the port
    /// plus the run handle. Mirrors the reqwest-based router tests in
    /// `service_v2`.
    async fn serve(enable_generate: Option<bool>) -> (u16, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        let builder = HttpService::builder().port(port);
        let builder = match enable_generate {
            Some(enabled) => builder.enable_engine_apis(enabled),
            None => builder,
        };
        let service = builder.build().unwrap();
        let cancel_token = CancellationToken::new();
        let handle = tokio::spawn(async move {
            service.run_with_listener(cancel_token, listener).await.ok();
        });
        // Give the server a moment to start listening.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        (port, handle)
    }

    #[tokio::test]
    async fn generate_route_no_model_returns_structured_404() {
        let (port, handle) = serve(Some(true)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
            .send()
            .await
            .expect("generate request failed");
        assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "not_found");
        handle.abort();
    }

    #[tokio::test]
    async fn generate_route_streaming_returns_501() {
        let (port, handle) = serve(Some(true)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[1,2,3],"sampling_params":{},"stream":true}"#)
            .send()
            .await
            .expect("generate request failed");
        assert_eq!(resp.status().as_u16(), StatusCode::NOT_IMPLEMENTED.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "not_implemented");
        handle.abort();
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn generate_route_404_by_default() {
        temp_env::async_with_vars(
            [(VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV, None::<&str>)],
            async {
                let (port, handle) = serve(None).await;
                let resp = reqwest::Client::new()
                    .post(format!("http://localhost:{}/inference/v1/generate", port))
                    .header("content-type", "application/json")
                    .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
                    .send()
                    .await
                    .expect("generate request failed");
                assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
                handle.abort();
            },
        )
        .await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn generate_route_is_registered_when_enabled_by_env() {
        temp_env::async_with_vars(
            [(VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV, Some("1"))],
            async {
                let (port, handle) = serve(None).await;
                let resp = reqwest::Client::new()
                    .post(format!("http://localhost:{}/inference/v1/generate", port))
                    .header("content-type", "application/json")
                    .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
                    .send()
                    .await
                    .expect("generate request failed");
                assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
                let body: serde_json::Value = resp.json().await.expect("json body");
                assert_eq!(body["error"]["type"], "not_found");
                handle.abort();
            },
        )
        .await;
    }

    #[test]
    fn complete_request_envelope_reaches_preprocessed_request_unchanged() {
        let raw = serde_json::json!({
            "request_id": "req-forward",
            "token_ids": [1, 2],
            "sampling_params": {
                "max_tokens": 8,
                "future_sampling_field": {"nested": true}
            },
            "model": "test-model",
            "stream": false,
            "stream_options": {"include_usage": true},
            "features": {"future_feature": [1, 2, 3]},
            "priority": 7,
            "kv_transfer_params": {"remote": "worker-a"},
            "future_top_level_field": {"anything": "works"}
        });
        let request: GenerateRequest =
            serde_json::from_value(raw.clone()).expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(&request, "test-model").expect("build request");
        let envelope = preprocessed
            .extra_args
            .as_ref()
            .and_then(|extra| extra.get("vllm_tito"))
            .expect("vllm_tito envelope");

        assert_eq!(envelope, &raw);
    }
}
