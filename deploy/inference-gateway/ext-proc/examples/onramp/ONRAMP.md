<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Router on-ramp (raw vLLM workers, no Dynamo control plane)

This directory is the **smallest possible** way to put Dynamo's KV/load-aware
router in front of an existing **vanilla `vllm serve`** fleet that already sits
behind a Gateway-API gateway with the Inference Extension (GAIE).

If you already run GAIE + vLLM, you **replace only your EndpointPicker (EPP)**
with the Dynamo EPP and you are done — no Dynamo operator, no Dynamo runtime,
no Dynamo worker.

| | This on-ramp (router-only mode) | Full Dynamo (full-dynamo-stack mode) |
|---|---|---|
| Workers | stock `vllm serve` | Dynamo workers (vLLM/SGLang/TRT-LLM) |
| Discovery | K8s pod reflector; membership comes from the `InferencePool` (selector + target port) | Dynamo discovery (etcd/K8s) |
| KV events | per-pod vLLM ZMQ → the in-process `SelectionCore` indexer subscribes directly | Dynamo event plane |
| Runtime | none — runtime-free (no etcd, no NATS, no event plane) | etcd + NATS |
| Operator | not required | required (DynamoGraphDeployment) |
| Tokenization | EPP tokenizes for routing; worker re-tokenizes | model-card preprocessor, no double tokenization |

The EPP is served in one of two modes, selected at startup by **`DYN_EPP_MODE`**
(`full-dynamo-stack` | `router-only`); the same binary serves both. Within
router-only mode the KV-aware selector runs in one of two backends, selected by
**`DYN_EPP_SELECTOR_MODE`** (`embedded` | `http`). **This on-ramp uses
`embedded`**: the EPP and a runtime-free `SelectionCore` are compiled into one
process, so there is no separate selection-service Deployment and no HTTP hop.
It is single-replica (the embedded selector holds its index/load state
in-process); use `http` mode with a replicated `dynamo.select_service` when you
need multiple EPP replicas to share selection state.

## Limitations

| Concern | Router-only mode, embedded selector (gap vs full Dynamo) |
|---|---|
Duplicate store/remove (vLLM "retries") | parity
In-stream ordering | parity
Transient disconnects | The in-process indexer reconnects on its own and keeps routing. KV-cache updates the worker sent during the gap are recovered from the worker's **replay** socket when `DYN_EPP_REPLAY_PORT` is set (and the vLLM worker exposes one); otherwise the index refreshes from new traffic.
Dropped events / gaps | The `SelectionCore` indexer does seq-watermark gap detection and replays missed events from the worker's replay socket when `DYN_EPP_REPLAY_PORT` is configured. Without a replay socket, gaps are dropped and the index re-warms from new traffic.
Initial cache state | A fresh or restarted EPP starts with an empty index and re-warms from live traffic (a single embedded selector has no peer to snapshot from; peer `/dump` recovery only exists in `http` mode with multiple replicas).
Backpressure / EPP restart | vLLM PUB drops to slow subscribers (ZMQ HWM) can be silently lost; on restart the index starts empty and re-warms from new traffic.
Data parallelism | V1 targets DP=1.
Disaggregated prefill/decode | Not supported by these PRs (aggregated: prefill and decode share one worker). Planned follow-up.
Multi-replica EPP | Not in this on-ramp — the embedded selector is single-process. Use `http` mode for replicated EPPs.

## Examples

If you are starting from scratch install the [Prerequisites](#1-prerequisites) and apply the provided example file:

- `agg.yaml` — aggregated: one vLLM pool, KV-aware load balancing, EPP with the **embedded** selector (single replica).

```bash
kubectl apply -n <ns> -f agg.yaml
```

Otherwise follow the steps below.

> Disaggregated prefill/decode serving is a planned follow-up and is not covered
> by this on-ramp.

## Example Progression from vLLM to vLLM + Dynamo + GAIE

### Initial Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-qwen
  labels:
    app: vllm-qwen
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm-qwen
  template:
    metadata:
      labels:
        app: vllm-qwen
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - "--model"
            - "Qwen/Qwen3-0.6B"
          ports:
            - name: http
              containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: "1"
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-qwen
spec:
  selector:
    app: vllm-qwen
  ports:
    - name: http
      port: 8000
      targetPort: 8000
```

### 1. Prerequisites

Set up a Gateway-API gateway + Inference Extension (GAIE) if you don't have one
yet. Follow the instructions for your gateway (e.g. [AgentGateway](https://agentgateway.dev/docs/kubernetes/main/quickstart/install/)) and [Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/guides/).

We provide a convenience script if you are installing from scratch for experimentation.
```bash
# Gateway API + Inference Extension CRDs + a gateway named `inference-gateway`
deploy/inference-gateway/scripts/install_gaie_crd_agentgateway.sh
```

Create the HuggingFace token secret the EPP uses to download the
tokenizer.

```bash
# HF token secret (the EPP downloads the tokenizer to tokenize for routing)
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=<your-token>
```

### 2. Create / Wire the InferencePool and HTTPRoute

Now that Gateway and GAIE is installed, we can create an InferencePool and an accompanying Dynamo EPP.

Point the `InferencePool` at your vLLM workers (see the
[`vllm-qwen-pool` InferencePool in `agg.yaml`](./agg.yaml#L222) for a complete example):

Specifically, these fields depend on the model you deploy, so make sure the settings below are adjusted to match your workers.

- `spec.selector` — labels matching your vLLM pods.
```yaml
  selector:
    matchLabels:
      app: vllm-qwen
```

- `spec.targetPorts[].number` — the vLLM serving port (usually `8000`).
```yaml
  targetPorts:
    - number: 8000
```

- `spec.endpointPickerRef` (a.k.a. `spec.extensionRef`) — the EPP service + port.
```yaml
  endpointPickerRef:
    kind: Service
    name: dynamo-epp
    port:
      number: 9002
```

Attach the `HTTPRoute` to the gateway and target the pool (see the
[`vllm-qwen-route` HTTPRoute in `agg.yaml`](./agg.yaml#L238) for a complete example):

- `spec.rules[].backendRefs[]` — targets the `InferencePool`.
```yaml
    - backendRefs:
        - group: inference.networking.k8s.io
          kind: InferencePool
          name: vllm-qwen-pool
          port: 8000
          weight: 1
```

- `spec.parentRefs[]` — your gateway (set `namespace` + `sectionName` when the
  gateway lives in another namespace, e.g. `agentgateway-system`).
```yaml
  parentRefs:
    - group: gateway.networking.k8s.io
      kind: Gateway
      name: inference-gateway
```

### 3. Update vLLM deployment

Add labels to your vLLM pods that match `InferencePool.spec.selector` (the EPP
reads the pool to learn which pods to watch — there is no separate selector env
var):

```yaml
metadata:
  name: vllm-qwen
  labels:
    app: vllm-qwen
```

or
```bash
kubectl -n <ns> label deployment <vllm-deployment> app=vllm-qwen --overwrite
```

### 4. Add the EPP to your deployment

Remove the vLLM router (if you have it already installed) and replace it with the EPP (Dynamo Router).
Add the EPP as a `Deployment` + `Service` (see the
[`dynamo-epp` Deployment in `agg.yaml`](./agg.yaml#L125) for a complete example) and set:

```yaml
kind: Deployment
metadata:
  name: dynamo-epp
  labels:
    app: dynamo-epp
```

1. **Set the EPP image.** The Dynamo EPP ships as the "frontend" image with each
   release. The default image supports both selector modes, so choosing
   `embedded` vs `http` is a pure runtime flag — no special build.

2. **Select router-only + embedded mode, and point at the pool.** The EPP reads
   the `InferencePool` it backs (the same object the gateway routes to) to learn
   the pod selector and HTTP target port — so no pod-selector/target-port env is
   needed. It watches pods in its own namespace (`POD_NAMESPACE`, via the
   downward API):

   ```yaml
   - name: DYN_EPP_MODE
     value: "router-only"
   - name: DYN_EPP_SELECTOR_MODE
     value: "embedded"          # in-process SelectionCore; no separate service
   - name: DYN_EPP_POOL_NAME
     value: "vllm-qwen-pool"    # the InferencePool this EPP backs
   - name: POD_NAMESPACE
     valueFrom:
       fieldRef:
         fieldPath: metadata.namespace
   ```

3. **Point at the workers' KV-event socket.** The worker publishes KV events via
   `--kv-events-config`; the in-process `SelectionCore` indexer subscribes on the
   matching port at each Ready pod's IP.

   vLLM worker arg:

   ```text
   --kv-events-config '{"enable_kv_cache_events":true,"endpoint":"tcp://*:5557"}'
   ```

   EPP env:

   ```yaml
   - name: DYN_EPP_KV_EVENT_PORT
     value: "5557"
   ```

4. **Set the model/tokenizer and block size.** The EPP downloads this tokenizer
   (HF id) to tokenize prompts for routing — it MUST match the model the workers
   serve — and the block size MUST equal the vLLM `--block-size`:

   ```yaml
   - name: DYN_MODEL_NAME
     value: "Qwen/Qwen3-0.6B"
   - name: DYN_KV_CACHE_BLOCK_SIZE
     value: "16"
   - name: HF_TOKEN
     valueFrom:
       secretKeyRef:
         name: hf-token-secret
         key: HF_TOKEN
   ```

5. **(Optional) Tune KV routing and recovery:**

   ```yaml
   # KV indexer thread pool for the embedded selector.
   - name: DYN_EPP_SELECTOR_THREADS
     value: "4"
   # ZMQ replay socket for gap recovery (only if the vLLM worker exposes one).
   - name: DYN_EPP_REPLAY_PORT
     value: "5558"
   # Scoring / queueing behavior use the standard router env, e.g.:
   - name: DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT
     value: "1.0"
   ```

   See the full list in the
   [environment contract](#router-only-mode-environment-contract) below.

In the end your Final vLLM deployment will look like below:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-qwen
  labels:
    app: vllm-qwen
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm-qwen
  template:
    metadata:
      labels:
        app: vllm-qwen
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - "--model"
            - "Qwen/Qwen3-0.6B"
            - "--served-model-name"     # NEW: pin the OpenAI model id
            - "Qwen/Qwen3-0.6B"
            - "--port"                  # NEW: explicit (8000 is vLLM's default)
            - "8000"
            - "--enable-prefix-caching" # NEW: required so vLLM emits prefix KV events
            - "--block-size"            # NEW: MUST equal the EPP's DYN_KV_CACHE_BLOCK_SIZE
            - "16"
            - "--kv-events-config"      # NEW: publish KV events on a ZMQ PUB socket for the EPP
            - '{"enable_kv_cache_events":true,"endpoint":"tcp://*:5557"}'
          ports:
            - name: http
              containerPort: 8000
            - name: kv-events           # NEW: KV-event PUB port the EPP subscribes to
              containerPort: 5557
          env:                          # NEW: HF token lets the worker pull the model (required for gated models)
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: HF_TOKEN
          resources:
            limits:
              nvidia.com/gpu: "1"
      tolerations:
        - effect: NoSchedule
          key: nvidia.com/gpu
          operator: Exists
```

## Test

```bash
# terminal 1
kubectl -n agentgateway-system port-forward svc/inference-gateway 8000:80

# terminal 2
GATEWAY_URL=http://localhost:8000

# quick health/smoke first
curl --max-time 20 -sS "$GATEWAY_URL/v1/models" | jq .

# then chat completion
curl --max-time 120 -sS "$GATEWAY_URL/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role":"user","content":"hello"}]
  }' | jq .

```

## router-only-mode environment contract

Required:

| Env | Meaning |
|---|---|
| `DYN_EPP_MODE=router-only` | Select router-only (on-ramp) mode; `full-dynamo-stack` (default) keeps Dynamo mode |
| `DYN_EPP_SELECTOR_MODE=embedded` | Run the selector in-process (this on-ramp). `http` uses a separate `dynamo.select_service` fleet |
| `DYN_EPP_POOL_NAME` | Name of the `InferencePool` this EPP backs; its selector + target port drive pod discovery |
| `DYN_MODEL_NAME` | Model id used to build the routing tokenizer (no model card in router-only mode) |
| `DYN_KV_CACHE_BLOCK_SIZE` | MUST equal vLLM `--block-size` |
| `POD_NAMESPACE` | EPP's own namespace to watch (inject via the downward API) |

Optional:

| Env | Default | Meaning |
|---|---:|---|
| `DYN_EPP_POOL_NAMESPACE` | `POD_NAMESPACE` | Namespace of the `InferencePool` |
| `DYN_EPP_KV_EVENT_PORT` | `5557` | vLLM `--kv-events-config` PUB port the selector subscribes to |
| `DYN_EPP_KV_EVENT_TOPIC` | `""` | vLLM `--kv-events-config` topic (accepted; topic filtering is not yet wired, so leave empty) |
| `DYN_EPP_REPLAY_PORT` | unset | Per-worker ZMQ replay socket for KV-event gap recovery (requires the vLLM worker to expose one) |
| `DYN_EPP_SELECTOR_THREADS` | `4` | KV indexer thread pool for the embedded selector |
| `DYN_DATA_PARALLEL_SIZE` | `1` | Engine data-parallel size (V1 targets 1) |
| `DYN_TOTAL_KV_BLOCKS` | unset | Per-worker total KV blocks hint for the load model |
| `DYN_MAX_NUM_BATCHED_TOKENS` | unset | Per-worker max batched tokens (needed only when selector queueing is enabled) |
| `DYN_ROUTER_*` | router defaults | Scoring/queueing knobs parsed by the standard router config (e.g. `DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT`) |
| `DYN_SECURE_SERVING` | `true` | EPP gRPC TLS toggle |

Not used in this mode (unlike the old inert-runtime design): `DYN_EPP_POD_SELECTOR`,
`DYN_EPP_TARGET_PORT` (both come from the `InferencePool`), `DYN_DISCOVERY_BACKEND`,
`DYN_EVENT_PLANE` (no Dynamo runtime/event plane), and the disagg knobs
(`DYN_EPP_ROLE_LABEL`, `DYN_ENFORCE_DISAGG`, `DYN_EPP_EMIT_PREFILLER_HOST_PORT`).

## How KV events reach the router without NATS

Vanilla vLLM publishes native KV-cache events on a ZMQ PUB socket
(`--kv-events-config`). In embedded router-only mode there is **no republisher and
no event plane**: for each Ready pod the EPP registers the pod's KV-event
endpoint (`tcp://<pod_ip>:<DYN_EPP_KV_EVENT_PORT>`) into the in-process
`SelectionCore`, whose indexer **subscribes directly** to that PUB socket and
maintains the KV/prefix index the selector scores against. When
`DYN_EPP_REPLAY_PORT` is set, the indexer also backfills missed events from the
worker's replay socket.
