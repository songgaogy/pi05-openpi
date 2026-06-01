# OpenPI policy server deployment

`policy/pi05-openpi/pipeline/` contains the deployment wrapper for OpenPI's official policy server. The robot runtime no longer starts a local bridge: it connects directly to OpenPI's `WebsocketPolicyServer` through the lightweight `openpi-client` package.

## Start the policy server

Use a separate terminal and the OpenPI environment. The checkpoint path is required explicitly so a real-robot run cannot silently load the wrong weights.

```bash
cd ~/A5
CHECKPOINT_DIR=/absolute/path/to/checkpoint/10000 \
  bash policy/pi05-openpi/pipeline/scripts/serve_openpi.sh
```

The wrapper runs the official command:

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  --warmup=ARX \
  policy:checkpoint \
  --policy.config=pi05_arx_finetune_single_task \
  --policy.dir=/absolute/path/to/checkpoint
```

Environment overrides:

| Variable | Default | Meaning |
|---|---|---|
| `CHECKPOINT_DIR` | required | OpenPI step directory (e.g. `.../10000`, not `.../10000/params`) |
| `OPENPI_ROOT` | `policy/pi05-openpi` | OpenPI checkout |
| `OPENPI_CONFIG` | `pi05_arx_finetune_single_task` | OpenPI training config |
| `OPENPI_PORT` | `8000` | Policy server port |
| `OPENPI_WARMUP` | `1` | Run dummy ARX inference until latency stabilizes before opening the WebSocket port |

The official server binds `0.0.0.0`. In the current same-host deployment, the robot client connects to `ws://localhost:8000`.

To enable Real-Time Chunking (smoother chunk transitions / async-latency robustness), set `RTC_ENABLE=1` plus the `RTC_*` variables. See [RTC implementation & usage detail and protocol](#rtcreal-time-chunking-implementation--usage-detail-and-protocol).

The launcher warms up JAX inference before the WebSocket server starts listening. For RTC this compiles both the first
vanilla chunk and subsequent RTC chunk paths, then resets server-side policy state. Set `OPENPI_WARMUP=0` only when
startup latency matters more than the first robot action latency.

## Start the robot client

Install the official lightweight client once in the robot environment:

```bash
pip install -e policy/pi05-openpi/packages/openpi-client
```

Then start rollout from another terminal:

```bash
PROMPT="put_shrimp_in_pot" \
  bash arx_pipeline/scripts/run_client.sh --actions-per-chunk 4
```

`--actions-per-chunk` is a runtime safety knob. When omitted, the client uses `arx_pipeline/configs/pi05.yaml`, currently `50`. The OpenPI model returns 50 actions, while the robot client locally timestamps committed actions at `rollout.exec_hz`.

## Adding another VLA

The extension boundary lives under `arx_pipeline/rollout/`:

- `policy_client.py` owns the remote RPC implementation.
- `policy_adapter.py` maps robot SHM observations to model inputs and model outputs back to ActionRing-ready chunks.

Add a client or adapter there instead of introducing another local gateway.

# RTC(real-time-chunking) implementation & usage detail and protocol

Real-Time Chunking (RTC) makes the policy condition each newly generated action chunk on the previously generated one, so chunk boundaries are smooth and the policy is robust to inference latency (the robot keeps executing the tail of the old chunk while the next one is being computed). The implementation follows the official reference in `third_party/real-time-chunking-kinetix/src/model.py` ([Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339), and the training-time variant [Training-Time Action Conditioning](https://arxiv.org/abs/2512.05964)), ported to pi05's flow-matching time convention. RTC is JAX-only.

## TL;DR usage

Server (RTC off by default — existing behavior is unchanged):

```bash
RTC_ENABLE=1 \
RTC_EXECUTE_HORIZON=25 \
RTC_INFERENCE_DELAY=1 \
RTC_METHOD=auto \
CHECKPOINT_DIR=/absolute/path/to/checkpoint \
  bash policy/pi05-openpi/pipeline/scripts/serve_openpi.sh
```

Client — `--actions-per-chunk` MUST equal the server's `RTC_EXECUTE_HORIZON`:

```bash
PROMPT="put_shrimp_in_pot" \
  bash arx_pipeline/scripts/run_client.sh --actions-per-chunk 25
```

> If the client keeps executing the full 50-action chunk before re-requesting (`--actions-per-chunk 50`), then `prefix_attention_horizon = 50 - 50 = 0` and RTC has no effect. RTC only helps when the client re-requests *before* exhausting the chunk (i.e. `execute_horizon < action_horizon`).

## Server environment variables

| Variable | Default | Meaning |
|---|---|---|
| `RTC_ENABLE` | `0` | `1` wraps the policy with server-side RTC; `0` keeps the plain policy |
| `RTC_EXECUTE_HORIZON` | `25` | Actions the client executes between inference calls; must match client `--actions-per-chunk` |
| `RTC_INFERENCE_DELAY` | `1` | Number of leading actions already committed (frozen) while a new chunk is computed |
| `RTC_METHOD` | `auto` | `auto` / `none` / `pinv` / `hard` (see below) |
| `RTC_PREFIX_ATTENTION_SCHEDULE` | `exp` | Prefix weight decay: `linear` / `exp` / `ones` / `zeros` (`pinv` only) |
| `RTC_MAX_GUIDANCE_WEIGHT` | `5.0` | Cap on the pinv guidance weight (`pinv` only) |

These map to the `realtime:realtime` tyro subcommand of `scripts/serve_policy.py`:

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_arx_finetune_single_task \
  --policy.dir=/absolute/path/to/checkpoint \
  realtime:realtime \
  --realtime.execute-horizon=25 \
  --realtime.inference-delay=1 \
  --realtime.method=auto
```

## Methods

- `none`: vanilla flow sampling, no conditioning on the previous chunk (RTC effectively disabled, but state is still tracked).
- `pinv`: **inference-time RTC** (paper 1). Soft guidance via the pseudoinverse-corrected denoiser velocity. Works with any existing finetuned checkpoint — no retraining required.
- `hard`: **training-time RTC** (paper 2). Hard-masks the frozen prefix to the previous chunk at the clean timestep. Only meaningful for a model trained with `rtc_simulated_delay` (otherwise the model never learned to consume a clean prefix).
- `auto`: picks `hard` if the loaded checkpoint's config has `rtc_simulated_delay` set, otherwise `pinv`.

## Parameters

- `execute_horizon` (E): how many actions are consumed per inference cycle. Smaller E = more frequent inference = stronger smoothing but higher compute.
- `inference_delay` (d): how many leading actions are assumed already committed and are therefore fully frozen. Must satisfy `0 <= d <= E`.
- `prefix_attention_horizon`: computed internally as `action_horizon - E` (`= 50 - E` for the arx config). It is the index past which the previous chunk is ignored. Prefix weights are `1` on `[0, d)`, decay to `0` over `[d, 50-E)`, and are `0` afterwards.

## Protocol / data flow (server-stateful)

```
robot client (actions-per-chunk = E)
        | obs
        v
WebsocketPolicyServer  --reset() on new connection-->  clears stored chunk
        |
        v
RealtimePolicy (openpi/policies/policy.py)
   - first call:   vanilla sample_actions, store returned output-space chunk
   - later calls:  prev_model = input_transform(prev_output, current_obs)
                   prev_shifted = concat(prev_model[:, E:], zeros[:, :E])
                   actions = model.sample_actions_rtc(
                       obs, prev_shifted, inference_delay=d,
                       prefix_attention_horizon=50-E, method=...)
                   hard mode: restore actions[:, :d] from prev_shifted[:, :d]
                   store returned output-space chunk
        | full 50-action chunk
        v
robot client executes first E actions, then re-requests
```

Key protocol points:

1. **Stateful server.** The previously returned output-space chunk is kept on the server. Before RTC conditioning, the input transforms encode it back into model space relative to the latest observation. This is required for policies such as ARX that model delta actions relative to the current state.
2. **Alignment by shifting.** Each new call shifts the re-encoded chunk by `execute_horizon` so index `0` aligns with the first action to generate. This assumes the client executed exactly `E` actions since the last call — hence `--actions-per-chunk == RTC_EXECUTE_HORIZON`.
3. **Episode reset.** `WebsocketPolicyServer` calls `policy.reset()` on every new connection, clearing the stored chunk. Start a new connection per episode (or call `reset()`) so RTC does not carry stale history across rollouts.
4. **First call.** With no history, the first inference falls back to plain sampling; conditioning starts from the second chunk.
5. **No wire-protocol change.** The obs/response msgpack payloads are unchanged; all RTC state lives server-side. The only client-side requirement is the `--actions-per-chunk` value.
6. **Synchronous-client hard prefix.** Training-time RTC masks prefix loss, so the sampler's final prefix values are not executable predictions. Since the current robot client executes the returned chunk from index `0`, the server restores the known clean prefix before returning and caching a hard-RTC chunk.

## Training-time RTC (the `hard` method)

To use `hard` at inference, the checkpoint must be fine-tuned with simulated delay so the model learns to denoise the suffix given a clean prefix:

1. In the training config, set `rtc_simulated_delay` on the model, e.g. `model=pi0_config.Pi0Config(pi05=True, action_horizon=50, rtc_simulated_delay=5)`.
2. Fine-tune (ideally from the existing checkpoint). `compute_loss` then randomly freezes a prefix of up to `rtc_simulated_delay` actions at the clean timestep and masks their loss.
3. Serve with `RTC_METHOD=auto` (resolves to `hard`) or `RTC_METHOD=hard`.

The `pinv` method needs no retraining and can be used on the current checkpoints directly.

## Implementation map

| File | Change |
|---|---|
| `src/openpi/models/pi0.py` | `get_prefix_weights`, per-token time in `embed_suffix`, training-time RTC in `compute_loss`, new `sample_actions_rtc` (`none`/`pinv`/`hard`) |
| `src/openpi/models/gemma.py` | `RMSNorm` adaRMS supports per-token conditioning |
| `src/openpi/models/pi0_config.py` | `Pi0Config.rtc_simulated_delay` |
| `src/openpi/policies/policy.py` | `RealtimePolicy` stateful wrapper (shift prev chunk, call RTC, `reset()`) |
| `src/openpi/policies/policy_config.py` | `RealtimeConfig`, `create_trained_policy(realtime=...)`, `auto` method selection |
| `src/openpi/serving/websocket_policy_server.py` | `reset()` on new connection |
| `scripts/serve_policy.py` | `realtime:realtime` subcommand |
| `pipeline/scripts/serve_openpi.sh` | `RTC_*` env passthrough |
| `src/openpi/models/pi0_rtc_test.py` | regression tests |
