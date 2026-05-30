# OpenPI policy server deployment

`policy/pi05-openpi/pipeline/` contains the deployment wrapper for OpenPI's
official policy server. The robot runtime no longer starts a local bridge: it connects
directly to OpenPI's `WebsocketPolicyServer` through the lightweight
`openpi-client` package.

## Start the policy server

Use a separate terminal and the OpenPI environment. The checkpoint path is
required explicitly so a real-robot run cannot silently load the wrong
weights.

```bash
cd ~/A5
CHECKPOINT_DIR=/absolute/path/to/checkpoint \
  bash policy/pi05-openpi/pipeline/scripts/serve_openpi.sh
```

The wrapper runs the official command:

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_arx_finetune_single_task \
  --policy.dir=/absolute/path/to/checkpoint
```

Environment overrides:

| Variable | Default | Meaning |
|---|---|---|
| `CHECKPOINT_DIR` | required | OpenPI checkpoint directory |
| `OPENPI_ROOT` | `policy/pi05-openpi` | OpenPI checkout |
| `OPENPI_CONFIG` | `pi05_arx_finetune_single_task` | OpenPI training config |
| `OPENPI_PORT` | `8000` | Policy server port |

The official server binds `0.0.0.0`. In the current same-host deployment, the
robot client connects to `ws://localhost:8000`.

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

`--actions-per-chunk` is a runtime safety knob. When omitted, the client uses
`arx_pipeline/configs/pi05.yaml`, currently `50`. The OpenPI model returns 50
actions, while the robot client locally timestamps committed actions at
`rollout.exec_hz`.

## Adding another VLA

The extension boundary lives under `arx_pipeline/rollout/`:

- `policy_client.py` owns the remote RPC implementation.
- `policy_adapter.py` maps robot SHM observations to model inputs and model
  outputs back to ActionRing-ready chunks.

Add a client or adapter there instead of introducing another local gateway.
