# hermes-body

A Reachy Mini front end for [Nous Research's hermes-agent](https://github.com/NousResearch/hermes-agent), modeled on [ClawBody](https://github.com/tomrikert/clawbody).

OpenAI Realtime is the embodied "brain" (sub-second voice loop, robot motion tools); hermes-agent is the "knowledge backend" reachable via a single `ask_hermes` tool. Cross-channel memory stays in sync via post-turn syncs.

See [PLAN.md](./PLAN.md) for the build plan and [reports/2026-05-03-hermes-body-reachy-mini-frontend.md](./reports/2026-05-03-hermes-body-reachy-mini-frontend.md) for the architectural rationale.

## Install

Prereqs: [mise](https://mise.jdx.dev/getting-started/installation.html) and Docker Desktop. Mise will install Python 3.11 itself.

### 1. Local install (Mac, with sim)

```bash
git clone https://github.com/the-focus-ai/hermes-body
cd hermes-body
mise trust                  # one-time: allow this project's mise.toml
mise install                # provisions Python 3.11.x and the .venv
mise run install            # installs reachy-mini[mujoco] + hermes-body editable
```

The `.venv` is auto-activated by mise whenever you `cd` into the project. After this you can use either `mise run <task>` or call `hermes-body`/`python` directly.

### 2. Run hermes-agent in Docker

hermes-body talks to hermes-agent over its OpenAI-compatible HTTP API. State lives in a Docker named volume (`hermes_volume`) so config, memories, and skills survive container rebuilds.

#### 2a. First-time setup (interactive wizard)

```bash
docker run -it --rm \
  -v hermes_volume:/opt/data \
  nousresearch/hermes-agent setup
```

This walks you through choosing an LLM provider (OpenRouter, DeepSeek, Anthropic, …), pasting that provider's API key, and selecting a default model. Settings are written to `/opt/data/.env` and `/opt/data/config.yaml` inside the volume.

#### 2b. Enable the OpenAI-compatible API server

The setup wizard does NOT enable the API server by default — hermes-body needs you to add four `API_SERVER_*` env vars to the volume's `.env`:

```bash
# Generate an auth key for hermes-body
KEY=hb-dev-$(openssl rand -hex 16)
echo "Save this for your .env later: $KEY"

# Append the API_SERVER_* block to /opt/data/.env (idempotent)
docker run --rm -v hermes_volume:/opt/data alpine sh -c "
grep -q '^API_SERVER_ENABLED=' /opt/data/.env || cat >> /opt/data/.env <<EOF

# === hermes-body API server ===
API_SERVER_ENABLED=true
API_SERVER_KEY=$KEY
API_SERVER_HOST=0.0.0.0
API_SERVER_PORT=8642
EOF
"
```

#### 2c. Run the gateway

**Interactive** (foreground; logs in your terminal, Ctrl+C to stop):

```bash
docker run -it --rm \
  -v hermes_volume:/opt/data \
  -p 8642:8642 \
  nousresearch/hermes-agent gateway run
```

**As a service** (detached, auto-restarts on crash or boot):

```bash
docker run -d --name hermes --restart unless-stopped \
  -v hermes_volume:/opt/data \
  -p 8642:8642 \
  nousresearch/hermes-agent gateway run
```

#### 2d. Verify

```bash
curl -fsS http://localhost:8642/v1/models -H "Authorization: Bearer $KEY"
# → {"object":"list","data":[{"id":"hermes-agent",...}]}
```

If you get connection refused, the API server vars from 2b weren't picked up — check `docker run --rm -v hermes_volume:/opt/data alpine tail /opt/data/.env`.

### 3. Configure `.env`

```bash
cp .env.example .env
$EDITOR .env
# Required:
#   OPENAI_API_KEY=sk-...                        # your OpenAI key (Realtime + TTS)
#   HERMES_API_KEY=<the KEY from step 2>
# Optional (defaults shown):
#   HERMES_BASE_URL=http://localhost:8642/v1     # change for remote Hermes
#   OPENAI_REALTIME_MODEL=gpt-realtime
#   OPENAI_VOICE=cedar
#   HEAD_TRACKER_TYPE=mediapipe
```

### 4. Verify the bridge

```bash
mise run smoke-hermes        # ask + get_agent_context + sync_turn
```

### 5. Run locally (sim)

```bash
# Terminal 1 (must be a real terminal — mjpython needs the macOS main thread):
mise run sim

# Terminal 2:
mise run gradio              # browser UI at http://localhost:7860 (recommended for sim)
# or
mise run run                 # console mode, uses laptop mic
```

## CLI flags

```bash
hermes-body                       # console mode, robot mic
hermes-body --gradio              # browser UI at localhost:7860
hermes-body --no-hermes           # disable ask_hermes tool
hermes-body --head-tracker yolo   # YOLO instead of MediaPipe
hermes-body --no-camera --no-face-tracking
```

## Deploy to a real Reachy Mini

The robot reaches your Mac's hermes-gateway over the LAN. Make sure macOS firewall allows inbound 8642 (System Settings → Network → Firewall) and the Hermes container binds to `0.0.0.0` (it does in the `docker run` above).

```bash
# Full first-time deploy: rsync code + pip install + push .env (rewriting
# HERMES_BASE_URL to your Mac's LAN IP) + verify reachability.
mise run deploy

# Subsequent code-only redeploys (~1 second):
mise run deploy-code

# Restart the running app on the robot via the dashboard API:
mise run robot-restart

# Tail clean hermes_body logs from the robot:
mise run robot-logs

# Open an SSH shell in ~/hermes-body on the robot:
mise run robot-shell

# Make hermes-body launch automatically on every boot:
mise run robot-install-autostart
mise run robot-uninstall-autostart   # if you change your mind
```

Configurable via env vars (defaults shown):

```bash
ROBOT_HOST=pollen@reachy-mini.local
ROBOT_PATH=hermes-body          # relative to remote $HOME
ROBOT_VENV=/venvs/apps_venv     # don't use mini_daemon's venv!
MAC_IFACE=en0                   # interface used to detect your LAN IP
HERMES_PORT=8642
```

After `mise run deploy`, the app shows up in the Reachy Mini dashboard at <http://reachy-mini.local:8000> and can be Run/Stop'd from there.

## mise tasks

```bash
mise tasks                # list all tasks

# Local install / dev
mise run install          # install runtime deps (incl. reachy-mini[mujoco])
mise run install-dev      # + pytest, ruff, mypy
mise run install-yolo     # + ultralytics, supervision (heavier face tracker)
mise run sim              # launch the MuJoCo simulator
mise run run              # hermes-body --debug
mise run gradio           # hermes-body --gradio --debug
mise run smoke-hermes     # smoke-test the Hermes bridge
mise run lint             # ruff check
mise run format           # ruff format
mise run clean            # nuke .venv + caches

# Robot deploy / management
mise run deploy                      # rsync + install + .env (with LAN IP)
mise run deploy-code                 # rsync only
mise run deploy-env                  # .env only
mise run robot-run                   # ssh + run hermes-body --debug
mise run robot-restart               # restart via dashboard API
mise run robot-shell                 # ssh into ~/hermes-body
mise run robot-logs                  # tail clean hermes_body logs
mise run robot-install-autostart     # systemd unit: start on boot
mise run robot-uninstall-autostart   # remove that unit
```

## Project layout

```
hermes-body/
├── mise.toml                    # tool versions + tasks
├── pyproject.toml               # python deps + entry points
├── systemd/
│   └── hermes-body-autostart.service   # installed by robot-install-autostart
└── src/hermes_body/
    ├── config.py                # env loading + validation
    ├── main.py                  # HermesBodyApp + HermesBodyCore + main()
    ├── hermes_bridge.py         # ask(), get_agent_context(), sync_turn()
    ├── openai_realtime.py       # OpenAIRealtimeHandler (Hermes-flavored)
    ├── moves.py                 # MovementManager + HeadLookMove
    ├── camera_worker.py         # CameraWorker + face tracking
    ├── gradio_app.py            # launch_gradio()
    ├── prompts.py               # ROBOT_BODY_INSTRUCTIONS + FALLBACK_IDENTITY
    ├── audio/
    │   ├── head_wobbler.py      # audio-driven head wobble
    │   └── filler.py            # "still working…" speech for slow tools
    ├── vision/                  # MediaPipe + YOLO face trackers
    └── tools/core_tools.py      # robot tools (look/camera/dance/emotion/…)
```

## Architecture

```
╭─────────────────╮     ╭──────────────────╮     ╭──────────────────╮
│ Reachy Mini mic │────▶│ OpenAI Realtime  │────▶│ Robot tools      │
╰─────────────────╯     │  • brain         │     │  (look/dance/…)  │
                        │  • voice loop    │     ╰──────────────────╯
                        │  • tool dispatch │     ╭──────────────────╮
                        │                  │────▶│ ask_hermes       │
                        ╰────────┬─────────╯     │  ↓ HTTP 8642     │
                                 │               │  ↓ Mac LAN       │
                                 ▼               │  hermes-agent    │
                        ╭──────────────────╮     │  Docker gateway  │
                        │ Reachy speakers  │     ╰──────────────────╯
                        ╰──────────────────╯
```

Continuous behaviors (face tracking, head wobble, breathing) run in local threads, never network-bound.
