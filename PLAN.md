# hermes-body — Build Plan

A Reachy Mini front end for Nous Research's hermes-agent, following ClawBody's architecture exactly: OpenAI Realtime is the embodied brain; hermes-agent is reached via a single `ask_hermes` tool. See `reports/2026-05-03-hermes-body-reachy-mini-frontend.md` for the architectural rationale.

## Decisions locked in

| Decision | Choice | Why |
|---|---|---|
| Hermes host | This Mac, LAN-only for v1 | Fastest debug loop; defer remote access |
| Robot/sim split | Both — sim default, robot when motion matters | Iterate on prompts in sim; validate on hardware |
| Secrets | `.env` in project root, gitignored | Standard, what ClawBody does |
| v1 scope | Full ClawBody parity | Voice + motion + camera + face tracking + head wobble + identity bootstrap + post-turn sync |
| Hermes backing model | OpenRouter | Try several models with one key |
| Identity | Dynamic from Hermes at startup | Robot speaks AS the Hermes agent; cross-channel memory |
| UI | CLI + Gradio (`--gradio` flag) | Browser mic invaluable for sim debugging |
| Realtime model | `gpt-realtime` (or `gpt-4o-realtime-preview` fallback) | Only realistic sub-second voice option |

## Architecture (one-liner)

`Reachy Mini mic → OpenAI Realtime (brain + tool dispatch) → either local robot tools OR ask_hermes → response → Realtime TTS → Reachy Mini speakers`. Continuous behaviors (face tracking, head wobble) run in local threads, never network-bound.

## File layout

```
hermes-body/
├── .env                       # gitignored — OPENAI_API_KEY, HERMES_*, etc.
├── .env.example               # committed template
├── .gitignore
├── pyproject.toml
├── README.md
├── PLAN.md                    # this file
├── reports/
│   └── 2026-05-03-hermes-body-reachy-mini-frontend.md
└── src/hermes_body/
    ├── __init__.py
    ├── config.py              # env loading + validation
    ├── main.py                # HermesBodyApp + HermesBodyCore + main()
    ├── hermes_bridge.py       # ask(), get_agent_context(), sync_turn()
    ├── openai_realtime.py     # OpenAIRealtimeHandler (port of clawbody)
    ├── moves.py               # MovementManager + HeadLookMove (lift)
    ├── camera_worker.py       # CameraWorker (lift)
    ├── gradio_app.py          # launch_gradio()
    ├── prompts.py             # ROBOT_BODY_INSTRUCTIONS + FALLBACK_IDENTITY
    ├── audio/
    │   ├── __init__.py
    │   └── head_wobbler.py    # HeadWobbler (lift)
    ├── vision/
    │   ├── __init__.py
    │   ├── mediapipe_tracker.py  # HeadTracker (MediaPipe variant, lift)
    │   └── yolo_head_tracker.py  # HeadTracker (YOLO variant, lift)
    └── tools/
        ├── __init__.py
        └── core_tools.py      # TOOL_SPECS, dispatch_tool_call, _handle_*
```

---

## Phase 0 — Environment setup (~30 min)

**Goal:** all three pieces (Mac venv, Reachy sim, hermes gateway) running and reachable.

1. **Mac venv:**
   ```bash
   cd /Users/wschenk/The-Focus-AI/hermes-body
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   ```
2. **Install Reachy Mini SDK + simulator:**
   ```bash
   pip install "reachy-mini[mujoco]"
   ```
3. **Smoke test the simulator:**
   ```bash
   mjpython -m reachy_mini.daemon.app.main --sim
   # 3D window should appear; dashboard at http://localhost:8000
   ```
4. **Install hermes-agent on Mac:**
   ```bash
   curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
   source ~/.zshrc
   hermes setup       # walk through; pick OpenRouter as provider, paste OPENROUTER_API_KEY
   ```
5. **Enable the API server:**
   ```bash
   cat >> ~/.hermes/.env <<'EOF'
   API_SERVER_ENABLED=true
   API_SERVER_KEY=hb-dev-$(openssl rand -hex 16)
   API_SERVER_HOST=0.0.0.0     # so the robot can reach it later
   API_SERVER_PORT=8642
   EOF
   ```
6. **Start gateway and verify:**
   ```bash
   hermes gateway
   # In another terminal:
   curl http://localhost:8642/v1/chat/completions \
     -H "Authorization: Bearer <key from .env>" \
     -H "Content-Type: application/json" \
     -d '{"model":"hermes-agent","messages":[{"role":"user","content":"hi"}]}'
   ```

**Done when:** sim window opens, hermes gateway prints `[API Server] listening on http://0.0.0.0:8642`, and the curl returns a 200 with a chat response.

---

## Phase 1 — Project skeleton (~30 min)

**Goal:** importable Python package with the Reachy Mini App entry point registered.

1. **`.gitignore`:**
   ```
   .venv/
   __pycache__/
   *.egg-info/
   .env
   ~/.hermes/images/
   ```
2. **`.env.example`:**
   ```bash
   # OpenAI Realtime (the brain)
   OPENAI_API_KEY=sk-...
   OPENAI_REALTIME_MODEL=gpt-realtime
   OPENAI_VOICE=cedar

   # Hermes (the knowledge tool)
   HERMES_BASE_URL=http://localhost:8642/v1
   HERMES_API_KEY=hb-dev-...
   HERMES_MODEL=hermes-agent

   # Robot/dev
   ROBOT_NAME=
   ENABLE_FACE_TRACKING=true
   HEAD_TRACKER_TYPE=mediapipe
   ```
3. **`pyproject.toml`:** the load-bearing config.
   ```toml
   [build-system]
   requires = ["setuptools>=61.0", "wheel"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "hermes-body"
   version = "0.1.0"
   description = "Reachy Mini front end for Nous Research's hermes-agent"
   requires-python = ">=3.11"
   dependencies = [
     "openai>=1.50.0",
     "fastrtc>=0.0.17",
     "numpy",
     "scipy",
     "python-dotenv",
     "httpx>=0.27",
     "websockets>=12.0",
     "opencv-python-headless",
     "gradio>=4.0",
     "mediapipe>=0.10.14",
   ]

   [project.scripts]
   hermes-body = "hermes_body.main:main"

   [project.entry-points."reachy_mini_apps"]
   hermes-body = "hermes_body.main:HermesBodyApp"

   [tool.setuptools.packages.find]
   where = ["src"]
   ```
4. **Stub modules:** create empty `src/hermes_body/{__init__,config,main,hermes_bridge,openai_realtime,prompts,moves,camera_worker,gradio_app}.py` plus `audio/__init__.py`, `vision/__init__.py`, `tools/__init__.py`, and a placeholder `HermesBodyApp` class in `main.py` so install succeeds.
5. **`config.py`:** load `.env`, expose typed config object, `validate()` returns a list of error strings.
6. **Install editable + smoke test:**
   ```bash
   pip install -e .
   python -c "from hermes_body.main import HermesBodyApp; print(HermesBodyApp)"
   ```

**Done when:** `pip install -e .` succeeds, `hermes-body --help` works, and a fresh `python -c "import hermes_body"` doesn't error.

---

## Phase 2 — HermesBridge (~1 hour)

**Goal:** working `ask()` / `get_agent_context()` / `sync_turn()` against the running gateway.

1. **`hermes_bridge.py`:** implement `HermesResponse` dataclass and `HermesBridge` class (see Pattern 3 in the report). Methods:
   - `ask(query, *, image_b64=None, system_context=None) -> HermesResponse`
   - `get_agent_context() -> str | None` (with the verbatim ClawBody prompt for the identity blob)
   - `sync_turn(user_msg, assistant_msg) -> None` (fire-and-forget OK)
2. **Image-attached `ask`:** when `image_b64` is provided, build the OpenAI vision content array (`[{"type":"text",...},{"type":"image_url",...}]`).
3. **Error handling:** distinguish `httpx.TimeoutException`, `HTTPStatusError`, generic exceptions. Always return a `HermesResponse` — never raise.
4. **Smoke script** (`tests/smoke_hermes.py`, throwaway):
   ```python
   import asyncio
   from hermes_body.hermes_bridge import HermesBridge

   async def main():
       h = HermesBridge()
       r = await h.ask("What's 2+2?")
       print("ask:", r)
       ctx = await h.get_agent_context()
       print("identity:", ctx[:200] if ctx else None)
       await h.sync_turn("hello", "hi there")
       print("sync ok")
   asyncio.run(main())
   ```

**Done when:** smoke script returns sensible answers from your running `hermes gateway`.

---

## Phase 3 — Lift movement / camera / vision from ClawBody (~2 hours)

**Goal:** all the local non-network code copy-pasted in, imports renamed, runnable in isolation.

1. **Clone ClawBody** (already at `/tmp/clawbody-src` from earlier research; re-clone if gone): `git clone https://github.com/tomrikert/clawbody.git /tmp/clawbody-src`.
2. **Lift verbatim with rename:**
   - `src/reachy_mini_openclaw/moves.py` → `src/hermes_body/moves.py`
   - `src/reachy_mini_openclaw/audio/head_wobbler.py` → `src/hermes_body/audio/head_wobbler.py`
   - `src/reachy_mini_openclaw/camera_worker.py` → `src/hermes_body/camera_worker.py`
   - `src/reachy_mini_openclaw/vision/mediapipe_tracker.py` → `src/hermes_body/vision/mediapipe_tracker.py`
   - `src/reachy_mini_openclaw/vision/yolo_head_tracker.py` → `src/hermes_body/vision/yolo_head_tracker.py` (optional — only if you want YOLO too)
3. **Sed the imports:** `sed -i '' 's/reachy_mini_openclaw/hermes_body/g' src/hermes_body/**/*.py src/hermes_body/*.py`.
4. **Drop config dependencies:** wherever lifted code does `from reachy_mini_openclaw.config import config`, replace with `from hermes_body.config import config`. Add the relevant fields to `config.py`.
5. **Sim test:** with `reachy-mini-daemon --sim` running, write a 30-line script that connects `ReachyMini()`, instantiates `MovementManager`, queues a `HeadLookMove(direction="left")`, and waits 2s. Watch the sim head turn.

**Done when:** the lifted modules import cleanly and a `MovementManager` test script makes the sim move.

---

## Phase 4 — Tools registry (~2 hours)

**Goal:** `TOOL_SPECS` and `dispatch_tool_call` lifted, plus `ask_hermes` tool spec.

1. **Lift `tools/core_tools.py` from ClawBody verbatim** (rename imports). Keep all handlers: `_handle_look`, `_handle_camera`, `_handle_face_tracking`, `_handle_dance`, `_handle_emotion`, `_handle_stop_moves`, `_handle_idle`.
2. **Strip OpenClaw-specific stuff:** the `_analyze_image_with_openai` helper stays (it uses OpenAI vision directly, no OpenClaw). The OpenClaw-fallback branch in `_handle_camera` goes — replace with "no description available, just return the b64 image" so the model can still see the pixels via Realtime.
3. **`ToolDependencies` dataclass:** drop the `openclaw_bridge` field, add `hermes_bridge: HermesBridge | None`.
4. **Add `ask_hermes` spec** (NOT in `core_tools.py` — keep it in `openai_realtime.py` next to other dynamic tool building, since whether it's offered depends on whether `hermes_bridge` is wired):
   ```python
   {
       "type": "function",
       "name": "ask_hermes",
       "description": (
           "Query Hermes (your full agent brain) for things you don't know off the top "
           "of your head: current weather, calendar, news, web search, smart home "
           "control, persistent memory, custom skills. Do NOT use this for chitchat or "
           "things you already know."
       ),
       "parameters": {
           "type": "object",
           "properties": {
               "query": {"type": "string"},
               "include_image": {"type": "boolean", "default": False},
           },
           "required": ["query"],
       },
   }
   ```
5. **Unit smoke:** dispatch each tool with mock deps; verify `dispatch_tool_call("look", '{"direction":"left"}', deps)` returns `{"status":"success","direction":"left"}`.

**Done when:** all robot tools dispatch correctly with a mocked `ToolDependencies`, and the `ask_hermes` spec is wired in `_build_tools()`.

---

## Phase 5 — OpenAIRealtimeHandler (~3 hours)

**Goal:** voice loop works end-to-end against the sim.

1. **Lift `openai_realtime.py` from ClawBody.** Rename imports. Keep:
   - `OPENAI_SAMPLE_RATE = 24000`
   - `ROBOT_BODY_INSTRUCTIONS` (move to `prompts.py`, lightly edit `ask_openclaw` → `ask_hermes`)
   - `FALLBACK_IDENTITY` (rewrite for Hermes-flavored generic identity)
   - `OpenAIRealtimeHandler` class — full lift
2. **Substitutions:**
   - `openclaw_bridge` field → `hermes_bridge` field (type `HermesBridge | None`)
   - `_handle_openclaw_query` → `_handle_hermes_query` (same shape — call `self.hermes.ask` instead of `self.openclaw_bridge.chat`)
   - `_sync_to_openclaw` → `_sync_to_hermes` (call `self.hermes.sync_turn`)
   - `_build_system_instructions` calls `self.hermes.get_agent_context()` instead of `self.openclaw_bridge.get_agent_context()`
   - In `_build_tools()`: append `ask_hermes` spec instead of `ask_openclaw`
   - In `_handle_tool_call()`: dispatch `ask_hermes` to `_handle_hermes_query`
3. **Reconnection loop:** keep ClawBody's exponential-backoff reconnect in `start_up()` verbatim — it handles WebSocket drops gracefully.
4. **Logging:** keep ClawBody's INFO-level `User: ...` and `Assistant: ...` logs — they're how you'll debug everything.

**Done when:** the handler instantiates, reads `OPENAI_API_KEY` from config, and `start_up()` logs `OpenAI Realtime session configured with N tools` (where N = robot tools + 1 for ask_hermes).

---

## Phase 6 — HermesBodyCore + main.py (~2 hours)

**Goal:** `hermes-body` CLI works against the sim end-to-end.

1. **`HermesBodyCore`:** orchestrator; lift from ClawBody's `ClawBodyCore`:
   - Constructor takes `robot=None`, `external_stop_event=None`
   - Wires up: `MovementManager`, `HeadWobbler`, `CameraWorker` (with optional `HeadTracker`), `HermesBridge`, `OpenAIRealtimeHandler`
   - `record_loop()` / `play_loop()` — verbatim from ClawBody
   - `run()` — enable motors, neutral pose, start movement/wobbler/camera/audio threads, gather tasks
   - `stop()` — graceful shutdown of all subsystems
2. **`HermesBodyApp`:** the `reachy_mini_apps` entry-point class. Trivial wrapper that creates a new event loop and calls `HermesBodyCore(robot=reachy_mini, external_stop_event=stop_event).run()`.
3. **`main()`:** argparse with the same flags as ClawBody (`--debug`, `--gradio`, `--robot-name`, `--no-camera`, `--no-hermes`, `--no-face-tracking`, `--head-tracker`).
4. **First end-to-end run:**
   ```bash
   # Terminal A: hermes gateway (already running from Phase 0)
   # Terminal B: simulator
   mjpython -m reachy_mini.daemon.app.main --sim
   # Terminal C: hermes-body
   hermes-body --debug
   ```
   Speak into your Mac mic. Say "hi". Hear a reply. Say "look left". Watch the sim turn.

**Done when:** chitchat works, `look` fires the local tool, `ask_hermes` ("what's the weather?") fires the Hermes round trip, and `response.done` triggers `sync_turn` (visible in `hermes gateway` logs as a new message).

---

## Phase 7 — Gradio UI (~1 hour)

**Goal:** `hermes-body --gradio` launches a browser UI at `localhost:7860` with mic + transcript.

1. **Lift `gradio_app.py` from ClawBody.** It uses FastRTC's `Stream` with the same `OpenAIRealtimeHandler`. Mostly a UI wrapper; rename imports.
2. **Launch from `main.py`** when `--gradio` is set: `from hermes_body.gradio_app import launch_gradio; launch_gradio(...)`.
3. **Test:** open `http://localhost:7860`, click mic, speak, see transcript stream.

**Done when:** browser-based voice loop works in addition to the CLI mic loop.

---

## Phase 8 — Sim validation (~2 hours)

**Goal:** confidence that everything works before touching the robot.

Run through this checklist with sim + hermes-body --debug + hermes gateway tail:

- [ ] **Greeting:** "Hi" → robot replies with personality from Hermes (identity blob is loaded). Verify the personality matches what `hermes` CLI shows for the same agent.
- [ ] **Look tool:** "Look to your left" → `_handle_look` fires, sim head turns left.
- [ ] **Dance tool:** "Do a dance" → `_handle_dance` fires, sim plays a move.
- [ ] **Emotion tool:** "Show me you're curious" → `_handle_emotion` fires, sim animates.
- [ ] **Camera tool:** "What do you see?" → `_handle_camera` fires; in sim, the camera returns the rendered scene; vision description comes back.
- [ ] **Face tracking toggle:** "Stop tracking faces" → `_handle_face_tracking({"enabled":false})` fires; verify the worker is paused.
- [ ] **ask_hermes (no image):** "What's the date?" or "Search the web for X" → `_handle_hermes_query` fires; Hermes processes; reply spoken.
- [ ] **ask_hermes (with image):** "Look at this and tell Hermes what you see" → `include_image=true` path; image arrives in Hermes logs.
- [ ] **Post-turn sync:** check `hermes gateway` logs after each turn — `[ROBOT BODY SYNC]` messages should appear.
- [ ] **Hermes down resilience:** stop `hermes gateway`, ask the robot something requiring Hermes; verify it speaks the tellable error instead of going silent.
- [ ] **Realtime reconnect:** kill your network for 5s, restore; verify Realtime reconnects (check the exponential-backoff log).

**Done when:** every box above is checked.

---

## Phase 9 — Robot deployment (~1 hour)

**Goal:** hermes-body runs on the actual Reachy Mini.

1. **Get hermes-agent reachable from the robot.** Find your Mac's LAN IP (`ipconfig getifaddr en0`); update `.env` with `HERMES_BASE_URL=http://<mac-lan-ip>:8642/v1`. Verify from the robot:
   ```bash
   ssh pollen@reachy-mini.local
   curl -s http://<mac-lan-ip>:8642/v1/models -H "Authorization: Bearer <key>" | head
   ```
   If that fails: macOS firewall is probably blocking port 8642. System Settings → Network → Firewall → allow Python.
2. **Push the code to the robot:**
   ```bash
   # Easiest: rsync from your dev machine
   rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
     /Users/wschenk/The-Focus-AI/hermes-body/ \
     pollen@reachy-mini.local:~/hermes-body/
   ```
   Or, if you've pushed to GitHub: `ssh pollen@reachy-mini.local 'git clone <url> hermes-body'`.
3. **Install into apps_venv:**
   ```bash
   ssh pollen@reachy-mini.local
   cd ~/hermes-body
   /venvs/apps_venv/bin/pip install -e .
   ```
4. **Copy `.env` to the robot:** `scp .env pollen@reachy-mini.local:~/hermes-body/.env`. Update `HERMES_BASE_URL` to the Mac IP.
5. **Verify dashboard discovery:** open `http://reachy-mini.local:8000`. The "hermes-body" app should appear in the list. Click Run.
6. **Run from CLI for first test** (easier to see logs):
   ```bash
   ssh pollen@reachy-mini.local
   /venvs/apps_venv/bin/hermes-body --debug
   ```
   Walk through the same Phase 8 checklist on real hardware.

**Done when:** the dashboard launches the app and the full Phase 8 checklist passes on the physical robot.

---

## Estimated total effort

| Phase | Estimate |
|---|---|
| 0. Environment setup | 30 min |
| 1. Project skeleton | 30 min |
| 2. HermesBridge | 1 hour |
| 3. Lift movement/camera/vision | 2 hours |
| 4. Tools registry | 2 hours |
| 5. OpenAIRealtimeHandler | 3 hours |
| 6. HermesBodyCore + main | 2 hours |
| 7. Gradio UI | 1 hour |
| 8. Sim validation | 2 hours |
| 9. Robot deployment | 1 hour |
| **Total** | **~15 hours** |

Realistically a long weekend.

## Open questions for later

- **YOLO vs MediaPipe face tracker:** v1 ships MediaPipe (lighter). Add YOLO option if accuracy matters.
- **Local vision (SmolVLM2):** ClawBody supports on-device vision. Skip for v1; revisit if Hermes-routed vision feels slow.
- **Gradio scene picker:** ClawBody has a `--scene minimal` flag for the sim. Skip for v1.
- **Tailscale deployment:** when you're ready to use the robot away from your Mac, do the tailnet setup.
- **Profile per-user:** Hermes supports `hermes -p <name>` profiles for isolated identities. Useful if multiple people want their own robot brain.
