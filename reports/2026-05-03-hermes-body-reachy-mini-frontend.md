---
title: "hermes-body: A Reachy Mini Front End for Nous HermesAgent"
date: 2026-05-03
topic: hermes-body-reachy-mini-frontend
recommendation: "ClawBody-faithful Reachy Mini App: OpenAI Realtime is the embodied brain; hermes-agent is reached via a single ask_hermes tool"
version_researched: "reachy_mini 1.7.0 / ReachyMiniOS v0.2.3 / hermes-agent 0.12.x / clawbody 0.1.0"
use_when:
  - You want a voice-first embodied front end for Reachy Mini that feels snappy (sub-second tool dispatch)
  - You want hermes-agent to provide knowledge, memory, web search, smart home, etc. — but not be in the audio critical path
  - You're building on the proven ClawBody pattern and want minimal divergence from it
  - You want to ship the result as an installable Reachy Mini App (Hugging Face Space)
avoid_when:
  - You need the robot to literally be hermes-agent's embodied output (every turn through the full Hermes agent loop) — see Alternative 1
  - You need fully offline operation (the recommended path uses OpenAI Realtime for STT/TTS/dialogue)
  - On-device LLM inference at conversational latency on the Pi 5 (the hardware can't run Hermes-quality models in real time)
project_context:
  language: Python 3.11+
  relevant_dependencies: "(none yet — the repo is freshly initialized)"
---

## Summary

`hermes-body` should be a near-direct port of [ClawBody][3], with one substitution: replace the `ask_openclaw` tool (and its bespoke WebSocket bridge to OpenClaw) with an `ask_hermes` tool that calls hermes-agent's OpenAI-compatible API server[2]. Everything else — the OpenAI Realtime session as the embodied brain, robot motion tools registered directly on that session, local face tracking and head wobble threads, the `ReachyMiniApp` packaging, the camera worker — is lifted verbatim[3].

The critical fact that drives this design: **in ClawBody, OpenAI Realtime is the brain that decides what to say and when to call robot tools.** OpenClaw is consulted only when Realtime fires `ask_openclaw(query, include_image)` for things requiring external knowledge[3]. Realtime carries the robot's identity (fetched once at startup and stuffed into `instructions`) and routes tool calls locally for sub-second latency. Putting hermes-agent in the audio critical path would add 1–3 seconds per turn and break the embodied feel; using it as a tool keeps the snappy loop intact while giving the robot access to Hermes' tools, memory, skills, MCP, and the OpenClaw migration path (`hermes claw migrate`)[2][9].

The hardware target is verified live. The local Reachy Mini at `reachy-mini.local` runs **ReachyMiniOS v0.2.3** with `reachy_mini 1.7.0`, two pre-existing venvs (`/venvs/mini_daemon` and `/venvs/apps_venv`), and the `reachy-mini-daemon` systemd service exposing the dashboard on port 8000. ClawBody installs into `/venvs/apps_venv` and registers itself via the `reachy_mini_apps` entry point — the same install path applies to hermes-body[1][3].

## Philosophy & Mental Model

There are three boxes, and **OpenAI Realtime is the brain**:

```
┌──────────────────────┐    PCM16 audio (24 kHz)   ┌────────────────────────┐
│  Reachy Mini (Pi 5)  │ ────────────────────────▶ │   OpenAI Realtime API  │
│  reachy-mini.local   │ ◀──────────────────────── │   • ASR + TTS          │
│  apps_venv           │                           │   • dialogue + reasoning│
│  hermes-body app     │  function_call events:    │   • function-call dispatch│
│                      │  look / camera / dance /  │   • carries Hermes      │
│  ┌────────────────┐  │  emotion / face_tracking /│     identity in `instructions`│
│  │ Realtime client│◀─│  ask_hermes               │                        │
│  │ MovementMgr    │  │                           └──────────┬─────────────┘
│  │ CameraWorker   │  │                                      │ ask_hermes(query)
│  │ FaceTracker    │  │                                      ▼
│  │ HeadWobbler    │  │   only when Realtime    ┌────────────────────────┐
│  │ HermesBridge   │──┼────────────────────────▶│   hermes-agent gateway │
│  └────────────────┘  │   decides it needs      │   :8642 (laptop / VPS) │
└──────────────────────┘   external knowledge    │   POST /v1/chat/...    │
                                                 └────────────────────────┘
```

The mental model — read carefully, this is the part the original report got wrong:

- **OpenAI Realtime owns the *whole* conversation, not just transport.** It does VAD, ASR, TTS, *and* generation. It also decides when to fire a robot tool (`look`, `dance`, `emotion`, `camera`, `face_tracking`) and when to ask Hermes for knowledge. This is why the loop feels embodied — tool calls execute on the same WebSocket as the audio stream, with no extra hop[3].
- **Hermes is a tool, not the brain.** Realtime calls `ask_hermes(query, include_image=False)` only when it needs something Realtime can't do on its own: web search, calendar, smart home, persistent cross-session memory, custom skills, MCP integrations[2][9]. Most utterances never touch Hermes.
- **Hermes carries the agent's identity.** At startup, hermes-body fetches an identity blob from Hermes ("who are you, what do you remember about this user, what's your personality") and stuffs it into Realtime's `instructions` field. The robot then *speaks as* the Hermes agent. After each turn, the user message + assistant response are synced back to Hermes so memory stays consistent across the robot, CLI, Telegram, etc[3][9].
- **Reachy Mini owns the body loop.** Movement, head wobble, face tracking, audio I/O — all run on the robot in their own threads, never blocked by network calls[3][6].

The single most important property of this design: **a typical chitchat utterance touches only the robot ↔ Realtime loop.** Network round trips to Hermes happen only when the model decides the question warrants them. That's why the robot feels alive instead of laggy.

## Setup

### 1. Bootstrap the Python package locally

In `/Users/wschenk/The-Focus-AI/hermes-body` (currently empty except for `.git`/`README.md`):

```bash
cd /Users/wschenk/The-Focus-AI/hermes-body
python3.11 -m venv .venv
source .venv/bin/activate

# Reachy Mini SDK + simulator (so you can dev without ssh-ing every time)
pip install "reachy-mini[mujoco]"

# OpenAI Realtime + transport
pip install "openai>=1.50.0" "fastrtc>=0.0.17" "python-dotenv" "numpy" "scipy"

# HTTP client for hermes-agent
pip install httpx

# Optional: face tracking parity with ClawBody
pip install mediapipe        # lighter
# OR: pip install ultralytics supervision opencv-python   # more accurate
```

### 2. Create the `pyproject.toml` with the App entry point

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "hermes-body"
version = "0.1.0"
description = "Reachy Mini front end with Nous Research's hermes-agent as the knowledge tool"
requires-python = ">=3.11"
dependencies = [
  "openai>=1.50.0",
  "fastrtc>=0.0.17",
  "numpy",
  "scipy",
  "python-dotenv",
  "httpx",
  "websockets>=12.0",
]

[project.scripts]
hermes-body = "hermes_body.main:main"

[project.entry-points."reachy_mini_apps"]
hermes-body = "hermes_body.main:HermesBodyApp"

[tool.setuptools.packages.find]
where = ["src"]
```

Source layout (mirrors ClawBody's `src/reachy_mini_openclaw/` almost exactly):

```
hermes-body/
├── pyproject.toml
├── README.md
└── src/hermes_body/
    ├── __init__.py
    ├── main.py             # HermesBodyApp + HermesBodyCore (port of clawbody main.py)
    ├── hermes_bridge.py    # ask_hermes target — replaces openclaw_bridge.py
    ├── openai_realtime.py  # Realtime handler — port of clawbody, change `ask_openclaw` → `ask_hermes`
    ├── moves.py            # MovementManager (lift verbatim)
    ├── camera_worker.py    # CameraWorker (lift verbatim)
    ├── audio/head_wobbler.py
    └── tools/core_tools.py # local robot tools dispatch (lift verbatim)
```

### 3. Stand up hermes-agent with the API server enabled

On a laptop or VPS the robot can reach over LAN[2][9]:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
hermes setup     # walk through model + token config

cat >> ~/.hermes/.env <<'EOF'
API_SERVER_ENABLED=true
API_SERVER_KEY=hermes-body-dev-key-change-me
API_SERVER_HOST=0.0.0.0          # so the robot on the LAN can reach it
API_SERVER_PORT=8642
EOF

hermes gateway   # prints: [API Server] API server listening on http://0.0.0.0:8642
```

Smoke test from your dev box:

```bash
curl http://<host>:8642/v1/chat/completions \
  -H "Authorization: Bearer hermes-body-dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"model":"hermes-agent","messages":[{"role":"user","content":"Hello!"}]}'
```

### 4. Configure hermes-body

`.env` next to `pyproject.toml`:

```bash
OPENAI_API_KEY=sk-...                              # for Realtime API (brain)
OPENAI_REALTIME_MODEL=gpt-realtime                 # or gpt-4o-realtime-preview
OPENAI_VOICE=cedar

HERMES_BASE_URL=http://<host>:8642/v1
HERMES_API_KEY=hermes-body-dev-key-change-me
HERMES_MODEL=hermes-agent

ENABLE_FACE_TRACKING=true
HEAD_TRACKER_TYPE=mediapipe
```

### 5. Install on the robot

```bash
ssh pollen@reachy-mini.local
git clone <your-repo> hermes-body && cd hermes-body
/venvs/apps_venv/bin/pip install -e .
# It now appears in the dashboard at http://reachy-mini.local:8000
```

## Core Usage Patterns

### Pattern 1: The dual entry point (CLI + Reachy Mini App)

ClawBody ships two entry points from the same module: a `main()` CLI for `hermes-body` and a `HermesBodyApp` class the dashboard launches[3][1]. Mirror this exactly:

```python
# src/hermes_body/main.py
import os, asyncio, threading
from typing import Optional
from reachy_mini import ReachyMini

class HermesBodyCore:
    def __init__(self, robot: Optional[ReachyMini] = None,
                 external_stop_event: Optional[threading.Event] = None):
        self._owns_robot = robot is None
        self.robot = robot or ReachyMini()
        self._external_stop_event = external_stop_event
        # ... wire up MovementManager, HeadWobbler, CameraWorker,
        #     HermesBridge, OpenAIRealtimeHandler

    async def run(self):
        self.robot.enable_motors()
        # ... start movement, audio, network loops

class HermesBodyApp:
    """Reachy Mini dashboard entry point (auto-discovered via entry_points)."""
    custom_app_url: Optional[str] = None

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app = HermesBodyCore(robot=reachy_mini, external_stop_event=stop_event)
        try:
            loop.run_until_complete(app.run())
        finally:
            app.stop(); loop.close()

def main():
    asyncio.run(HermesBodyCore().run())
```

When the dashboard launches you, `reachy_mini` is already initialized and you must respect `stop_event`[1]. When `main()` runs, you own the lifecycle (including `media.close()` and `client.disconnect()` on shutdown).

### Pattern 2: Robot motion tools live on the Realtime session, not on Hermes

This is the line that the original draft of this report got backwards. ClawBody registers all robot tools — `look`, `camera`, `face_tracking`, `dance`, `emotion`, `stop_moves`, `idle` — directly with the OpenAI Realtime session via `session.update({"tools": [...]})`[3]. Realtime fires them mid-utterance and a local `dispatch_tool_call` runs them on the robot. Keep this exactly the same:

```python
# src/hermes_body/openai_realtime.py — port of clawbody, only the ask_* tool changes
async def _run_session(self):
    tools = self._build_tools()  # robot tools + ask_hermes
    async with self.client.beta.realtime.connect(model=config.OPENAI_REALTIME_MODEL) as conn:
        await conn.session.update(session={
            "modalities": ["text", "audio"],
            "instructions": await self._build_system_instructions(),  # identity from Hermes
            "voice": config.OPENAI_VOICE,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper-1"},
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 600,
            },
            "tools": tools,
            "tool_choice": "auto",
        })
        self.connection = conn
        async for event in conn:
            await self._handle_event(event)

async def _handle_event(self, event):
    # ... audio in/out handling ...
    if event.type == "response.function_call_arguments.done":
        await self._handle_tool_call(event)

async def _handle_tool_call(self, event):
    name, args_json, call_id = event.name, event.arguments, event.call_id
    self.deps.movement_manager.set_processing(True)
    try:
        if name == "ask_hermes":
            result = await self._handle_hermes_query(args_json)
        else:
            # robot motion tools — local dispatch, sub-100ms
            result = await dispatch_tool_call(name, args_json, self.deps)
    except Exception as e:
        result = {"error": str(e)}
    await self.connection.conversation.item.create(item={
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(result),
    })
    await self.connection.response.create()
```

The `tools/core_tools.py` file (`get_tool_specs()` + `dispatch_tool_call()` + `_handle_look`, `_handle_camera`, `_handle_dance`, `_handle_emotion`, `_handle_face_tracking`, `_handle_stop_moves`, `_handle_idle`) lifts verbatim from ClawBody[3]. None of those tools needs Hermes — they're local hardware actions.

### Pattern 3: The `ask_hermes` tool — single point of contact with hermes-agent

Replace ClawBody's `ask_openclaw` tool spec and handler. The shape mirrors ClawBody's `ask_openclaw`[3] but the bridge is a 50-line OpenAI client instead of a 600-line WebSocket protocol:

```python
# Tool spec added to _build_tools() alongside the robot motion specs:
{
    "type": "function",
    "name": "ask_hermes",
    "description": (
        "Query Hermes (the agent's full brain) for things requiring external tools, "
        "knowledge, or persistent memory. Use this for: weather, calendar, web "
        "searches, news, smart home control, accessing detailed memories, scheduled "
        "jobs, custom skills, or any task that needs Hermes' tools. Do NOT use this "
        "for chitchat or things you already know."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query":         {"type": "string",  "description": "What to ask Hermes"},
            "include_image": {"type": "boolean", "description": "Attach the current camera frame", "default": False},
        },
        "required": ["query"],
    },
}
```

```python
# src/hermes_body/hermes_bridge.py
import os, base64, httpx
from dataclasses import dataclass, field

@dataclass
class HermesResponse:
    content: str
    error: str | None = None

@dataclass
class HermesBridge:
    base_url: str = field(default_factory=lambda: os.environ["HERMES_BASE_URL"])
    api_key:  str = field(default_factory=lambda: os.environ["HERMES_API_KEY"])
    model:    str = field(default_factory=lambda: os.getenv("HERMES_MODEL", "hermes-agent"))

    async def ask(self, query: str, *, image_b64: str | None = None,
                  system_context: str | None = None) -> HermesResponse:
        content = (
            [{"type": "text", "text": query},
             {"type": "image_url",
              "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "low"}}]
            if image_b64 else query
        )
        messages = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        messages.append({"role": "user", "content": content})
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": messages, "stream": False},
                )
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                return HermesResponse(content=text)
        except httpx.TimeoutException:
            return HermesResponse(content="", error="Hermes timed out")
        except httpx.HTTPStatusError as e:
            return HermesResponse(content="", error=f"Hermes HTTP {e.response.status_code}")
        except Exception as e:
            return HermesResponse(content="", error=str(e))

    async def get_agent_context(self) -> str | None:
        """Fetch identity/memory/personality to embed in Realtime's instructions."""
        resp = await self.ask(
            query="Provide your current context summary for the robot body.",
            system_context=(
                "You are being asked to provide your current context for your robot body. "
                "Output a comprehensive context summary another AI can use to embody you. "
                "Include: 1) IDENTITY (who you are, name, personality, speaking style), "
                "2) USER CONTEXT (what you know about the user), "
                "3) RECENT CONTEXT (ongoing topics), "
                "4) MEMORIES (key things relevant to interactions), "
                "5) CURRENT STATE (time/date awareness, ongoing tasks). "
                "Be specific and personal. Output ONLY the context summary, no preamble."
            ),
        )
        return resp.content if not resp.error else None

    async def sync_turn(self, user_msg: str, assistant_msg: str) -> None:
        """After each turn, push the exchange back to Hermes for memory continuity."""
        await self.ask(
            query=(f"[ROBOT BODY SYNC] The following happened through the Reachy Mini robot:\n"
                   f"User said: {user_msg}\n"
                   f"You responded: {assistant_msg}\n"
                   f"Remember this as part of your ongoing conversation."),
            system_context=("[ROBOT BODY SYNC] This conversation happened through your Reachy Mini "
                            "robot body. Remember it as part of your ongoing conversation."),
        )
```

The handler in the Realtime class:

```python
async def _handle_hermes_query(self, args_json: str) -> dict:
    args = json.loads(args_json)
    query = args.get("query", "")
    include_image = args.get("include_image", False)

    image_b64 = None
    if include_image and self.deps.camera_worker is not None:
        frame = self.deps.camera_worker.get_latest_frame()
        if frame is not None:
            import cv2
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_b64 = base64.b64encode(buf).decode()

    resp = await self.hermes.ask(
        query, image_b64=image_b64,
        system_context="User is asking through their Reachy Mini robot. Keep response concise for voice.",
    )
    if resp.error:
        return {"error": f"Hermes error: {resp.error}. Tell the user the backend is unreachable."}
    return {"response": resp.content}
```

Note `include_image=True` uses hermes-agent's inline image content blocks, which Hermes routes to a vision-capable model automatically (or to `vision_analyze` if the configured model is text-only)[2][11].

### Pattern 4: Identity bootstrap — Hermes lends its personality to Realtime

ClawBody fetches OpenClaw's context at session start and concatenates it with the static `ROBOT_BODY_INSTRUCTIONS` block to build the Realtime `instructions` field[3]. Do the same with Hermes:

```python
async def _build_system_instructions(self) -> str:
    agent_context = None
    if self.hermes is not None:
        agent_context = await self.hermes.get_agent_context()
    if not agent_context:
        agent_context = FALLBACK_IDENTITY  # generic Hermes-flavored identity blurb
    return f"{agent_context}\n\n{ROBOT_BODY_INSTRUCTIONS}"
```

`ROBOT_BODY_INSTRUCTIONS` is the same one ClawBody uses, lightly edited to mention `ask_hermes` instead of `ask_openclaw`[3]:

```text
## Your Robot Body (Reachy Mini)
You are currently embodied in a Reachy Mini robot. You have physical capabilities:

**Movement Tools (use naturally during conversation):**
- look — Move head to look left/right/up/down/front
- emotion — Express happy/sad/surprised/curious/thinking/confused/excited
- dance — Dance to celebrate or express joy
- camera — Capture what you see through your camera
- face_tracking — Toggle face tracking on/off
- stop_moves / idle — Cancel current motion / return to neutral

**How to Use Your Body:**
- Look around while thinking or to emphasize points
- Express emotions that match what you're saying
- Dance when celebrating good news
- Use the camera when asked "what do you see?"
- Reference your body naturally ("let me look", "I can see…")

**Conversation Style for Voice:**
- Keep responses concise — you're speaking, not typing
- Use natural speech patterns ("hmm", "well", "let me see")

**Extended Capabilities (via ask_hermes tool):**
For things needing Hermes' full agent capabilities, use ask_hermes:
- Calendar, weather, news lookups
- Web searches, browsing
- Smart home / Home Assistant
- Persistent cross-session memory and skills
- Any task needing external tools or up-to-date info
```

### Pattern 5: Sync each turn back to Hermes for memory continuity

ClawBody pushes `(user_msg, assistant_msg)` to OpenClaw after every turn so the same conversation can continue from Telegram, the CLI, or another channel[3]. Hermes is built around this exact pattern (cross-platform conversation continuity is a headline feature)[9]:

```python
# In _handle_event(), on response.done:
if event.type == "response.done":
    self._speaking = False
    await self._sync_to_hermes()

async def _sync_to_hermes(self):
    if self._last_user_message and self._last_assistant_response and self.hermes:
        await self.hermes.sync_turn(self._last_user_message, self._last_assistant_response)
        self._last_user_message = None
        self._last_assistant_response = None
```

The downside: each turn fires a background Hermes request. That's fine — it's not in the critical path because it runs after `response.done`. If you want to be extra safe, fire-and-forget with `asyncio.create_task(self._sync_to_hermes())` so a slow Hermes never delays the next utterance.

### Pattern 6: Continuous behaviors stay local (face tracking, head wobble)

Lift `CameraWorker` + `HeadTracker` (MediaPipe or YOLO) verbatim from ClawBody[3]. They run in their own threads at ~25 Hz and update `MovementManager` directly. They never touch Realtime, never touch Hermes. The only Realtime tool that interacts with them is `face_tracking` (on/off), which dispatches locally:

```python
# Inside HermesBodyCore.__init__ — same as ClawBody
self.head_tracker = HeadTracker(device="cpu")
self.camera_worker = CameraWorker(reachy_mini=self.robot, head_tracker=self.head_tracker)
self.camera_worker.set_head_tracking_enabled(True)
self.movement_manager.camera_worker = self.camera_worker  # closes the perception→motion loop
```

Same for `HeadWobbler`: it consumes outgoing audio amplitude from Realtime's `response.audio.delta` events and nudges the head to make speech feel embodied[3]. Pure local thread, zero network.

### Pattern 7: Audio loops are pure ClawBody (Reachy mic → Realtime → speakers)

```python
# Inside HermesBodyCore.run() — verbatim ClawBody pattern
self.robot.media.start_recording()
self.robot.media.start_playing()

self._tasks = [
    asyncio.create_task(self.handler.start_up(),  name="realtime-handler"),
    asyncio.create_task(self.record_loop(),       name="record-loop"),
    asyncio.create_task(self.play_loop(),         name="play-loop"),
]

async def record_loop(self):
    sr = self.robot.media.get_input_audio_samplerate()
    while not self._should_stop():
        frame = self.robot.media.get_audio_sample()
        if frame is not None:
            await self.handler.receive((sr, frame))
        await asyncio.sleep(0.01)

async def play_loop(self):
    out_sr = self.robot.media.get_output_audio_samplerate()
    while not self._should_stop():
        out = await self.handler.emit()
        if isinstance(out, tuple):
            in_sr, audio = out
            audio = audio.flatten().astype("float32") / 32768.0 * 0.5
            if in_sr != out_sr:
                from scipy.signal import resample
                audio = resample(audio, int(len(audio) * out_sr / in_sr)).astype("float32")
            self.robot.media.push_audio_sample(audio)
        await asyncio.sleep(0.01)
```

### Pattern 8: Simulator-first dev loop

You don't want to deploy to the Pi every time you change a prompt. Reachy Mini ships a MuJoCo simulator that is API-identical to the real robot[3][5]:

```bash
# Terminal 1 (your laptop):
reachy-mini-daemon --sim
# On macOS use: mjpython -m reachy_mini.daemon.app.main --sim

# Terminal 2:
hermes-body            # CLI entry — talks to the sim
```

The sim listens on `http://localhost:8000` and accepts `ReachyMini()` connections without code changes. Develop here; deploy to `apps_venv` only when the loop works.

## Anti-Patterns & Pitfalls

### Don't: Put hermes-agent in the audio critical path

```python
# bad — every utterance round-trips to Hermes; latency 1–3s per turn
async def on_user_transcript(text):
    response = await self.hermes.ask(text)
    await self.realtime.say(response)
```

**Why it's wrong:** This is the architecture I incorrectly recommended in the first draft. It shifts the brain from Realtime to Hermes, which means even "hi" pays the full agent-loop tax — provider selection, tool consideration, memory I/O. The robot stops feeling embodied and starts feeling like a Discord bot with a body[3].

### Instead: Let Realtime drive, ask_hermes only when warranted

```python
# good — Realtime decides, fires ask_hermes when knowledge is actually needed
"tools": [look_spec, camera_spec, dance_spec, emotion_spec, ask_hermes_spec, ...]
"tool_choice": "auto"
```

Realtime answers chitchat itself with sub-second latency; Hermes is consulted only on demand[3].

### Don't: Re-implement OpenClaw's WebSocket protocol against Hermes

```python
# bad — invents a protocol Hermes doesn't speak
ws = await websockets.connect("ws://hermes-host:8642")
await ws.send(json.dumps({"type":"req","method":"chat.send", "params": {...}}))
```

**Why it's wrong:** Hermes' API server is plain OpenAI HTTP (`/v1/chat/completions`, `/v1/responses`)[2]. The OpenClaw protocol-3 framing in ClawBody's `openclaw_bridge.py` exists only because OpenClaw didn't expose an OpenAI-compatible surface — that constraint is gone.

### Instead: A 50-line `httpx` bridge

See `HermesBridge` in Pattern 3.

### Don't: Block the asyncio loop on `reachy_mini.goto_target`

```python
# bad
self.robot.goto_target(head=pose, duration=2.0)  # blocks 2 seconds!
```

**Why it's wrong:** `goto_target` is synchronous; it returns when motion finishes[6]. Block the event loop and your audio loops stutter.

### Instead: Run movement on its own thread

ClawBody wraps everything in a `MovementManager` thread that pulls intents off a queue[3]. Async code just enqueues:

```python
self.movement_manager.queue_move(HeadLookMove(direction="left", ...))
```

Or use the SDK's `async_play_move` for predefined moves.

### Don't: Install hermes-body into `mini_daemon`'s venv

```bash
# bad
sudo /venvs/mini_daemon/bin/pip install -e hermes-body
```

**Why it's wrong:** `mini_daemon` runs the dashboard service. Adding app deps can break it on next boot. The robot ships **two** venvs for exactly this reason[1].

### Instead: Use `apps_venv`

```bash
/venvs/apps_venv/bin/pip install -e .
```

Same path ClawBody uses, same path the dashboard launches Apps from[1][3].

### Don't: Forget to handle Hermes being down

```python
# bad — a 30s Hermes timeout blocks the conversation
result = await self.hermes.ask(query)
return {"response": result.content}
```

**Why it's wrong:** When Hermes is unreachable, Realtime is still waiting on the tool result and the user hears silence. ClawBody's `_handle_openclaw_query` returns a structured error with a script for what Realtime should say[3]:

### Instead: Return a tellable-error string

```python
if resp.error:
    return {
        "error": f"Hermes error: {resp.error}. "
                 "Tell the user your backend is unreachable and to try again."
    }
```

Realtime will read that to the user and keep the conversation alive.

## Why This Choice

### Decision Criteria

| Criterion | Weight | How hermes-body (this design) scores |
|-----------|--------|--------------------------------------|
| Voice latency on chitchat | High | Sub-second — Realtime answers without round-tripping to Hermes |
| Time-to-first-conversation | High | ~1 day — almost everything is borrowed from ClawBody; only the bridge changes |
| Tool-call snappiness | High | Local dispatch on the Realtime session; ~10 ms motion tools |
| Reuses standard interfaces | High | Realtime tool calls (standard) + plain OpenAI HTTP to Hermes |
| Decoupled deploy story | High | Hermes runs anywhere on the LAN; robot only needs network access |
| Tool / memory / skill richness | Medium-High | Available via `ask_hermes`; not invoked on every turn |
| Migration path from ClawBody/OpenClaw | High | `hermes claw migrate` exists; replace one tool name + bridge file[9] |

### Key Factors

- **ClawBody already proved the architecture.** Realtime-as-brain with a single `ask_X` tool to a backend agent is the pattern that gives you both snappy voice *and* deep knowledge[3]. Don't second-guess it.
- **Hermes' OpenAI-compatible API is the unlock.** Without it, the bridge would still be a custom WebSocket protocol. With it, the bridge is ~50 lines[2].
- **Memory continuity comes free.** Hermes is built for cross-platform conversation continuity[9]; the `sync_turn` push pattern means the same agent identity works through the robot, CLI, Telegram, etc.
- **The robot's verified state confirms feasibility:** `reachy_mini 1.7.0` in `apps_venv`, daemon + dashboard running, ports 8000/8443/7860 in use, ClawBody-style apps already run on this exact device.

## Alternatives Considered

### Alternative 1: hermes-agent as the brain (every turn through the full Hermes loop)

- **What it is:** Skip Realtime as the dialogue brain. Use Realtime only for ASR + TTS. Every transcribed utterance goes to `hermes-agent /v1/chat/completions`; the response goes to TTS.
- **Why not chosen:** The first draft of this report recommended exactly this and it was wrong. Even chitchat pays the full Hermes agent tax — provider selection, tool consideration, memory I/O — and you're looking at 1–3 seconds per turn instead of sub-second. Robots that pause for two seconds before saying "hi" feel broken.
- **Choose this instead when:** You're building something where every turn legitimately needs the agent loop (e.g., a robot that does coding work out loud) and latency isn't the primary UX concern.
- **Key tradeoff:** Deeper agent integration ↔ broken embodied feel.

### Alternative 2: Run Hermes locally on the robot

- **What it is:** Install hermes-agent inside `apps_venv` and call `AIAgent.chat()` in-process[7].
- **Why not chosen:** The Pi 5-class CPU can't run Hermes-quality models in real time. Even pointed at a local Ollama with a small model, the agent loop will starve audio threads.
- **Choose this instead when:** You have on-device acceleration and need fully offline operation.
- **Key tradeoff:** Privacy + offline ↔ latency + voice quality.

### Alternative 3: Skip Hermes entirely — Realtime + function calls

- **What it is:** Use OpenAI Realtime for everything, define robot tools as Realtime functions, no Hermes at all.
- **Why not chosen:** You lose the whole Hermes agent layer — persistent memory across sessions, skills, MCP, Telegram/Discord parity, OpenClaw migration[2][9]. The point of the project is to give *Hermes* a body.
- **Choose this instead when:** You're building a generic voice toy and don't care about cross-session memory.
- **Key tradeoff:** Simpler ↔ no persistent agent identity.

### Alternative 4: Fork ClawBody and swap the bridge in-place

- **What it is:** Clone `tomrikert/clawbody`, gut `openclaw_bridge.py`, point it at Hermes.
- **Why not chosen:** You inherit Clawson's name and personality, the `agent:<id>:<session>` keying scheme, and OpenClaw-specific bridge assumptions[3]. Cleaner to keep ClawBody as a *reference* and write hermes-body fresh, lifting the genuinely transferable pieces (movement, head wobble, audio loops, face tracker, App scaffolding, Realtime handler shell).
- **Choose this instead when:** You want to ship today and don't mind some Clawson DNA.
- **Key tradeoff:** Faster start ↔ harder to evolve cleanly.

## Caveats & Limitations

- **OpenAI Realtime is a hard dependency for the recommended path.** The whole sub-second feel depends on it. Lose internet, lose voice. There is no graceful local fallback unless you also wire up local Whisper + Edge TTS (out of scope for v1).
- **Realtime does the dialogue reasoning.** That means model choice for the voice brain is whatever OpenAI offers (`gpt-realtime`, `gpt-4o-realtime-preview`). You only get to swap the *Hermes* model — Realtime's model is fixed by the API.
- **`ask_hermes` is the only Hermes touchpoint.** If Realtime decides not to call it, Hermes never sees the conversation (until the post-turn `sync_turn`). Tune the tool description and `ROBOT_BODY_INSTRUCTIONS` carefully so Realtime knows when to defer to Hermes.
- **`API_SERVER_KEY` exposes Hermes' full toolset including a terminal**[2]. Treat the bearer like an SSH key. Never bind to `0.0.0.0` over the public internet — keep Hermes on a LAN or behind WireGuard.
- **The `model` field in the Hermes request is cosmetic.** The actual model is server-side[2]. Don't try to switch per-turn.
- **Conversation history is stateless on `/v1/chat/completions`.** Each `ask_hermes` call ships only the user's question — Hermes' own memory layer is the continuity story. If you want richer multi-turn within Hermes for follow-up `ask_hermes` calls, switch to `/v1/responses` with `previous_response_id`[2].
- **No file upload through the API** (inline images are fine on both endpoints)[2]. Camera frames go via `image_url` data URLs.
- **Reachy Mini's `mini_daemon` venv is sacred.** Install only into `/venvs/apps_venv`.
- **`goto_target` blocks.** Always run motion on its own thread (the ClawBody `MovementManager` pattern), or use `async_play_move`[6].
- **Native Windows isn't supported for hermes-agent** — install host must be Linux, macOS, or WSL2[9].

## References

[1] [Make and publish your Reachy Mini App](https://huggingface.co/blog/pollen-robotics/make-and-publish-your-reachy-mini-apps) — Reachy Mini App lifecycle, `ReachyMiniApp.run(reachy_mini, stop_event)` contract, file layout, dashboard install model.
[2] [Hermes Agent — API Server docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server) — OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/responses`), default port `8642`, bearer auth, SSE format, image input support.
[3] [tomrikert/clawbody on GitHub](https://github.com/tomrikert/clawbody) — Reference implementation. Verified by reading `openai_realtime.py`: Realtime owns dialogue + tool dispatch; OpenClaw is consulted only via the `ask_openclaw` tool. Source for: dual entry point, MovementManager, HeadWobbler, CameraWorker, identity-bootstrap pattern, post-turn sync, ROBOT_BODY_INSTRUCTIONS template, `tools/core_tools.py` dispatch.
[4] [pollen-robotics/reachy_mini on GitHub](https://github.com/pollen-robotics/reachy_mini) — Official SDK repo, current `v1.7.0`, dependencies (`fastapi`, `uvicorn`, `huggingface-hub`, `reachy-mini-rust-kinematics`).
[5] [Reachy Mini desktop app (simulator pathway)](https://github.com/pollen-robotics/reachy-mini-desktop-app) — Local dashboard / simulator integration.
[6] [reachy_mini Python SDK reference (verified live on reachy-mini.local)](https://pollen-robotics.github.io/reachy_mini/) — `goto_target(head, antennas, duration, method, body_yaw)` signature confirmed; full `ReachyMini` public API (`look_at_world`, `play_move`, `media`, `imu`, etc.).
[7] [Hermes Agent — Python library guide](https://hermes-agent.nousresearch.com/docs/guides/python-library) — `AIAgent` programmatic API; alternative path if you ever want to in-process Hermes (Alternative 2).
[8] [reachy_mini SDK quickstart](https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/SDK/quickstart.md) — Install + `ReachyMini` context-manager usage.
[9] [NousResearch/hermes-agent on GitHub](https://github.com/NousResearch/hermes-agent) — Project README: install one-liner, `hermes gateway` entry, OpenClaw migration (`hermes claw migrate`), provider list, cross-platform conversation continuity.
[10] [Hermes Agent — Voice Mode docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/voice-mode) — Confirms Hermes' voice mode is chunk-based (record → STT → agent → TTS), not realtime streaming. Justifies keeping OpenAI Realtime as the voice transport.
[11] [Hermes Agent — Vision & Image Paste docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/vision) — Inline `image_url` content blocks, automatic routing to vision-capable models vs text-only models via the `vision_analyze` auxiliary tool. Justifies the `include_image=True` path in `ask_hermes`.
[12] [Hermes Agent — MCP integration docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) — Stdio + HTTP MCP servers via `mcp_servers:` config. Not used in v1 (kept as a future option) but worth knowing exists.
