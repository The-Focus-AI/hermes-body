---
title: "Reachy Mini LLM Brains: Survey of Available Conversation/Agent Apps on Hugging Face"
date: 2026-05-04
topic: reachy-mini-llm-brains
recommendation: pollen-robotics/reachy_mini_conversation_app (reference baseline)
version_researched: v0.5.0 (April 20, 2026)
use_when:
  - You want a head-to-head comparison of LLM-driven controllers ("brains") for Reachy Mini that already exist in the HF ecosystem
  - You're deciding whether to fork an existing brain (cloud Realtime, local Llama, NeMo agent) or keep building Hermes from scratch
  - You're scoping which motion / tool primitives to expose to your LLM (move_head, dance, emotion, vision, head_tracking)
avoid_when:
  - You want VLA / direct motor-action policy models (Pi0, SmolVLA, GR00T) — that's a different survey
  - You want app reviews for non-LLM utilities (radio, metronome, hand-tracker)
project_context:
  language: Python
  relevant_dependencies:
    - reachy_mini SDK (Pollen Robotics)
    - openai (Realtime API) — already used by Hermes
    - fastrtc (low-latency audio streaming, used by reference apps)
    - pollen-robotics/reachy-mini-emotions-library (HF dataset)
    - pollen-robotics/reachy-mini-dances-library (HF dataset)
---

## Summary

There are **at least six distinct LLM-driven "brains" for Reachy Mini** publicly available, and they cluster into three architectural families: (1) **cloud Realtime brains** (OpenAI Realtime / Gemini Live) wrapped around the Pollen tool-dispatch pattern, (2) **fully-local voice agents** running Whisper + a small LLM + Piper/Kokoro TTS, and (3) **multi-model agentic stacks** (NVIDIA's NeMo Agent Toolkit + Pipecat + ReAct) that route between a reasoning LLM, a VLM, and a fast chit-chat model. Hermes-body fits in family 1 today (OpenAI Realtime handler, antenna/look tools).

The de-facto reference is **`pollen-robotics/reachy_mini_conversation_app`** (199 stars, Apache-2.0, v0.5.0 shipped 2026-04-20)[1]. It supports Hugging Face's free realtime backend, OpenAI Realtime (`gpt-realtime`), and Gemini Live (`gemini-3.1-flash-live-preview`), and exposes a tool surface — `move_head`, `camera`, `head_tracking`, `dance`, `stop_dance`, `play_emotion`, `stop_emotion`, `do_nothing` — that the wider community has copied[1]. Pulling in their layered motion system (queued primary moves blended with speech-reactive wobble + head-tracking) and the open emotion/dance datasets is probably the single highest-leverage thing Hermes could borrow[1][3].

The community has actively forked this baseline: **`gamepop/reachy-mini-gemini`** strips it down to two tools for Gemini Live[2]; **`dwain-barnes/reachy_mini_conversation_app_local`** swaps cloud APIs for Ollama/LM Studio + Distil-Whisper + Kokoro TTS targeting Jetson-class edge devices[4]; **Curtis Burkhalter's HP-ZGX-Nano agent** runs Llama 3.1 8B (INT4) + whisper.cpp + Piper for HIPAA-friendly local deployments[5]; **NVIDIA's `brevdev/reachy-personal-assistant`** uses NeMo Agent Toolkit + Pipecat to route between Nemotron 3 Nano 30B, Nemotron Nano 12B v2 VL, and Phi-3-Mini[6]; and **`Halfzipp/reachy_mini_megan`** is a community Space pitched as a "witty robot companion" with calendar, web search, and computer-use tools[7].

## Philosophy & Mental Model

Across all six brains, the same shape repeats — and it's worth internalizing because it's also where Hermes sits:

```
┌──────────────┐   audio    ┌─────────────────┐   tool calls   ┌──────────────┐
│ Mic / Camera │ ─────────► │  LLM / Realtime │ ─────────────► │  Reachy SDK  │
│  (Reachy)    │            │     backend     │                │ (motion+LED) │
│              │ ◄───────── │                 │ ◄───────────── │              │
└──────────────┘   speech   └─────────────────┘   sensor data  └──────────────┘
                                       ▲
                                       │ (optional VLM, web search, RAG, ...)
```

The key abstractions:

1. **Realtime audio loop** — most projects use `fastrtc` for low-latency bidirectional streaming so the LLM doesn't have to wait for VAD-bounded turns[1][2].
2. **Tool dispatch is async and non-blocking.** Pollen's design rule: "tool calls never block the audio stream"[1]. A `move_head` or `dance` tool returns immediately and motion is queued.
3. **Layered motion.** Primary moves (dances, emotions, poses, breathing) are queued by the tool dispatcher; speech-reactive wobble and head-tracking are blended on top[1]. This is what makes the robot feel alive while it talks instead of going still.
4. **Motion content is data, not code.** Pollen publishes emotions and dances as HF datasets (`pollen-robotics/reachy-mini-emotions-library`, `pollen-robotics/reachy-mini-dances-library`)[1] — every brain in the survey pulls from these instead of hand-rolling animations.
5. **Vision is optional and pluggable.** The Pollen reference app uses the realtime backend for vision by default but swaps to local SmolVLM2 with `--local-vision`[1]; NVIDIA dedicates a separate VLM (Nemotron 12B) routed to by a Phi-3 classifier[6].

If you're building a brain and you don't have these five concepts, you're going to hit one of the pitfalls below.

## Setup

Most brains follow the Pollen App CLI conventions (compatible with HF Spaces). To install the reference brain locally for inspection:

```bash
# Clone reference + install in editable mode
git clone https://github.com/pollen-robotics/reachy_mini_conversation_app
cd reachy_mini_conversation_app
uv sync
cp .env.example .env
```

Edit `.env` to pick a backend:

```bash
# .env — choose one
BACKEND_PROVIDER=huggingface          # default, no key needed
# BACKEND_PROVIDER=openai
# OPENAI_API_KEY=sk-...
# BACKEND_PROVIDER=gemini
# GEMINI_API_KEY=...
# Optional: run vision locally
# LOCAL_VISION_MODEL=HuggingFaceTB/SmolVLM2-2.2B-Instruct
```

Run it against a real Reachy Mini or simulator (the SDK exposes `--sim`):

```bash
# Terminal A: SDK daemon
uv run mjpython -m reachy_mini.daemon.app.main --sim --no-localhost-only

# Terminal B: the brain
uv run python -m reachy_mini_conversation_app
```

For NVIDIA's stack you also need a Pipecat service and a NeMo Agent Toolkit service running on ports 7860 and 8001, with three terminals total[6].

## Core Usage Patterns

### Pattern 1: Tool dispatch as the only motion API

Every brain in the survey converges on a small, declarative tool surface that the LLM calls. The Pollen baseline:

```python
# Conceptual — the reference app exposes these to the realtime backend
TOOLS = [
    "move_head",       # absolute or relative head pose
    "camera",          # capture a frame, return to VLM/realtime backend
    "head_tracking",   # toggle face/object tracking
    "dance",           # play named choreography from HF dataset
    "stop_dance",
    "play_emotion",    # play named emotion clip from HF dataset
    "stop_emotion",
    "do_nothing",      # explicit noop so the model can decline to act
]
```

The `do_nothing` tool is non-obvious but important: without it, function-calling models tend to *always* invoke a tool, leading to twitchy robots[1].

### Pattern 2: Layered motion blending

Don't have the LLM micromanage every joint. Queue a high-level primary move, and let a lower layer mix in ambient behavior:

```python
# Conceptual layering used by Pollen and copied by forks
motion_stack = [
    PrimaryMoveQueue(),          # dances, emotions, poses, breathing
    SpeechReactiveWobble(),      # antenna + head sway driven by TTS amplitude
    HeadTracking(),              # face follow, blended with primary
]
final_pose = blend(motion_stack, robot.current_state)
```

This is also what makes Hermes feel "expressive" with the antenna+look tools — you're already partway there but the dance/emotion library is a free upgrade.

### Pattern 3: Backend-agnostic handler

Pollen's app abstracts the realtime provider so you can swap OpenAI, Gemini, or HF without rewriting the tool layer:

```python
# Pseudocode of the handler abstraction
class RealtimeHandler(Protocol):
    async def stream(self, mic, on_text, on_tool_call): ...

handler = {
    "openai":     OpenAIRealtimeHandler(model="gpt-realtime"),
    "gemini":     GeminiLiveHandler(model="gemini-3.1-flash-live-preview"),
    "huggingface": HFRealtimeHandler(),
}[os.environ["BACKEND_PROVIDER"]]
```

This is the pattern Hermes-body should adopt before adding more providers — otherwise each new backend becomes a hard fork.

### Pattern 4: Local-only stack for privacy / offline

Curtis Burkhalter's brain shows the canonical local pipeline (HIPAA-grade). Three Docker services on a Grace Blackwell box, talked to over HTTP:

```
Reachy mic ──POST /process──► whisper.cpp (STT)
                              └► vLLM + Llama 3.1 8B AWQ INT4 (LLM)
                                  └► Piper (TTS)
Reachy speaker ◄──WAV──────────┘
```

End-to-end latency <2.5s on a single device, no internet[5]. Dwain Barnes's variant does the same with Ollama/LM Studio + Distil-Whisper + Kokoro and targets <3s on Jetson Nano (8GB+ RAM)[4].

### Pattern 5: Multi-model routing (NeMo Agent Toolkit)

NVIDIA's brain uses a tiny Phi-3 classifier as a router that picks the right model for the turn:

```yaml
# Excerpt from brevdev/reachy-personal-assistant config.yml
router:
  route_config:
    - name: chit_chat            # → fast text model
    - name: image_understanding  # → Nemotron 12B VL + camera frame
    - name: other                # → ReAct agent w/ wikipedia_search tool

functions:
  wikipedia_search:
    _type: wiki_search
    max_results: 2
  react_agent:
    _type: react_agent
    llm_name: agent_llm
    max_tool_calls: 4            # cap to prevent spiraling
    tool_names: [wikipedia_search]
```

The `max_tool_calls` cap is the key safety — without it, ReAct agents on a robot will spiral and the user just hears silence while motors twitch[6].

## Anti-Patterns & Pitfalls

### Don't: block the audio loop on a tool call

```python
# Bad — the realtime audio stops while the dance plays
async def on_tool_call(name, args):
    if name == "dance":
        await reachy.play_dance(args["name"])  # blocks for 8 seconds
    return {"status": "ok"}
```

**Why it's wrong:** the LLM stops streaming audio mid-sentence and the robot looks broken. Pollen explicitly designs for this: "tool calls never block the audio stream"[1].

### Instead: fire-and-forget into a motion queue

```python
async def on_tool_call(name, args):
    if name == "dance":
        motion_queue.put(("dance", args["name"]))   # returns immediately
    return {"status": "queued"}
```

### Don't: omit a `do_nothing` / `noop` tool

```python
TOOLS = ["move_head", "dance", "play_emotion"]   # no escape hatch
```

**Why it's wrong:** function-calling LLMs over-call tools when they have no graceful "decline" — every casual remark triggers a head-twitch[1].

### Instead: include an explicit noop

```python
TOOLS = ["move_head", "dance", "play_emotion", "do_nothing"]
```

### Don't: hard-code emotions and dances

```python
EMOTIONS = {"happy": [...lots of joint angles...], "sad": [...]}   # in code
```

**Why it's wrong:** you're rebuilding what the community already publishes, and you'll never match the polish of Pollen's recorded clips. Every surveyed brain pulls from `pollen-robotics/reachy-mini-emotions-library` and `pollen-robotics/reachy-mini-dances-library`[1][4].

### Instead: load HF datasets as motion content

```python
from datasets import load_dataset
emotions = load_dataset("pollen-robotics/reachy-mini-emotions-library")
dances   = load_dataset("pollen-robotics/reachy-mini-dances-library")
```

### Don't: let a ReAct agent loop unbounded on a robot

```yaml
react_agent:
  tool_names: [search_web, calendar, ...]
  # no max_tool_calls
```

**Why it's wrong:** the robot goes silent for 30+ seconds while the agent thrashes. NVIDIA's config explicitly caps tool calls[6].

### Instead: cap depth and surface progress

```yaml
react_agent:
  tool_names: [search_web, calendar, ...]
  max_tool_calls: 4
# and have the brain emit a verbal "let me check..." between tool calls
```

## Why This Choice

Treating `pollen-robotics/reachy_mini_conversation_app` as the reference baseline (rather than recommending a *replacement* brain) is the right call because (a) it's the most-starred and most-actively-maintained option, (b) every other community brain is a fork or rewrite of its tool surface, and (c) Hermes already overlaps with it architecturally.

### Decision Criteria

| Criterion | Weight | How the Pollen baseline scored |
|-----------|--------|--------------------------------|
| Provider flexibility | High | 3 backends (HF, OpenAI Realtime, Gemini Live) with handler abstraction[1] |
| Tool surface completeness | High | 8 tools covering motion, emotion, dance, vision, tracking, noop[1] |
| Motion library reuse | High | Pulls from canonical HF datasets (emotions, dances)[1] |
| Maintenance signal | High | 199 stars, v0.5.0 in April 2026, Apache-2.0[1] |
| Local/offline capable | Medium | Optional `--local-vision` only; full local needs a fork (dwain-barnes / curtburk)[4][5] |
| Multi-step reasoning | Medium | None native — for ReAct/agent loops, look at NVIDIA's NeMo brain[6] |

### Key Factors

- **Tool dispatch pattern is the lingua franca.** Every brain in the survey exposes some subset of `move_head / dance / play_emotion / camera / head_tracking / do_nothing`. Adopting that vocabulary makes Hermes interoperable with the ecosystem.
- **Motion content has already been crowdsourced.** Pollen's emotion + dance datasets are the highest-leverage borrow — Hermes's antenna/look tools are great but they don't yet tap the choreography library.
- **Realtime backends are commoditized.** OpenAI Realtime, Gemini Live, and HF realtime all expose roughly the same handler shape. Hermes can be backend-agnostic for free if it copies Pollen's handler protocol.
- **Local vs cloud is a config decision, not an architecture decision** — provided you keep the tool layer separate from the realtime backend layer.

## Alternatives Considered

### `gamepop/reachy-mini-gemini`

- **What it is:** Minimal Gemini Live brain with two tools (`move_head`, `express_emotion`) and wired/wireless support[2].
- **Why not chosen as primary reference:** very small surface (10 stars, 2 tools); no dance/vision/tracking.
- **Choose this instead when:**
  - You only care about Gemini Live and want the smallest possible reference implementation
  - You're doing wireless/over-the-air control experiments
- **Key tradeoff:** clarity over capability — much easier to read than the Pollen baseline, but you'll outgrow it in a week.

### `dwain-barnes/reachy_mini_conversation_app_local`

- **What it is:** Local-only fork of the Pollen app: Ollama or LM Studio + phi-3-mini-4k-instruct + Distil-Whisper + Kokoro-82M, targeting Jetson Nano (8GB+) with <3s end-to-end latency[4].
- **Why not chosen as primary reference:** narrower scope (single backend pattern, no cloud option) and only 10 stars.
- **Choose this instead when:**
  - You need fully offline operation on edge hardware
  - Privacy is non-negotiable (no audio/video leaves the device)
  - You want to demo without a network
- **Key tradeoff:** sacrifices model quality (phi-3-mini vs gpt-realtime) for full local control.

### Curtis Burkhalter's local voice agent (HF blog, March 2026)

- **What it is:** Whisper.cpp + Llama 3.1 8B INT4 (vLLM) + Piper TTS on an HP ZGX Nano (Grace Blackwell, 128GB unified memory), <2.5s e2e[5]. Designed for HIPAA-grade deployments.
- **Why not chosen as primary reference:** No tool calling — straight ASR→LLM→TTS pipeline. Read it for the deployment story, not the agent design.
- **Choose this instead when:**
  - You need air-gapped or HIPAA-grade deployment
  - You have access to GB10/DGX-class hardware
  - You don't need motion tool calling (just conversation + ambient motion)
- **Key tradeoff:** no agent loop / tools, but the most polished local infra story in the survey.

### NVIDIA `brevdev/reachy-personal-assistant`

- **What it is:** NeMo Agent Toolkit + Pipecat + Reachy SDK. Routes between Nemotron 3 Nano 30B (reasoning), Nemotron Nano 12B v2 VL (vision), and Phi-3-Mini (router) using a YAML-configured ReAct agent. Open source, not on HF Spaces[6].
- **Why not chosen as primary reference:** much heavier setup (3 terminals, 65GB+ VRAM, DGX Spark target), and the agent layer is overkill for a conversation app.
- **Choose this instead when:**
  - You need real multi-step reasoning (web search, calendar, tool chains)
  - You have DGX Spark or comparable hardware
  - You want the LangChain/LangGraph/CrewAI escape hatches
- **Key tradeoff:** agentic power vs. operational complexity — three services to keep alive instead of one.

### `Halfzipp/reachy_mini_megan` ("Megan")

- **What it is:** Community HF Space pitched as a witty companion that can answer questions, check calendar, search the web, and operate the user's computer[7].
- **Why not chosen as primary reference:** Space metadata is sparse — implementation details (which LLM, license) aren't visible without cloning. Treat as inspiration, not reference.
- **Choose this instead when:**
  - You want a richer "assistant" persona with productivity tools (calendar, web, computer-use)
  - You're studying how others package non-conversation tools
- **Key tradeoff:** broader tool surface, less documentation.

## Caveats & Limitations

- **Survey scope is LLM brains only.** VLA / direct policy models for Reachy Mini (Pi0, SmolVLA, GR00T N1.6 referenced in NVIDIA's blog[6]) are a different product category and would need their own report.
- **HF Spaces directory is gated.** `huggingface.co/spaces/pollen-robotics/Reachy_Mini_Apps` returned 401 to unauthenticated WebFetch — to get an exhaustive list, you'd need to log in or hit it from the Reachy Mini Control dashboard. Community apps from the search results may undercount.
- **No native Anthropic/Claude brain found.** As of May 2026 there's no public HF Space that uses Claude as the realtime backend. (Claude Code is referenced as a *development* tool via Skill builder, not a runtime brain.) If you want a Claude-driven Reachy brain, you'd be the first to publish it — natural fork of the Pollen baseline once Claude ships a Realtime-style API.
- **Most "brains" are forks, not greenfield.** The community has standardized on Pollen's tool surface; deviating from it (e.g., renaming `move_head` to something else) breaks compatibility with shared examples and tutorials.
- **Star counts are low across the board.** Outside the Pollen baseline (199), every brain is in single/double digits. The ecosystem is real but young — pick on architecture fit, not popularity.
- **Hermes is already in the cloud-Realtime family.** This survey doesn't tell you to rewrite Hermes. It tells you which capabilities (dances, emotions, head-tracking, vision swap, multi-backend handler) you can lift from neighbors.

## References

[1] [pollen-robotics/reachy_mini_conversation_app](https://github.com/pollen-robotics/reachy_mini_conversation_app) — Official reference brain. Apache-2.0, 199 stars, v0.5.0 (2026-04-20). Three realtime backends (HF, OpenAI Realtime `gpt-realtime`, Gemini Live `gemini-3.1-flash-live-preview`), 8-tool surface, layered motion system, optional local SmolVLM2.

[2] [gamepop/reachy-mini-gemini](https://github.com/gamepop/reachy-mini-gemini) — Minimal community Gemini Live brain. Apache-2.0, 10 stars. Two tools (`move_head`, `express_emotion`), six emotion presets, wired+wireless.

[3] [Reachy Mini Conversation App (HF Space)](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app) — HF Space wrapper for the Pollen reference brain.

[4] [dwain-barnes/reachy_mini_conversation_app_local](https://github.com/dwain-barnes/reachy_mini_conversation_app_local) — Fully local fork of the Pollen app. Apache-2.0, 10 stars. Ollama/LM Studio + phi-3-mini-4k-instruct + Distil-Whisper + Kokoro-82M, Jetson Nano target, <3s e2e.

[5] [Building a Fully Local Voice AI Agent on a Reachy Mini Robot](https://huggingface.co/blog/curtburk/reachy-voice-agent) — Curtis Burkhalter et al., March 2026. whisper.cpp + Llama 3.1 8B AWQ INT4 (vLLM) + Piper TTS on HP ZGX Nano (GB10, 128GB). <2.5s latency, HIPAA-friendly. No tool calling.

[6] [NVIDIA brings agents to life with DGX Spark and Reachy Mini](https://huggingface.co/blog/nvidia-reachy-mini) — NVIDIA + HF blog. NeMo Agent Toolkit + Pipecat orchestration; Nemotron 3 Nano 30B + Nemotron Nano 12B v2 VL + Phi-3-Mini router; ReAct agent w/ wiki search; ElevenLabs TTS. Source at [brevdev/reachy-personal-assistant](https://github.com/brevdev/reachy-personal-assistant).

[7] [Halfzipp/reachy_mini_megan (HF Space)](https://huggingface.co/spaces/Halfzipp/reachy_mini_megan) — Community "Megan" companion brain. Calendar, web search, computer-use tools per HF metadata. Implementation details not surfaced without auth.

[8] [Reachy Mini — Open-Source Robot for AI Builders (HF blog)](https://huggingface.co/blog/reachy-mini) — Background on the Reachy Mini platform, app store model, and HF Spaces integration.

[9] [Reachy Mini Integrations & Apps docs](https://huggingface.co/docs/reachy_mini/en/SDK/integration) — Official integration guide: App CLI, JS Web Apps, HTTP/WebSocket daemon API, AI experimentation tips.

[10] [pollen-robotics/reachy_mini SDK](https://github.com/pollen-robotics/reachy_mini) — Underlying robot SDK that every brain in this survey calls into.

[11] [Reachy Mini Apps hub (HF Space)](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini_Apps) — Central app discovery Space (auth-gated for full listing).

[12] [Build a Talking Robot with Gemini Live and Reachy Mini](https://dev.to/googleai/build-a-talking-robot-with-gemini-live-and-reachy-mini-20e2) — Google DEV article walking through the Gemini Live integration pattern.
