---
title: "Megan (Halfzipp/reachy_mini_megan): Deep Dive into a Classical-Pipeline Reachy Mini Brain"
date: 2026-05-04
topic: reachy-mini-megan-deep-dive
recommendation: Halfzipp/reachy_mini_megan v1.0.0 (changelog through 1.2.9, 2026-02-13)
version_researched: 1.0.0 (HF Space sha e1bbf1ef21ef4f58910c4523fa1f306f1a16642c)
use_when:
  - You want a worked reference for a wake-word + STT + text-LLM + TTS Reachy brain (the opposite design point from Hermes/Realtime brains)
  - You're designing the JSON shape that your text-LLM brain returns (text/emotion/gesture/memory) and want a battle-tested template
  - You're adding speaker recognition, DOA body-yaw tracking, scheduled behaviors, screen vision, or computer-use to a Reachy brain and want a prior-art read
  - You're trying to decide whether to use LLM tool-calling or hand-written regex `CommandHandler` dispatchers — Megan is the cautionary case study
avoid_when:
  - You want a Realtime/Live audio-loop brain (use `pollen-robotics/reachy_mini_conversation_app` instead)
  - You want a multi-provider / provider-abstracted brain (Megan is OpenAI-only)
  - You want something to fork and rebrand quickly — the persona, household, and pets are hardcoded throughout `prompts.py` and `command_dispatcher.py`
project_context:
  language: Python (>=3.10)
  relevant_dependencies:
    - reachy-mini >=1.2.7
    - openai >=1.12.0 (chat, TTS, STT, vision — all GPT-4o family)
    - openwakeword (local wake word — "Megan")
    - pyaudio + sounddevice + pygame (audio I/O, three libraries because Windows)
    - speechbrain (optional, voice recognition via spkrec-ecapa-voxceleb)
    - pyautogui + mss + Pillow (optional, computer use + screen vision)
    - mediapipe (face detection via blaze_face_short_range.tflite)
    - trafilatura + youtube-transcript-api (optional, podcast mode)
---

## Summary

**Megan** is a 1.0.0-tagged community Reachy Mini brain by "Brad" (HF user `Halfzipp`), published as a static HF Space at `Halfzipp/reachy_mini_megan` and registered as a Reachy Mini App via the `reachy_mini_apps` entry point[1]. It is the largest brain in the May-2026 Reachy ecosystem by an order of magnitude — **~8,900 lines of Python across 12 core modules**, vs. 8 tools and a single handler in the Pollen reference. It is also architecturally the most different: rather than a Realtime/Live audio-loop with LLM tool-calling, Megan is a classical **wake-word → STT → text LLM → TTS → motion** pipeline with a **2,992-line regex-based `CommandHandler` dispatcher in front of the LLM**[1][2].

The design optimizes for **persona depth and ambient behavior** — voice-recognized speakers get personalized sass levels, the head smoothly tracks faces at 100Hz between utterances, ReSpeaker DOA turns the body toward whoever is talking, and the LLM returns a single JSON object with `text + emotion + gesture + memories` per turn so memory extraction happens "for free" inline[3][4]. Provider story is **OpenAI-only**: GPT-4o-mini for chat, gpt-4o-mini-transcribe for STT, gpt-4o-mini-tts (voice "coral") for TTS, and GPT-4o for both screen and camera vision. Optional ElevenLabs for a second "Brittany" voice used in weather/news segments[5].

This report exists alongside the broader survey at `2026-05-04-reachy-mini-llm-brains.md`. Treat that one as the map; treat this one as the field guide for the most divergent point on the map.

## Philosophy & Mental Model

Megan's mental model is "**a small set of explicit, hand-coded behaviors wrapped around a single chat-shaped LLM call**" — the inverse of an agent loop. The LLM is consulted once per turn for a *response object*, not invoked iteratively as a tool-calling reasoner.

```
                                       ┌──────────────────────┐
   audio (mic, ReSpeaker DOA)          │  CommandHandlers     │
   ──────────────────────►  STT  ────► │  (regex/keyword)     │ ─► handled? ─► reply
                                       │  ~30 handler classes │
                                       └──────────┬───────────┘
                                                  │ no
                                                  ▼
                                       ┌──────────────────────┐
                                       │  ConversationManager │
   memories + speaker + weather + ───► │  GPT-4o-mini call    │ ─► JSON
   learning + history                  │  response_format=json│   {text,emotion,
                                       └──────────┬───────────┘    gesture,memories}
                                                  ▼
                                       ┌──────────────────────┐
                                       │  TTS + AnimationCtrl │
                                       │  + MovementManager   │
                                       │  (100Hz face track)  │
                                       └──────────────────────┘
```

Five concepts to internalize:

1. **The LLM returns a structured response object, not a stream.** `prompts.py` constrains the model to JSON `{text, emotion, gesture, memories[]}`. Memory extraction is in-band, not a second call (`conversation_manager.py:166-185`)[3].
2. **Commands run before the LLM.** ~30 `CommandHandler` subclasses pattern-match the user text first; only un-handled text reaches `ConversationManager.ask()`. This is fast and cheap but extremely brittle (the changelog flags "16 false-trigger-prone phrases" hardened in 1.2.8)[2][6].
3. **Persona is layered system messages.** Per turn, `ConversationManager.ask()` builds: base `SYSTEM_PROMPT` → speaker-specific attitude block → environmental context (weather, date) → learned patterns → long-term memories → conversation history → user message[3].
4. **Motion is a 100Hz background process.** `MovementManager` runs continuously during conversation, smoothly interpolating to whatever target (face, DOA angle, neutral) — gestures pause/resume it rather than fighting it. This was added in 1.2.9 explicitly to fix "head snaps to neutral between utterances"[6].
5. **Hardware is enumerated on startup, not assumed.** `_resolve_audio_devices()` lists every pyaudio + sounddevice device, substring-matches against env config, and audio-tests each candidate to skip silent ones (`main.py:225-292`). This is the single most copyable utility in the codebase.

## Setup

Megan is published as a static HF Space — the Space itself just hosts the source files. To install on the machine that will actually drive the robot:

```bash
# 1. Clone the Space (or pip-install once published to PyPI)
git clone https://huggingface.co/spaces/Halfzipp/reachy_mini_megan
cd reachy_mini_megan
pip install -e .              # or: pip install -e ".[all]" for every optional feature

# 2. Configure
cp .env.example .env
# minimum: OPENAI_API_KEY=sk-...
```

Selected env vars (full list in `.env.example`)[5]:

```bash
# Required
OPENAI_API_KEY=sk-...

# Model overrides (defaults shown)
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_TTS_MODEL=gpt-4o-mini-tts
OPENAI_STT_MODEL=gpt-4o-mini-transcribe
OPENAI_VISION_MODEL=gpt-4o
OPENAI_TTS_VOICE=coral

# Wake word
WAKE_WORD_THRESHOLD=0.5

# Optional features
SUPERMEMORY_API_KEY=...        # cloud semantic memory
OPENWEATHERMAP_API_KEY=...     # weather awareness
WEATHER_LOCATION=Sydney,AU
ELEVENLABS_API_KEY=...         # alt TTS provider
ELEVENLABS_REPORTER_VOICE_ID=  # second voice for weather/news ("Brittany")
NEWS_TOPICS=AI,robotics,technology
QUIET_HOURS_START=22
QUIET_HOURS_END=7

# Per-app audio device pinning (substring match — listed at startup)
AUDIO_INPUT_DEVICE=Razer Kraken
AUDIO_OUTPUT_DEVICE=Reachy Mini
```

Run via the entry point declared in `pyproject.toml`:

```bash
reachy-mini-megan
# or programmatically: ReachyMiniMegan().run(reachy, stop_event)
```

Megan declares itself as a Reachy Mini App via `[project.entry-points."reachy_mini_apps"]`, so the Reachy Mini Control dashboard discovers it automatically once installed[1].

## Core Usage Patterns

### Pattern 1: Single-call structured response (text + emotion + gesture + memories)

The single most copyable pattern. The system prompt forces JSON output and includes the memory extraction instruction in-band:

```python
# prompts.py:80-106 — the OUTPUT FORMAT block
"""
Always respond in JSON with:
{
  "text": "<your spoken response>",
  "emotion": "<one-word emotion>",
  "gesture": "<suggested gesture>",
  "memories": []
}
The "memories" field should contain things worth remembering long-term about the user.
Include memories ONLY when the user shares genuinely important info like:
- Personal facts (name, job, family, pets, location)
- Preferences (favorites, likes, dislikes)
...
Leave memories as an empty array [] for casual chat, greetings, or temporary states.
"""

# Valid emotions (Plutchik's wheel + custom):
VALID_EMOTIONS = ["unimpressed","joy","trust","fear","surprise","sadness",
                  "disgust","anger","anticipation","annoyed","confused","dramatic"]
VALID_GESTURES = ["head_tilt","tiny_flex","scan_room","shrug","listening",
                  "pre_speak","dance","spin"]
```

The dispatch:

```python
# conversation_manager.py:166-185 (condensed)
response = self.client.chat.completions.create(
    model=self.chat_model,                       # gpt-4o-mini
    messages=messages,                           # system + speaker + env + memory + history
    response_format={"type": "json_object"},
    timeout=API_TIMEOUT,
)
data = json.loads(response.choices[0].message.content)
data["text"] = self.clean_response_text(data["text"])     # strips emoji-hex garbage

# History stored without memories (keeps context window clean)
history_data = {k: v for k, v in data.items() if k != "memories"}
self.conversation_history.append({"role": "assistant", "content": json.dumps(history_data)})

self._process_inline_memories(data.get("memories", []))   # writes to local memory store
return data
```

Why this works: one round-trip per turn; memory extraction is amortized into the same call; `clean_response_text` defends against the GPT-4o-mini "trailing emoji hex" failure mode.

### Pattern 2: Layered system messages per turn

Every turn rebuilds the system context from live state — the conversation history is short (`max_history * 2`) and stateless context is re-injected each turn[3]:

```python
# conversation_manager.py:142-164 (condensed)
messages = [{"role": "system", "content": self.system_prompt}]

if speaker_context:                              # voice-identified speaker
    messages.append({"role": "system", "content": speaker_context})

if environmental_context:                        # weather + date + holidays
    messages.append({"role": "system",
                     "content": f"CURRENT ENVIRONMENT:\n{environmental_context}"})

if learning_context:                             # patterns/habits per speaker
    messages.append({"role": "system",
                     "content": f"LEARNED PATTERNS:\n{learning_context}"})

messages.append({"role": "system",
                 "content": f"LONG-TERM MEMORIES:\n{memory_context}"})
messages.extend(self.conversation_history)       # rolling window of recent turns
```

Speaker context is constructed from the voice-recognition system's per-speaker "attitude" record (sass level, style):

```python
attitude = self.voice_recognition.get_speaker_attitude(self.current_speaker)
speaker_context = f"""CURRENT SPEAKER: {self.current_speaker.capitalize()} (identified by voice, {self.current_speaker_confidence*100:.0f}% confidence)
INTERACTION STYLE FOR {self.current_speaker.upper()}: {attitude.get('style', 'default sass')}
SASS LEVEL: {attitude.get('sass_level', 0.7)}/1.0
Remember to address them by name occasionally and adjust your attitude accordingly."""
```

### Pattern 3: Pre-LLM regex command dispatcher

The dispatcher is the part you should **study and not copy**. It's an `ABC` with `can_handle(text) -> bool` and `handle(text) -> bool`, with one subclass per feature area: memory, dance, reminder, calendar, web search, screen vision, computer use, etc.

```python
# command_dispatcher.py:30-69 (condensed)
class CommandHandler(ABC):
    @abstractmethod
    def can_handle(self, text: str) -> bool: ...
    @abstractmethod
    def handle(self, text: str) -> bool: ...

class MemoryCommandHandler(CommandHandler):
    def can_handle(self, text: str) -> bool:
        t = text.lower()
        triggers = [
            "remember that", "remember this", "I work as", "My name is",
            "what do you remember", "what do you know about me",
            "forget everything", "reset memory", "start over",
            "consolidate" in t and "memor" in t,
            "organize"    in t and "memor" in t,
            "clean up"    in t and "memor" in t,
        ]
        return any(t in trigger if isinstance(trigger, str) else trigger
                   for trigger in triggers)
```

The total dispatcher file is **2,992 lines** across ~15 such classes. The 1.2.8 changelog notes "Command Trigger Phrase Hardening — fixed ~16 false-trigger-prone phrases across 6 command handlers"[6] — the maintenance cost is real and growing.

**Why Megan does it this way:** in a chat-first brain there's no realtime tool-call channel, so the only way to bind voice commands to deterministic Python (set a reminder, click a button on screen, play a recorded podcast) is to intercept text *before* it goes to the LLM. The right modern alternative is to use OpenAI's tool-calling on the same chat completion — see "Anti-Patterns" below.

### Pattern 4: Continuous 100Hz movement during the entire conversation

The 1.2.9 changelog frames this as *the* fix that made conversation feel natural[6]:

> Previously face tracking only worked during the ~3s speech window, causing head to snap to neutral between states. AnimationController breathing/thinking animations suppressed during conversation to avoid competing motor commands. Gestures wrapped with pause/resume so they play cleanly then face tracking resumes immediately.

Conceptually:

```python
# Pseudocode reconstructed from the changelog + main.py imports
movement_manager = MovementManager(reachy, rate_hz=100)
movement_manager.start()                         # owns the head 100% of the time

# During conversation:
self._conversation_owns_movement = True
animations.suppress_breathing_and_thinking()     # don't fight the 100Hz loop

# Per-frame target = blend(face_tracking, doa_yaw, current_gesture_pose)
# Gesture playback:
movement_manager.pause()
animations.perform_gesture("head_tilt")          # plays via goto_target
movement_manager.resume()                        # face tracking resumes immediately
```

Hermes will eventually need this exact pattern — the antenna/look tools currently move and stop, leaving the head static between LLM turns.

### Pattern 5: ReSpeaker DOA + camera face-tracking unified via `SpatialAwareness`

The 1.2.9 changelog also documents a non-obvious fact about the ReSpeaker mic array DOA: **it reports near-absolute angles, not relative to the body**. The naive "add DOA offset to current yaw" approach causes overshoot to ±90°[6]:

> DOA offset now used as **absolute world-frame body yaw target** (was incorrectly additive, causing overshoot to ±90°)
> First-speaker identification skip now only applies once per conversation
> Live DOA tracking during speech: if speaker moves >15° while talking, body yaw updates in real-time

A dedicated `spatial_awareness.py` module ties DOA + camera together with mode auto-detection (`combined`, `doa_only`, `camera_only`, `none`) and DOA-health monitoring with automatic mode downgrade if DOA gets stuck. Worth reading if Hermes ever adds a mic array.

## Anti-Patterns & Pitfalls

### Don't: route voice commands with a hand-written regex dispatcher

```python
# command_dispatcher.py — actual code shape, abridged
class MemoryCommandHandler(CommandHandler):
    def can_handle(self, text: str) -> bool:
        t = text.lower()
        triggers = ["remember that", "remember this", "I work as", "My name is",
                    "what do you remember", "what do you know about me",
                    "forget everything", "reset memory", "start over", ...]
        return any(trig in t for trig in triggers)
```

**Why it's wrong:** at ~30 handler classes and 2,992 lines, the dispatcher (a) has overlapping triggers that cause false positives (acknowledged in changelog 1.2.8: "16 false-trigger-prone phrases"[6]), (b) requires a code change for every new command, and (c) duplicates intent-classification work that the LLM already does well.

### Instead: put the same commands behind LLM tool-calling

```python
# Conceptual — use tools on the same chat completion
TOOLS = [
    {"type": "function", "function": {
        "name": "set_reminder",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string"},
            "when": {"type": "string", "description": "ISO timestamp or recurrence rule"},
        }, "required": ["task", "when"]}}},
    {"type": "function", "function": {
        "name": "remember_fact",
        "parameters": {"type": "object", "properties": {
            "fact": {"type": "string"},
        }, "required": ["fact"]}}},
    # ...etc, one per CommandHandler subclass
]

response = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, tools=TOOLS,
    response_format={"type":"json_object"},   # still get the {text,emotion,gesture}
)
# Dispatch tool_calls to the same MemoryCommandHandler.handle() bodies, no regex.
```

You keep all of Megan's handler bodies, you keep the JSON response shape, and you delete the regex `can_handle` layer entirely.

### Don't: leave breathing/thinking animations running while a movement manager is also driving the head

```python
# What every "naive" Reachy brain does
animations.start_breathing_loop()   # writes to head pose
movement_manager.start_face_track() # also writes to head pose
# → motors fight, head jitters
```

**Why it's wrong:** the 1.2.9 changelog explicitly fixes this: "AnimationController breathing/thinking animations suppressed during conversation to avoid competing motor commands"[6].

### Instead: one writer at a time, with explicit pause/resume

```python
movement_manager.pause()
animations.perform_gesture("head_tilt")   # one writer
movement_manager.resume()                 # back to face tracking
```

### Don't: treat ReSpeaker DOA as a relative offset

```python
# Naive — looks reasonable, breaks at ±90°
target_yaw = current_yaw + doa_offset
```

**Why it's wrong:** ReSpeaker DOA reports near-absolute angles in the room frame, not deltas from the body[6]. Adding it to current_yaw causes runaway overshoot.

### Instead: treat DOA as an absolute world-frame target

```python
target_yaw = doa_offset   # absolute world target, not additive
```

### Don't: assume the system audio defaults are right

Megan's `_resolve_audio_devices()` (`main.py:225-292`) lists every device, substring-matches the env config, and **audio-tests each candidate** to filter out silent ones. The reason: on Windows in particular, multiple devices match the same name, and only one of them actually captures audio.

```python
# Megan's pattern — literally test each candidate
for idx, name, ch in sd_candidates:
    with sd.InputStream(device=idx, channels=1, samplerate=16000, dtype="int16") as test:
        test_data, _ = test.read(8000)   # 0.5s
        peak = int(np.max(np.abs(test_data.squeeze())))
    if peak > 50:
        self._sd_input_device = idx
        break
```

This is worth lifting into Hermes verbatim — it's a 60-line utility that turns "it doesn't work on Windows" into a non-issue.

## Why This Choice

You wouldn't pick Megan as a *runtime* for Hermes — it's a parallel architecture, not an upgrade path. You pick Megan as a **reference for specific patterns** and as a worked example of how far a chat-first Reachy brain can go before its hand-written dispatcher becomes a liability.

### Decision Criteria

| Criterion | Weight | How Megan scored |
|-----------|--------|------------------|
| JSON response shape (text/emotion/gesture/memory) | High | ✅ Best-in-class — a copyable template |
| Persona depth & speaker awareness | High | ✅ Per-speaker sass levels, voice ID, attitude lookup |
| Continuous motion during conversation | High | ✅ 100Hz movement manager, gesture pause/resume |
| Hardware-pinned audio device handling | Medium | ✅ Substring match + audio-test (best in survey) |
| Provider flexibility | High | ❌ OpenAI-only, no abstraction layer |
| Realtime/Live audio | High | ❌ Classical wake-word + STT pipeline only |
| Maintainability of command surface | High | ❌ 2,992-line regex dispatcher, ~30 classes |
| Reusability outside one household | Medium | ❌ Persona, pets, partner names hardcoded |

### Key Factors

- **The single-LLM-call-with-inline-memories shape is the right primitive** for any chat-first brain. The cost is one round-trip; the value is structured `text + emotion + gesture` plus opportunistic memory writes.
- **The 100Hz `MovementManager` plus gesture pause/resume is a real research result.** It is the difference between "the robot looks alive while talking" and "the robot's head snaps around like a security camera."
- **The dispatcher is the cautionary tale.** It's what happens when you bolt features onto a chat brain without giving the LLM a tool-calling channel — eventually you have ~3,000 lines of trigger-phrase regex.

## Alternatives Considered

(Cross-reference: see the broader survey `2026-05-04-reachy-mini-llm-brains.md` for the full alternatives list. Below is just where Megan sits relative to its closest neighbors.)

### `pollen-robotics/reachy_mini_conversation_app` (the Realtime baseline)

- **Where Megan diverges:** Pollen uses Realtime/Live APIs with LLM tool-calling; Megan uses chat completions with regex dispatch.
- **Choose Pollen instead when:** you want low-latency voice, you want backend abstraction, you want to interoperate with the documented tool surface (`move_head`, `dance`, `play_emotion`, etc.).
- **Tradeoff:** Pollen has nothing comparable to Megan's persona depth, scheduled behaviors, calendar integration, podcast mode, screen vision, or computer use — those would all be *new* code on top of Pollen.

### `dwain-barnes/reachy_mini_conversation_app_local`

- **Where Megan diverges:** Dwain's brain is local (Ollama + Distil-Whisper + Kokoro); Megan is cloud (OpenAI everything).
- **Choose Dwain instead when:** privacy / offline / Jetson-class edge hardware.
- **Tradeoff:** loses GPT-4o-mini quality on chat *and* loses GPT-4o vision — Megan's screen/camera features would need to be rebuilt with a local VLM (e.g. SmolVLM2).

## Caveats & Limitations

- **OpenAI-only, top to bottom.** Chat (gpt-4o-mini), STT (gpt-4o-mini-transcribe), TTS (gpt-4o-mini-tts), vision (gpt-4o). No abstraction layer; swapping any one provider means surgery in `conversation_manager.py`, `main.py` (TTS), and `screen_vision.py` (vision).
- **No Realtime / Live API.** Latency is bounded below by `wake_word + STT + chat + TTS` round-trips. Acceptable for Megan's "wake word → reply" interaction model; insufficient for natural turn-taking like Hermes does.
- **Persona is hardcoded for Brad's household.** `prompts.py` references Brad, Hailey (sometimes spelled "Hayley"), and the cats Charley, Timmy, Jess by name. Cleaning this up before reuse is non-trivial because the persona is also assumed by many `CommandHandler` response strings.
- **The dispatcher does not scale.** 2,992 lines and growing; the changelog flags repeated false-trigger maintenance work. The right migration is to LLM tool-calling on the same chat completion.
- **Static HF Space — no live demo.** The Space hosts source files only; the runtime expects local hardware. You can't try Megan in a browser the way you can a Gradio Space.
- **Some declared HF model dependencies look vestigial.** `Salesforce/ctrl` is listed in the Space's `models[]` metadata but no code imports it; only `speechbrain/spkrec-ecapa-voxceleb` is actually used (for voice recognition). Don't read the metadata as a complete dependency list.
- **Windows-flavored audio plumbing.** Three audio libraries (pyaudio + sounddevice + pygame), WASAPI assumptions, per-device substring matching. The audio-resolve code is robust *because* the platform is brittle. Linux users will have a smoother time but should still expect to set `AUDIO_INPUT_DEVICE` / `AUDIO_OUTPUT_DEVICE`.
- **Computer-use is allow-listed but still real.** `pyautogui.FAILSAFE = True`, `MAX_ACTIONS_PER_SEQUENCE = 10`, action whitelist (`SAFE_ACTIONS`), key whitelist (`ALLOWED_KEYS`), explicit "do it" / "cancel" voice approval gate. Treat the safety surface as a *floor*, not a ceiling — if you fork this, audit it.

## Local Cache

The full Megan source tree fetched during this research is at:

```
/Users/wschenk/.cache/huggingface/hub/spaces--Halfzipp--reachy_mini_megan/snapshots/e1bbf1ef21ef4f58910c4523fa1f306f1a16642c/
```

Files of interest:

| File | Lines | Purpose |
|---|---:|---|
| `reachy_mini_megan/main.py` | 2015 | Entry point, audio device resolution, `ReachyMiniMegan(ReachyMiniApp)` |
| `reachy_mini_megan/command_dispatcher.py` | 2992 | The regex dispatcher (~30 `CommandHandler` subclasses) |
| `reachy_mini_megan/animation_controller.py` | 853 | Breathing, thinking, listening animations |
| `reachy_mini_megan/computer_use.py` | 614 | OCR-targeted clicking, allow-listed actions, failsafe |
| `reachy_mini_megan/voice_recognition.py` | 485 | SpeechBrain ECAPA enrollment, per-speaker attitude |
| `reachy_mini_megan/screen_vision.py` | 373 | GPT-4o screen analysis |
| `reachy_mini_megan/calendar_integration.py` | 364 | Google Calendar + meeting alerts |
| `reachy_mini_megan/conversation_manager.py` | 359 | The single-call JSON LLM pattern |
| `reachy_mini_megan/state.py` | 314 | Conversation/idle/sleep state machine |
| `reachy_mini_megan/config.py` | 227 | env loading + dataclass |
| `reachy_mini_megan/memory_manager.py` | 184 | Optional Supermemory.AI integration |
| `reachy_mini_megan/prompts.py` | 134 | The single system prompt + valid emotions/gestures |

`hf` CLI commands used to fetch them:

```bash
hf spaces info Halfzipp/reachy_mini_megan
hf download --repo-type space Halfzipp/reachy_mini_megan README.md pyproject.toml \
    .env.example CHANGELOG.md TODO.txt \
    reachy_mini_megan/main.py reachy_mini_megan/prompts.py \
    reachy_mini_megan/command_dispatcher.py reachy_mini_megan/conversation_manager.py \
    reachy_mini_megan/animation_controller.py reachy_mini_megan/memory_manager.py \
    reachy_mini_megan/computer_use.py reachy_mini_megan/screen_vision.py \
    reachy_mini_megan/voice_recognition.py reachy_mini_megan/state.py \
    reachy_mini_megan/config.py reachy_mini_megan/calendar_integration.py
```

## Borrows-for-Hermes Shortlist

Concrete things to lift, in rough priority order:

1. **The single-call JSON response shape** (`prompts.py:80-106` + `conversation_manager.py:166-185`). Copy the format, copy the in-band memory extraction, copy `clean_response_text()`. ~50 lines, biggest leverage.
2. **The 100Hz `MovementManager` with gesture pause/resume** (1.2.9 changelog pattern). Solves the "head freezes between LLM turns" problem before Hermes hits it.
3. **Audio-tested device resolution** (`main.py:225-292`). 60-line utility, makes Hermes work on contributors' machines without manual config.
4. **The DOA-as-absolute-world-yaw correction** if/when Hermes ever adds a ReSpeaker mic array. Already-paid-for debugging.
5. **The `SAFE_ACTIONS` + `ALLOWED_KEYS` + `MAX_ACTIONS_PER_SEQUENCE` + voice-approval pattern** if Hermes ever grows computer-use tools.
6. **The per-speaker "attitude" injection block** (`conversation_manager.py:122-130`) — easy upgrade once Hermes has any form of speaker recognition.

Things to **not** lift:

- The 2,992-line regex command dispatcher. Use OpenAI tool-calling on the chat completion instead.
- The hardcoded household persona. Read `prompts.py` for the structure, write your own contents.
- The vestigial `Salesforce/ctrl` model declaration. Don't cargo-cult the HF Space metadata.

## References

[1] [Halfzipp/reachy_mini_megan (HF Space)](https://huggingface.co/spaces/Halfzipp/reachy_mini_megan) — Static HF Space hosting the Megan source. Created 2026-01-12, last modified 2026-03-01. Tags: `reachy_mini`, `reachy-mini-app`, `reachy_mini_python_app`. Declares `pollen-robotics/reachy-mini-emotions-library` as a dataset dep.

[2] `command_dispatcher.py` (local cache, sha e1bbf1ef21ef…) — 2,992-line regex `CommandHandler` dispatcher with ~30 handler subclasses.

[3] `conversation_manager.py` (local cache) — `ConversationManager.ask()` shows the per-turn message construction pattern: system prompt + speaker context + environmental context + learning context + long-term memories + history → JSON-mode `chat.completions.create(model="gpt-4o-mini")`.

[4] `prompts.py` (local cache) — `SYSTEM_PROMPT`, valid emotion/gesture lists, JSON output contract.

[5] `.env.example` (local cache) — Full env var surface: OpenAI keys/models, Supermemory, OpenWeatherMap, ElevenLabs, news topics, quiet hours, audio device pinning.

[6] `CHANGELOG.md` (local cache) — Versions 1.2.8 (2026-02-10) and 1.2.9 (2026-02-13). Sources for the 100Hz MovementManager pattern, the DOA absolute-yaw correction, the gesture pause/resume design, the false-trigger-phrase hardening note, and the per-app audio device selection design.

[7] `pyproject.toml` (local cache) — Dependencies, optional-extras (`music`, `voice`, `cloud-memory`, `screen-vision`, `computer-use`, `elevenlabs`, `podcast`, `all`), entry points (`reachy-mini-megan` CLI; `reachy_mini_megan` registered in the `reachy_mini_apps` group).

[8] `README.md` (local cache) — Feature list, voice command catalog, recent-features highlights (spatial awareness, screen vision, calendar, podcast, web search, camera vision).

[9] `computer_use.py` (local cache) — `SAFE_ACTIONS`, `ALLOWED_KEYS`, `MAX_ACTIONS_PER_SEQUENCE = 10`, `pyautogui.FAILSAFE = True`. Allow-listed action surface for safe screen interaction.

[10] `main.py` (local cache, lines 225-292) — `_resolve_audio_devices()`: substring matching against pyaudio + sounddevice device lists, with live audio-test per candidate to skip silent devices.

[11] Companion report: `reports/2026-05-04-reachy-mini-llm-brains.md` — Survey of all six known LLM brains for Reachy Mini; Megan is one of six. This deep dive should be read alongside that survey.
