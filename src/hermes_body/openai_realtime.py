"""OpenAI Realtime API handler embodying the Hermes agent.

Architecture:
    Startup: Fetch Hermes agent context (personality, memories, user info)
    Runtime: User speaks -> OpenAI Realtime (as Hermes agent) -> Robot speaks
             -> Tools for robot motion + ask_hermes for extended capabilities
             -> Conversations synced back to Hermes for memory continuity
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
from typing import Any, Final, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError

from hermes_body.audio.filler import FillerSpeaker
from hermes_body.config import config
from hermes_body.hermes_bridge import HermesBridge
from hermes_body.prompts import (
    FALLBACK_IDENTITY,
    ROBOT_BODY_INSTRUCTIONS,
    get_session_voice,
)
from hermes_body.tools.core_tools import (
    ToolDependencies,
    dispatch_tool_call,
    get_tool_specs,
)

# Tools that take long enough that we want to play "still working" filler audio.
SLOW_TOOLS = {"ask_hermes", "camera"}

logger = logging.getLogger(__name__)

# OpenAI Realtime API audio format
OPENAI_SAMPLE_RATE: Final[Literal[24000]] = 24000


# ask_hermes tool spec — appended dynamically when a HermesBridge is wired in.
ASK_HERMES_TOOL_SPEC = {
    "type": "function",
    "name": "ask_hermes",
    "description": (
        "Query Hermes (your full agent brain) for things you don't know off the "
        "top of your head: current weather, calendar, news, web search, smart "
        "home control, persistent memory, custom skills. Do NOT use this for "
        "chitchat or things you already know."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question or request to send to Hermes",
            },
            "include_image": {
                "type": "boolean",
                "description": (
                    "Whether to include current camera image (for 'what do you "
                    "see' style queries)"
                ),
                "default": False,
            },
        },
        "required": ["query"],
    },
}


class OpenAIRealtimeHandler(AsyncStreamHandler):
    """Realtime handler that embodies the Hermes agent."""

    def __init__(
        self,
        deps: ToolDependencies,
        hermes_bridge: Optional[HermesBridge] = None,
        gradio_mode: bool = False,
    ):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OPENAI_SAMPLE_RATE,
            input_sample_rate=OPENAI_SAMPLE_RATE,
        )

        self.deps = deps
        self.hermes = hermes_bridge
        self.gradio_mode = gradio_mode

        self.client: Optional[AsyncOpenAI] = None
        self.connection: Any = None

        self.output_queue: asyncio.Queue[
            Tuple[int, NDArray[np.int16]] | AdditionalOutputs
        ] = asyncio.Queue()

        self.last_activity_time = 0.0
        self.start_time = 0.0
        self._speaking = False

        self._agent_context: Optional[str] = None
        self._last_user_message: Optional[str] = None
        self._last_assistant_response: Optional[str] = None

        self._shutdown_requested = False
        self._connected_event = asyncio.Event()

        # Filler speech for slow tools — initialised in start_up() once the
        # OpenAI client exists.
        self._filler: Optional[FillerSpeaker] = None

    def copy(self) -> "OpenAIRealtimeHandler":
        return OpenAIRealtimeHandler(self.deps, self.hermes, self.gradio_mode)

    def _build_tools(self) -> list[dict]:
        """Build the tool list — robot motion specs + (optionally) ask_hermes."""
        tools = list(get_tool_specs())
        if self.hermes is not None:
            tools.append(ASK_HERMES_TOOL_SPEC)
        return tools

    async def start_up(self) -> None:
        """Connect to OpenAI Realtime with infinite reconnect loop."""
        api_key = config.OPENAI_API_KEY
        if not api_key:
            logger.error("OPENAI_API_KEY not configured")
            raise ValueError("OPENAI_API_KEY required")

        self.client = AsyncOpenAI(api_key=api_key)
        self.start_time = asyncio.get_event_loop().time()
        self.last_activity_time = self.start_time

        # Pre-generate filler audio for slow tools (one-time, ~2-3s).
        try:
            self._filler = FillerSpeaker(
                output_queue=self.output_queue,
                openai_client=self.client,
                voice=get_session_voice(),
            )
            await self._filler.generate()
        except Exception as e:
            logger.warning("Filler speech disabled: %s", e)
            self._filler = None

        attempt = 0
        max_backoff = 30

        while not self._shutdown_requested:
            attempt += 1
            try:
                await self._run_session()
                if self._shutdown_requested:
                    return
                attempt = 0
            except ConnectionClosedError as e:
                logger.warning("WebSocket closed unexpectedly (attempt %d): %s", attempt, e)
            except Exception as e:
                logger.error("Session error (attempt %d): %s", attempt, e)
            finally:
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

            if self._shutdown_requested:
                return

            delay = min(max_backoff, (2 ** min(attempt - 1, 5))) + random.uniform(0, 1)
            logger.info("Reconnecting in %.1f seconds...", delay)
            await asyncio.sleep(delay)

    async def _run_session(self) -> None:
        """Run a single OpenAI Realtime session."""
        model = config.OPENAI_REALTIME_MODEL
        logger.info("Connecting to OpenAI Realtime API with model: %s", model)

        system_instructions = await self._build_system_instructions()

        async with self.client.beta.realtime.connect(model=model) as conn:
            tools = self._build_tools()

            await conn.session.update(
                session={
                    "modalities": ["text", "audio"],
                    "instructions": system_instructions,
                    "voice": get_session_voice(),
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
                }
            )
            logger.info(
                "OpenAI Realtime session configured with %d tools", len(tools)
            )

            self.connection = conn
            self._connected_event.set()

            async for event in conn:
                await self._handle_event(event)

    async def _build_system_instructions(self) -> str:
        """Combine Hermes-fetched identity (or fallback) with robot capabilities."""
        agent_context = None
        if self.hermes is not None:
            logger.info("Fetching agent context from Hermes...")
            try:
                agent_context = await self.hermes.get_agent_context()
            except Exception as e:
                logger.warning("Failed to fetch Hermes context: %s", e)

        if agent_context:
            self._agent_context = agent_context
            logger.info("Using Hermes agent context (%d chars)", len(agent_context))
            return f"{agent_context}\n\n{ROBOT_BODY_INSTRUCTIONS}"

        logger.warning("Could not fetch Hermes context, using fallback identity")
        return f"{FALLBACK_IDENTITY}\n\n{ROBOT_BODY_INSTRUCTIONS}"

    async def _handle_event(self, event: Any) -> None:
        event_type = event.type

        if event_type == "input_audio_buffer.speech_started":
            self._speaking = False
            self.deps.movement_manager.set_processing(False)
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            self.deps.movement_manager.set_listening(True)
            logger.info("User started speaking")

        if event_type == "input_audio_buffer.speech_stopped":
            self.deps.movement_manager.set_listening(False)
            logger.info("User stopped speaking")

        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.transcript
            if transcript and transcript.strip():
                logger.info("User: %s", transcript)
                self._last_user_message = transcript
                await self.output_queue.put(
                    AdditionalOutputs({"role": "user", "content": transcript})
                )

        if event_type == "response.created":
            self._speaking = True
            logger.debug("Response started")

        if event_type == "response.audio.delta":
            self.deps.movement_manager.set_processing(False)
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.feed(event.delta)
            self.last_activity_time = asyncio.get_event_loop().time()
            audio_data = np.frombuffer(
                base64.b64decode(event.delta), dtype=np.int16
            ).reshape(1, -1)
            await self.output_queue.put((OPENAI_SAMPLE_RATE, audio_data))

        if event_type == "response.audio_transcript.done":
            response_text = event.transcript
            logger.info(
                "Assistant: %s",
                response_text[:100] if len(response_text) > 100 else response_text,
            )
            self._last_assistant_response = response_text
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": response_text})
            )

        if event_type == "response.done":
            self._speaking = False
            self.deps.movement_manager.set_processing(False)
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            logger.debug("Response completed")
            await self._sync_to_hermes()

        if event_type == "response.function_call_arguments.done":
            await self._handle_tool_call(event)

        if event_type == "error":
            err = getattr(event, "error", None)
            msg = getattr(err, "message", str(err))
            code = getattr(err, "code", "")
            logger.error("OpenAI error [%s]: %s", code, msg)

    async def _handle_tool_call(self, event: Any) -> None:
        tool_name = getattr(event, "name", None)
        args_json = getattr(event, "arguments", None)
        call_id = getattr(event, "call_id", None)

        if not isinstance(tool_name, str) or not isinstance(args_json, str):
            return

        logger.info(
            "Tool call: %s(%s)",
            tool_name,
            args_json[:50] if len(args_json) > 50 else args_json,
        )

        self.deps.movement_manager.set_processing(True)

        # Start "still working" filler audio for slow tools so the user
        # doesn't sit in silence while we wait on Hermes / vision.
        is_slow = tool_name in SLOW_TOOLS
        if is_slow and self._filler is not None:
            self._filler.start(label=tool_name)

        try:
            if tool_name == "ask_hermes":
                result = await self._handle_hermes_query(args_json)
            else:
                result = await dispatch_tool_call(tool_name, args_json, self.deps)
            logger.debug("Tool '%s' result: %s", tool_name, str(result)[:100])
        except Exception as e:
            logger.error("Tool '%s' failed: %s", tool_name, e)
            result = {"error": str(e)}
        finally:
            if is_slow and self._filler is not None:
                self._filler.stop()

        if isinstance(call_id, str) and self.connection:
            await self.connection.conversation.item.create(
                item={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                }
            )
            # Only request a new response if one isn't already streaming.
            # Otherwise OpenAI returns `conversation_already_has_active_response`
            # and the model gets confused. The in-flight response will pick
            # up the new tool output in its next turn.
            if not self._speaking:
                try:
                    await self.connection.response.create()
                except Exception as e:
                    logger.debug("response.create skipped: %s", e)

    async def _sync_to_hermes(self) -> None:
        """Push the last completed turn back to Hermes for memory continuity."""
        if not self.hermes or not self.hermes.is_available:
            return

        if self._last_user_message and self._last_assistant_response:
            try:
                await self.hermes.sync_turn(
                    self._last_user_message,
                    self._last_assistant_response,
                )
                self._last_user_message = None
                self._last_assistant_response = None
            except Exception as e:
                logger.debug("Failed to sync conversation: %s", e)

    async def _handle_hermes_query(self, args_json: str) -> dict:
        """Handle an `ask_hermes` tool invocation."""
        if self.hermes is None:
            return {
                "error": (
                    "Hermes bridge is not initialized. Tell the user you cannot "
                    "reach your backend right now and to try again later."
                )
            }

        try:
            args = json.loads(args_json)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON arguments"}

        query = args.get("query", "")
        include_image = args.get("include_image", False)

        image_b64: Optional[str] = None
        if include_image and self.deps.camera_worker:
            frame = self.deps.camera_worker.get_latest_frame()
            if frame is not None:
                try:
                    import cv2

                    _, buffer = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    image_b64 = base64.b64encode(buffer).decode("utf-8")
                    logger.debug("Captured camera image for Hermes query")
                except Exception as e:
                    logger.warning("Failed to encode camera image: %s", e)

        logger.info("Sending ask_hermes query: %s", query[:80])
        try:
            response = await self.hermes.ask(
                query,
                image_b64=image_b64,
                system_context=(
                    "User is asking through their Reachy Mini robot. "
                    "Keep response concise for voice."
                ),
            )
        except Exception as e:
            logger.error("Hermes query failed: %s", e)
            return {
                "error": (
                    f"Hermes query failed: {e}. Tell the user there was a "
                    "technical issue reaching your backend."
                )
            }

        if response.error:
            logger.warning("Hermes query error: %s", response.error)
            if "timeout" in response.error.lower():
                return {
                    "error": (
                        "The request to Hermes timed out — the backend is taking "
                        "too long. Tell the user you're having trouble reaching "
                        "your backend and to try again."
                    )
                }
            return {
                "error": (
                    f"Hermes returned an error: {response.error}. Tell the user "
                    "there was a problem processing their request."
                )
            }

        if not response.content:
            return {
                "error": (
                    "Hermes returned an empty response. Tell the user you got "
                    "no data back and to try again."
                )
            }

        return {"response": response.content}

    async def receive(self, frame: Tuple[int, NDArray]) -> None:
        if not self.connection:
            return

        input_sr, audio = frame

        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:
                audio = audio.T
            if audio.shape[1] > 1:
                audio = audio[:, 0]

        audio = audio.flatten()

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if input_sr != OPENAI_SAMPLE_RATE:
            num_samples = int(len(audio) * OPENAI_SAMPLE_RATE / input_sr)
            audio = resample(audio, num_samples).astype(np.float32)

        audio_int16 = (audio * 32767).astype(np.int16)

        try:
            audio_b64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_b64)
        except Exception as e:
            logger.debug("Failed to send audio: %s", e)

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        self._shutdown_requested = True

        if self.connection:
            try:
                await self.connection.close()
            except Exception as e:
                logger.debug("Connection close: %s", e)
            self.connection = None

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
