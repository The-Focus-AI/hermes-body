"""Filler-speech generator for slow tool calls.

When `ask_hermes` (or another slow tool) is in flight, the robot would
otherwise sit silently for 5-30 seconds — which feels broken. This module
pre-generates short "still working" phrases via OpenAI TTS at startup and
plays them periodically into the audio output queue while a slow tool is
running.

Usage:
    filler = FillerSpeaker(output_queue, openai_client, voice="cedar")
    await filler.generate()              # one-time at startup
    filler.start("ask_hermes")           # while tool is in flight
    ...
    filler.stop()                        # when tool returns
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Phrases the robot can say while a slow tool is in flight.
# Kept short so they don't compete with the eventual real answer.
DEFAULT_PHRASES = [
    "Still on it.",
    "Almost there.",
    "One moment.",
    "Hmm.",
    "Just a sec.",
    "Working on it.",
    "Almost done.",
]

# OpenAI TTS sample rate (matches Realtime).
SAMPLE_RATE = 24000


class FillerSpeaker:
    """Pre-generates and plays short filler clips while a slow tool runs."""

    def __init__(
        self,
        output_queue: asyncio.Queue,
        openai_client: AsyncOpenAI,
        voice: str = "cedar",
        first_filler_after: float = 6.0,
        repeat_every: float = 8.0,
        phrases: Optional[list[str]] = None,
    ):
        """Args:
        output_queue: the same queue the Realtime handler writes audio into.
        openai_client: AsyncOpenAI for TTS generation.
        voice: TTS voice (use the same voice as the realtime session).
        first_filler_after: seconds of silence before the first filler.
        repeat_every: seconds between subsequent fillers.
        phrases: override list of phrases to generate.
        """
        self.queue = output_queue
        self.client = openai_client
        self.voice = voice
        self.first_filler_after = first_filler_after
        self.repeat_every = repeat_every
        self.phrases = phrases or list(DEFAULT_PHRASES)

        self._clips: list[NDArray[np.int16]] = []
        self._task: Optional[asyncio.Task] = None
        self._generated = False
        self._lock = asyncio.Lock()

    async def generate(self) -> None:
        """Pre-generate all phrases. Idempotent. Safe to call early at startup."""
        async with self._lock:
            if self._generated:
                return
            logger.info(
                "Generating %d filler clips via OpenAI TTS (voice=%s)…",
                len(self.phrases),
                self.voice,
            )
            for phrase in self.phrases:
                try:
                    clip = await self._tts(phrase)
                    if clip is not None:
                        self._clips.append(clip)
                except Exception as e:
                    logger.warning("TTS failed for filler %r: %s", phrase, e)
            logger.info(
                "Filler ready: %d/%d clips generated", len(self._clips), len(self.phrases)
            )
            self._generated = True

    async def _tts(self, text: str) -> Optional[NDArray[np.int16]]:
        """Render `text` to int16 PCM at SAMPLE_RATE via OpenAI TTS."""
        # gpt-4o-mini-tts returns PCM directly when format=pcm; gpt-4o-mini-tts
        # outputs 24kHz, matching what Realtime expects.
        try:
            resp = await self.client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=self.voice,
                input=text,
                response_format="pcm",
            )
            raw: bytes = await resp.aread()
            return np.frombuffer(raw, dtype=np.int16).reshape(1, -1)
        except Exception as e:
            logger.debug("TTS error for %r: %s", text, e)
            return None

    def start(self, label: str = "tool") -> None:
        """Begin playing filler clips into the output queue periodically."""
        if self._task is not None and not self._task.done():
            return
        if not self._clips:
            logger.debug("No filler clips available — skipping (label=%s)", label)
            return
        self._task = asyncio.create_task(self._loop(label), name=f"filler-{label}")

    def stop(self) -> None:
        """Cancel any running playback loop. Safe to call multiple times."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _loop(self, label: str) -> None:
        try:
            await asyncio.sleep(self.first_filler_after)
            while True:
                clip = random.choice(self._clips)
                await self.queue.put((SAMPLE_RATE, clip))
                logger.debug("Played filler clip (label=%s)", label)
                await asyncio.sleep(self.repeat_every)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("Filler loop ended: %s", e)
