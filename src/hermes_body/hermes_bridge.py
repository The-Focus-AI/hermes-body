"""Bridge to a hermes-agent gateway via its OpenAI-compatible chat API.

Three operations:

- `ask(query, ...)` — single round-trip; returns a HermesResponse (never raises).
- `get_agent_context()` — fetch identity / memory / personality summary used to
  bootstrap the Realtime session's system instructions.
- `sync_turn(user_msg, assistant_msg)` — fire-and-forget post a turn back to
  Hermes so cross-channel memory stays in sync.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

from hermes_body.config import config

logger = logging.getLogger(__name__)


@dataclass
class HermesResponse:
    """Response from the Hermes gateway. `error` is None on success."""

    content: str
    error: Optional[str] = None


@dataclass
class HermesBridge:
    """Client for hermes-agent's OpenAI-compatible chat API."""

    base_url: str = field(default_factory=lambda: config.HERMES_BASE_URL)
    api_key: str = field(default_factory=lambda: config.HERMES_API_KEY)
    model: str = field(default_factory=lambda: config.HERMES_MODEL)
    timeout: float = 120.0

    async def ask(
        self,
        query: str,
        *,
        image_b64: Optional[str] = None,
        system_context: Optional[str] = None,
    ) -> HermesResponse:
        """Send a single chat message to Hermes and return its reply.

        Args:
            query: The user-visible text part of the message.
            image_b64: Optional base64-encoded JPEG to attach as vision input.
            system_context: Optional system message prepended to the request.

        Returns:
            HermesResponse — never raises; populates `error` on failure.
        """
        if image_b64:
            content = [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "low",
                    },
                },
            ]
        else:
            content = query

        messages: list[dict] = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        messages.append({"role": "user", "content": content})

        body = {"model": self.model, "messages": messages, "stream": False}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return HermesResponse(content=text or "")

        except httpx.TimeoutException:
            logger.warning("Hermes request timed out")
            return HermesResponse(content="", error="Hermes timed out")
        except httpx.HTTPStatusError as e:
            logger.warning("Hermes HTTP error: %s", e.response.status_code)
            return HermesResponse(
                content="", error=f"Hermes HTTP {e.response.status_code}"
            )
        except Exception as e:
            logger.warning("Hermes request failed: %s", e)
            return HermesResponse(content="", error=str(e))

    async def get_agent_context(self) -> Optional[str]:
        """Fetch identity/memory/personality summary for the Realtime session.

        Returns the raw text from Hermes, or None if the call failed or the
        response was empty.
        """
        resp = await self.ask(
            query="Provide your current context summary for the robot body.",
            system_context=(
                "You are being asked to provide your current context for your robot body. "
                "Output a comprehensive context summary that another AI can use to embody you. "
                "Include: "
                "1. YOUR IDENTITY: Who you are, your name, your personality traits, how you speak. "
                "2. USER CONTEXT: What you know about the user (name, preferences, relationship). "
                "3. RECENT CONTEXT: Summary of recent conversations or important ongoing topics. "
                "4. MEMORIES: Key things you remember that are relevant to interactions. "
                "5. CURRENT STATE: Any relevant time/date awareness, ongoing tasks. "
                "Be specific and personal. This context will be used by your robot body to speak and act AS YOU. "
                "Output ONLY the context summary, no preamble."
            ),
        )

        if resp.error:
            logger.warning("Failed to fetch Hermes agent context: %s", resp.error)
            return None
        if not resp.content.strip():
            logger.warning("Hermes returned empty agent context")
            return None

        logger.info("Fetched Hermes agent context (%d chars)", len(resp.content))
        return resp.content

    async def sync_turn(self, user_message: str, assistant_message: str) -> None:
        """Post a robot-body turn back to Hermes for memory continuity.

        Fire-and-forget: errors are logged but never raised.
        """
        try:
            await self.ask(
                query=(
                    "[ROBOT BODY SYNC] The following happened through the Reachy Mini robot:\n"
                    f"User said: {user_message}\n"
                    f"You responded: {assistant_message}\n"
                    "Remember this as part of your ongoing conversation."
                ),
                system_context=(
                    "[ROBOT BODY SYNC] The following conversation happened through your "
                    "Reachy Mini robot body. Remember it as part of your ongoing "
                    "conversation with the user. Reply with just 'ok'."
                ),
            )
            logger.debug("Synced turn to Hermes")
        except Exception as e:
            logger.debug("Failed to sync turn to Hermes: %s", e)

    @property
    def is_available(self) -> bool:
        """The HTTP bridge is stateless; it's 'available' if a key is present."""
        return bool(self.api_key)


# Async helper to fire sync_turn without awaiting
def schedule_sync(bridge: HermesBridge, user_msg: str, assistant_msg: str) -> None:
    """Schedule a sync_turn on the running event loop without blocking."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(bridge.sync_turn(user_msg, assistant_msg))
    except RuntimeError:
        # No running loop — best-effort run synchronously
        asyncio.run(bridge.sync_turn(user_msg, assistant_msg))
