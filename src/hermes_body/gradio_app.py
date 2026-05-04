"""Gradio web UI for hermes-body.

Lets you start/stop a conversation in a browser, useful for sim debugging
since you can use the laptop mic instead of the robot mic.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

import gradio as gr

logger = logging.getLogger(__name__)


def launch_gradio(
    robot_name: Optional[str] = None,
    enable_camera: bool = True,
    enable_hermes: bool = True,
    enable_face_tracking: bool = True,
    head_tracker_type: Optional[str] = None,
    share: bool = False,
) -> None:
    """Launch the Gradio web UI on http://localhost:7860."""
    from hermes_body.config import config
    from hermes_body.main import HermesBodyCore

    # Single mutable container so the inner closures can re-bind it
    state: dict = {"app": None, "thread": None}

    def start_conversation():
        if state["app"] is not None:
            return "Already running"

        try:
            app = HermesBodyCore(
                robot_name=robot_name,
                enable_camera=enable_camera,
                enable_hermes=enable_hermes,
                enable_face_tracking=enable_face_tracking,
                head_tracker_type=head_tracker_type,
            )
            state["app"] = app

            def run_app():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(app.run())
                except Exception as e:
                    logger.error("App error: %s", e)
                finally:
                    loop.close()

            thread = threading.Thread(target=run_app, daemon=True)
            thread.start()
            state["thread"] = thread

            return "Started successfully"
        except Exception as e:
            return f"Error: {e}"

    def stop_conversation():
        if state["app"] is None:
            return "Not running"
        try:
            state["app"].stop()
            state["app"] = None
            return "Stopped"
        except Exception as e:
            return f"Error: {e}"

    with gr.Blocks(title="hermes-body") as demo:
        gr.Markdown(
            """
        # 🤖 hermes-body

        Reachy Mini front end for Nous Research's hermes-agent.
        Voice loop powered by OpenAI Realtime; "brain" is your local Hermes.
        """
        )

        with gr.Tab("Conversation"):
            with gr.Row():
                start_btn = gr.Button("▶️ Start", variant="primary")
                stop_btn = gr.Button("⏹️ Stop", variant="secondary")
            status_text = gr.Textbox(label="Status", interactive=False)

            start_btn.click(start_conversation, outputs=[status_text])
            stop_btn.click(stop_conversation, outputs=[status_text])

        with gr.Tab("Settings"):
            gr.Markdown(
                f"""
            ### Current Configuration

            - **OpenAI Realtime model**: `{config.OPENAI_REALTIME_MODEL}`
            - **Voice**: `{config.OPENAI_VOICE}`
            - **Hermes base URL**: `{config.HERMES_BASE_URL}`
            - **Hermes model**: `{config.HERMES_MODEL}`
            - **Camera enabled**: `{enable_camera}`
            - **Hermes enabled**: `{enable_hermes}`
            - **Face tracking**: `{enable_face_tracking}`
            - **Head tracker**: `{head_tracker_type or config.HEAD_TRACKER_TYPE}`

            Edit `.env` to change these.
            """
            )

        with gr.Tab("About"):
            gr.Markdown(
                """
            ## About hermes-body

            - **OpenAI Realtime API** — sub-second voice loop and tool dispatch
            - **hermes-agent** — open-weights "brain" with persistent memory and tools
            - **Reachy Mini** — physical embodiment

            See `PLAN.md` and `reports/` for the architectural rationale.
            """
            )

    demo.launch(share=share, server_name="0.0.0.0", server_port=7860)
