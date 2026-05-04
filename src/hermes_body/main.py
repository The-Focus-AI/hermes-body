"""hermes-body — main application entry point.

Components:
- OpenAI Realtime API for voice I/O + low-latency reasoning
- Hermes gateway for extended capabilities and persistent memory
- Reachy Mini robot for physical embodiment
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment from project root early so config picks it up.
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env", override=True)

logger = logging.getLogger(__name__)

PID_FILE = "/tmp/hermes-body.pid"


def acquire_instance_lock() -> bool:
    """Ensure only one hermes-body instance runs at a time.

    Returns True if this is the only instance. Prints an error and returns
    False if another hermes-body process is already running.
    """
    if os.path.exists(PID_FILE):
        try:
            old_pid = int(Path(PID_FILE).read_text().strip())
            os.kill(old_pid, 0)  # signal 0 checks existence without sending
            print(
                f"hermes-body is already running (pid {old_pid}). Only one instance allowed at a time.",
                file=sys.stderr,
            )
            return False
        except (OSError, ValueError):
            # PID file exists but process is dead — clean up stale file
            os.remove(PID_FILE)

    Path(PID_FILE).write_text(str(os.getpid()))
    return True


def release_instance_lock() -> None:
    """Remove the PID file."""
    try:
        if os.path.exists(PID_FILE) and int(Path(PID_FILE).read_text().strip()) == os.getpid():
            os.remove(PID_FILE)
    except Exception:
        pass


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not debug:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("hermes-body — Reachy Mini front end for Nous Research's hermes-agent"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    hermes-body                         # console mode (robot mic)
    hermes-body --gradio                # browser UI at localhost:7860
    hermes-body --debug                 # verbose logs
    hermes-body --no-hermes             # disable ask_hermes tool
    hermes-body --head-tracker yolo     # use YOLO face tracker
        """,
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch Gradio web UI instead of console mode",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        help="Robot name for connection (default: auto-discover)",
    )
    parser.add_argument("--no-camera", action="store_true", help="Disable camera functionality")
    parser.add_argument("--no-hermes", action="store_true", help="Disable Hermes integration")
    parser.add_argument("--no-face-tracking", action="store_true", help="Disable face tracking")
    parser.add_argument(
        "--head-tracker",
        type=str,
        choices=["mediapipe", "yolo"],
        help="Face tracker backend (overrides HEAD_TRACKER_TYPE env)",
    )
    return parser.parse_args()


class HermesBodyCore:
    """Core orchestrator. Wires together robot, voice loop, and Hermes."""

    def __init__(
        self,
        robot: Any | None = None,
        robot_name: str | None = None,
        enable_camera: bool = True,
        enable_hermes: bool = True,
        enable_face_tracking: bool = True,
        head_tracker_type: str | None = None,
        external_stop_event: threading.Event | None = None,
    ):
        from reachy_mini import ReachyMini

        from hermes_body.audio.head_wobbler import HeadWobbler
        from hermes_body.config import config
        from hermes_body.hermes_bridge import HermesBridge
        from hermes_body.moves import MovementManager
        from hermes_body.openai_realtime import OpenAIRealtimeHandler
        from hermes_body.tools.core_tools import ToolDependencies

        self._external_stop_event = external_stop_event
        self._owns_robot = robot is None

        # Validate config
        if enable_hermes and not config.HERMES_API_KEY:
            logger.warning(
                "HERMES_API_KEY not set — disabling Hermes integration. Pass --no-hermes to silence this warning."
            )
            enable_hermes = False

        errors = config.validate()
        # Re-validate without the hermes requirement if we just disabled it
        errors = [e for e in errors if "HERMES_API_KEY" not in e or enable_hermes]
        if errors:
            for error in errors:
                logger.error("Config error: %s", error)
            sys.exit(1)

        # Connect robot
        if robot is not None:
            self.robot = robot
            logger.info("Using provided Reachy Mini instance")
        else:
            logger.info("Connecting to Reachy Mini...")
            robot_kwargs: dict = {}
            if robot_name:
                robot_kwargs["robot_name"] = robot_name
            try:
                self.robot = ReachyMini(**robot_kwargs)
            except TimeoutError as e:
                logger.error("Connection timeout: %s", e)
                logger.error("Check that the robot is powered on and reachable.")
                sys.exit(1)
            except Exception as e:
                logger.error("Robot connection failed: %s", e)
                sys.exit(1)
            logger.info("Connected to robot: %s", self.robot.client.get_status())

        # Movement
        logger.info("Initializing movement system...")
        self.movement_manager = MovementManager(current_robot=self.robot)
        self.head_wobbler = HeadWobbler(set_speech_offsets=self.movement_manager.set_speech_offsets)

        # Hermes bridge
        self.hermes_bridge: HermesBridge | None = None
        if enable_hermes:
            logger.info(
                "Initializing Hermes bridge (base_url=%s, model=%s)...",
                config.HERMES_BASE_URL,
                config.HERMES_MODEL,
            )
            self.hermes_bridge = HermesBridge()

        # Camera + face tracking
        self.camera_worker = None
        self.head_tracker = None
        if enable_camera:
            logger.info("Initializing camera worker...")
            from hermes_body.camera_worker import CameraWorker

            tracker_type = head_tracker_type or config.HEAD_TRACKER_TYPE
            if enable_face_tracking and config.ENABLE_FACE_TRACKING:
                self.head_tracker = self._initialize_head_tracker(tracker_type)

            self.camera_worker = CameraWorker(
                reachy_mini=self.robot,
                head_tracker=self.head_tracker,
            )
            self.camera_worker.set_head_tracking_enabled(self.head_tracker is not None)

        # Tool dependencies
        self.deps = ToolDependencies(
            movement_manager=self.movement_manager,
            head_wobbler=self.head_wobbler,
            robot=self.robot,
            camera_worker=self.camera_worker,
            hermes_bridge=self.hermes_bridge,
        )

        # Realtime handler
        self.handler = OpenAIRealtimeHandler(
            deps=self.deps,
            hermes_bridge=self.hermes_bridge,
        )

        # State
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    def _initialize_head_tracker(self, tracker_type: str | None) -> Any | None:
        """Initialize the configured head tracker. None on failure."""
        if tracker_type is None:
            tracker_type = "mediapipe"

        if tracker_type == "mediapipe":
            try:
                from hermes_body.vision.mediapipe_tracker import HeadTracker

                logger.info("Initializing MediaPipe face tracker...")
                tracker = HeadTracker()
                logger.info("MediaPipe face tracker initialized")
                return tracker
            except ImportError as e:
                logger.warning("MediaPipe tracker not available: %s", e)
            except Exception as e:
                logger.error("Failed to initialize MediaPipe tracker: %s", e)

        elif tracker_type == "yolo":
            try:
                from hermes_body.vision.yolo_head_tracker import HeadTracker

                logger.info("Initializing YOLO face tracker...")
                tracker = HeadTracker(device="cpu")
                logger.info("YOLO face tracker initialized")
                return tracker
            except ImportError as e:
                logger.warning("YOLO tracker not available: %s", e)
                logger.warning("Install with: pip install ultralytics supervision")
            except Exception as e:
                logger.error("Failed to initialize YOLO tracker: %s", e)

        logger.warning("No face tracker available — face tracking disabled")
        return None

    def _should_stop(self) -> bool:
        if self._stop_event.is_set():
            return True
        if self._external_stop_event is not None and self._external_stop_event.is_set():
            return True
        return False

    async def record_loop(self) -> None:
        input_sr = self.robot.media.get_input_audio_samplerate()
        logger.info("Recording at %d Hz", input_sr)

        while not self._should_stop():
            audio_frame = self.robot.media.get_audio_sample()
            if audio_frame is not None:
                await self.handler.receive((input_sr, audio_frame))
            await asyncio.sleep(0.01)

    async def play_loop(self) -> None:
        output_sr = self.robot.media.get_output_audio_samplerate()
        logger.info("Playing at %d Hz", output_sr)

        while not self._should_stop():
            output = await self.handler.emit()
            if output is not None and isinstance(output, tuple):
                input_sr, audio_data = output
                audio_data = audio_data.flatten().astype("float32") / 32768.0
                audio_data = audio_data * 1.0  # full volume
                if input_sr != output_sr:
                    from scipy.signal import resample

                    num_samples = int(len(audio_data) * output_sr / input_sr)
                    audio_data = resample(audio_data, num_samples).astype("float32")
                self.robot.media.push_audio_sample(audio_data)
            await asyncio.sleep(0.01)

    async def run(self) -> None:
        # Test Hermes
        if self.hermes_bridge is not None:
            test_resp = await self.hermes_bridge.ask("ping", system_context="Reply 'pong'.")
            if test_resp.error:
                logger.warning(
                    "Hermes gateway not reachable (%s) — ask_hermes calls will fail until it comes back",
                    test_resp.error,
                )
            else:
                logger.info("Hermes gateway reachable")

        # Max out system volume on the robot (PCM -> 100%)
        try:
            import subprocess

            subprocess.run(
                ["amixer", "sset", "PCM", "100%"],
                capture_output=True,
                timeout=3,
            )
            logger.info("System volume set to 100%")
        except Exception:
            pass  # not on robot, or amixer not available

        # Enable motors and move to neutral
        logger.info("Enabling motors and moving to neutral position...")
        try:
            from reachy_mini.utils import create_head_pose

            self.robot.enable_motors()
            neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            self.robot.goto_target(
                head=neutral,
                antennas=[0.0, 0.0],
                duration=2.0,
                body_yaw=0.0,
            )
            time.sleep(2)
            logger.info("Robot at neutral position with motors enabled")
        except Exception as e:
            logger.error("Failed to initialize robot pose: %s", e)

        # Wire camera worker into movement manager for face tracking
        if self.camera_worker is not None:
            self.movement_manager.camera_worker = self.camera_worker
            logger.info("Face tracking connected to movement system")

        logger.info("Starting movement system...")
        self.movement_manager.start()
        self.head_wobbler.start()

        if self.camera_worker is not None:
            logger.info("Starting camera worker...")
            self.camera_worker.start()

        logger.info("Starting audio...")
        self.robot.media.start_recording()
        self.robot.media.start_playing()
        time.sleep(1)

        logger.info("Ready! Speak to me...")

        handler_task = asyncio.create_task(self.handler.start_up(), name="openai-handler")
        self._tasks = [
            handler_task,
            asyncio.create_task(self.record_loop(), name="record-loop"),
            asyncio.create_task(self.play_loop(), name="play-loop"),
        ]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Tasks cancelled")

    def stop(self) -> None:
        logger.info("Stopping...")
        self._stop_event.set()

        for task in self._tasks:
            if not task.done():
                task.cancel()

        try:
            self.head_wobbler.stop()
            self.movement_manager.stop()
        except Exception as e:
            logger.debug("Movement stop error: %s", e)

        if self.camera_worker is not None:
            try:
                self.camera_worker.stop()
            except Exception as e:
                logger.debug("Camera stop error: %s", e)

        if self._owns_robot:
            try:
                self.robot.media.close()
            except Exception as e:
                logger.debug("Media close: %s", e)
            try:
                self.robot.client.disconnect()
            except Exception as e:
                logger.debug("Robot disconnect: %s", e)

        logger.info("Stopped")


from reachy_mini.apps.app import ReachyMiniApp


class HermesBody(ReachyMiniApp):
    """Reachy Mini Apps entry point for hermes-body.

    Invoked by the Reachy Mini daemon when a user installs hermes-body from
    Hugging Face. The daemon runs `python -m hermes_body.main`, which calls
    `wrapped_run()` on this class — that opens the robot connection and calls
    our `run()` with a connected `ReachyMini` instance.
    """

    custom_app_url: str | None = None

    def run(self, reachy_mini, stop_event: threading.Event) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        app = HermesBodyCore(
            robot=reachy_mini,
            external_stop_event=stop_event,
        )

        try:
            loop.run_until_complete(app.run())
        except Exception as e:
            logger.error("Error running app: %s", e)
        finally:
            app.stop()
            loop.close()


def main() -> None:
    if not acquire_instance_lock():
        sys.exit(1)
    atexit.register(release_instance_lock)
    signal.signal(signal.SIGTERM, lambda *_: release_instance_lock() or sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: release_instance_lock() or sys.exit(0))

    args = parse_args()
    setup_logging(args.debug)

    from hermes_body.config import set_face_tracking_enabled, set_hermes_enabled

    if args.no_face_tracking:
        set_face_tracking_enabled(False)
    if args.no_hermes:
        set_hermes_enabled(False)

    if args.gradio:
        logger.info("Starting Gradio UI...")
        from hermes_body.gradio_app import launch_gradio

        launch_gradio(
            robot_name=args.robot_name,
            enable_camera=not args.no_camera,
            enable_hermes=not args.no_hermes,
            enable_face_tracking=not args.no_face_tracking,
            head_tracker_type=args.head_tracker,
        )
    else:
        app = HermesBodyCore(
            robot_name=args.robot_name,
            enable_camera=not args.no_camera,
            enable_hermes=not args.no_hermes,
            enable_face_tracking=not args.no_face_tracking,
            head_tracker_type=args.head_tracker,
        )

        try:
            asyncio.run(app.run())
        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            app.stop()


if __name__ == "__main__":
    # `python -m hermes_body.main` is how the Reachy Mini daemon launches an app:
    # it owns the ReachyMini connection and expects us to call wrapped_run() so
    # the connection is opened/closed inside its lifecycle. The standalone
    # `hermes-body` console script (defined in pyproject.toml) calls main()
    # directly and never hits this block.
    app = HermesBody()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
