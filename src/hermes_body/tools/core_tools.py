"""Core robot tool definitions for hermes-body.

These tools let the Realtime brain control robot motion and capture images.
The `ask_hermes` tool is registered separately in `openai_realtime.py` so
the bridge dependency stays optional.

Vision priority for the `camera` tool:
1. OpenAI Vision API (gpt-4o-mini)
2. Raw base64 image returned to the model (no description)
"""

import base64
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

# --- Prerecorded emotion + dance libraries (from robot venv) ---
RECORDED_MOVES = None
EmotionQueueMove = None  # type: ignore[assignment]
DanceMove = None  # type: ignore[assignment]

try:
    from reachy_mini.motion.recorded_move import RecordedMoves  # noqa: F811
    from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove  # noqa: F811

    RECORDED_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
except Exception:
    pass

try:
    from reachy_mini_dances_library.dance_move import DanceMove  # noqa: F811
except Exception:
    pass

# Map simple emotion names the model uses to the closest prerecorded animation.
_EMOTION_MAP: dict[str, str] = {
    "happy": "cheerful1",
    "sad": "sad1",
    "surprised": "surprised1",
    "curious": "curious1",
    "thinking": "thoughtful1",
    "confused": "confused1",
    "excited": "enthusiastic1",
    "scared": "scared1",
    "shy": "shy1",
    "angry": "irritated1",
    "bored": "boredom1",
    "proud": "proud1",
    "grateful": "grateful1",
    "tired": "tired1",
    "loving": "loving1",
    "fear": "fear1",
    "disgusted": "disgusted1",
    "relieved": "relief1",
    "impatient": "impatient1",
    "frustrated": "frustrated1",
    "success": "success1",
    "laughing": "laughing1",
    "welcoming": "welcoming1",
    "calming": "calming1",
}

if TYPE_CHECKING:
    from hermes_body.audio.head_wobbler import HeadWobbler
    from hermes_body.hermes_bridge import HermesBridge
    from hermes_body.moves import MovementManager

logger = logging.getLogger(__name__)


async def _analyze_image_with_openai(frame: np.ndarray, prompt: str) -> str | None:
    """Analyze an image using OpenAI's gpt-4o-mini vision endpoint.

    Returns a short text description, or None if the call fails.
    """
    try:
        import cv2
        from openai import AsyncOpenAI

        from hermes_body.config import config

        api_key = config.OPENAI_API_KEY
        if not api_key:
            logger.warning("No OPENAI_API_KEY for vision analysis")
            return None

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_image = base64.b64encode(buffer).decode("utf-8")

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
        )
        text = response.choices[0].message.content
        return text.strip() if text else None

    except Exception as e:
        logger.error("OpenAI vision analysis failed: %s", e)
        return None


@dataclass
class ToolDependencies:
    """Dependencies required by the robot tools."""

    movement_manager: "MovementManager"
    head_wobbler: "HeadWobbler"
    robot: Any  # ReachyMini instance
    camera_worker: Any | None = None
    hermes_bridge: Optional["HermesBridge"] = None


# Tool specifications in OpenAI Realtime format.
TOOL_SPECS = [
    {
        "type": "function",
        "name": "look",
        "description": (
            "Move the robot's head to look in a specific direction. Use this to direct attention or emphasize a point."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right", "up", "down", "front"],
                    "description": "Direction to look. 'front' returns to neutral.",
                }
            },
            "required": ["direction"],
        },
    },
    {
        "type": "function",
        "name": "camera",
        "description": (
            "Capture an image from the robot's camera to see what's in front of "
            "you. Use when asked about your surroundings or to identify "
            "objects/people."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "face_tracking",
        "description": (
            "Enable or disable face tracking. When enabled, the robot will automatically look at detected faces."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "True to enable face tracking, False to disable",
                }
            },
            "required": ["enabled"],
        },
    },
    {
        "type": "function",
        "name": "dance",
        "description": (
            "Perform a dance animation using the robot's full body. Use to express "
            "joy, celebrate, or respond to music. The robot has 20+ choreographed dances."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dance_name": {
                    "type": "string",
                    "enum": [
                        "groovy_sway_and_roll",
                        "headbanger_combo",
                        "simple_nod",
                        "yeah_nod",
                        "chicken_peck",
                        "side_to_side_sway",
                        "pendulum_swing",
                        "dizzy_spin",
                        "jackson_square",
                        "neck_recoil",
                        "stumble_and_recover",
                        "grid_snap",
                        "chin_lead",
                        "head_tilt_roll",
                        "interwoven_spirals",
                        "polyrhythm_combo",
                        "sharp_side_tilt",
                        "side_glance_flick",
                        "side_peekaboo",
                        "uh_huh_tilt",
                    ],
                    "description": "The dance to perform. All dances are choreographed full-body animations.",
                }
            },
            "required": ["dance_name"],
        },
    },
    {
        "type": "function",
        "name": "emotion",
        "description": (
            "Express an emotion through a prerecorded, choreographed full-body animation. "
            "The robot has 75+ animations covering joy, sadness, fear, surprise, "
            "anger, curiosity, pride, gratitude, shyness, confusion, thinking, "
            "boredom, impatience, relief, and more."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "emotion_name": {
                    "type": "string",
                    "description": (
                        "The emotion to express. Use simple names: happy, sad, surprised, "
                        "curious, thinking, confused, excited, scared, shy, angry, bored, "
                        "proud, grateful, tired, loving, fear, disgusted, relieved, "
                        "impatient, frustrated, success, laughing, welcoming, calming. "
                        "These map to the best matching prerecorded animation automatically."
                    ),
                }
            },
            "required": ["emotion_name"],
        },
    },
    {
        "type": "function",
        "name": "antennas",
        "description": (
            "Move the robot's antennas (two expressive stalks on its head) to show "
            "mood, attention, or emphasis. The antennas can move independently — "
            "use them to be more expressive during conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "enum": [
                        "curious",
                        "excited",
                        "sad",
                        "point_left",
                        "point_right",
                        "listen",
                        "surprised",
                        "shy",
                        "angry",
                        "confused",
                        "neutral",
                        "wiggle",
                        "perk_left",
                        "perk_right",
                        "droop",
                    ],
                    "description": (
                        "Antenna preset: curious (one up), excited (both up), sad (both down), "
                        "point_left/right (directional), listen (slightly forward), "
                        "surprised (both max up), shy (both drooped), angry (one up one down), "
                        "confused (asymmetric), neutral (center), wiggle (opposing), "
                        "perk_left/right (one ear up), droop (both fully down)"
                    ),
                }
            },
            "required": ["preset"],
        },
    },
    {
        "type": "function",
        "name": "stop_moves",
        "description": "Stop all current movements and clear the movement queue.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "idle",
        "description": "Do nothing and remain idle. Use when you want to stay still.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


def get_tool_specs() -> list[dict]:
    """Get the list of robot tool specifications."""
    return TOOL_SPECS


async def dispatch_tool_call(
    tool_name: str,
    arguments_json: str,
    deps: ToolDependencies,
) -> dict[str, Any]:
    """Dispatch a tool call to the appropriate handler."""
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON arguments: {arguments_json}"}

    handlers = {
        "look": _handle_look,
        "camera": _handle_camera,
        "face_tracking": _handle_face_tracking,
        "dance": _handle_dance,
        "emotion": _handle_emotion,
        "antennas": _handle_antennas,
        "stop_moves": _handle_stop_moves,
        "idle": _handle_idle,
    }

    handler = handlers.get(tool_name)
    if handler is None:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        return await handler(args, deps)
    except Exception as e:
        logger.error("Tool '%s' failed: %s", tool_name, e, exc_info=True)
        return {"error": str(e)}


async def _handle_look(args: dict, deps: ToolDependencies) -> dict:
    from hermes_body.moves import HeadLookMove

    direction = args.get("direction", "front")
    try:
        _, current_ant = deps.robot.get_current_joint_positions()
        current_head = deps.robot.get_current_head_pose()
        move = HeadLookMove(
            direction=direction,
            start_pose=current_head,
            start_antennas=tuple(current_ant),
            duration=1.0,
        )
        deps.movement_manager.queue_move(move)
        return {"status": "success", "direction": direction}
    except Exception as e:
        return {"error": str(e)}


async def _handle_camera(args: dict, deps: ToolDependencies) -> dict:
    """Capture image and get a description.

    Tries OpenAI gpt-4o-mini first; if that fails, returns a base64 image
    so the Realtime model can attempt to reason about it directly.
    """
    logger.info("Camera tool called, camera_worker=%s", deps.camera_worker is not None)

    if deps.camera_worker is None:
        return {"error": "Camera not available"}

    try:
        frame = deps.camera_worker.get_latest_frame()
        if frame is None and deps.robot is not None:
            try:
                frame = deps.robot.media.get_frame()
            except Exception as e:
                logger.error("Direct frame capture failed: %s", e)

        if frame is None:
            return {"error": "No frame available from camera"}

        vision_prompt = (
            "Describe what you see in this image. Be specific about people, "
            "objects, and the environment. Keep it concise (2-3 sentences). "
            "Speak as if you are looking through your own eyes."
        )

        # Option 1: OpenAI Vision API
        description = await _analyze_image_with_openai(frame, vision_prompt)
        if description:
            logger.info("OpenAI vision response: %s", description[:100])
            return {
                "status": "success",
                "description": description,
                "source": "openai_vision",
            }

        # Option 2: return raw image so the Realtime model still has the pixels
        try:
            import cv2

            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64_image = base64.b64encode(buffer).decode("utf-8")
            return {
                "status": "partial",
                "description": ("I captured an image but vision description was unavailable."),
                "image_b64": b64_image,
            }
        except Exception:
            return {
                "status": "partial",
                "description": ("I captured an image but couldn't analyze it. No vision processing available."),
            }
    except Exception as e:
        logger.error("Camera tool error: %s", e, exc_info=True)
        return {"error": str(e)}


async def _handle_face_tracking(args: dict, deps: ToolDependencies) -> dict:
    enabled = args.get("enabled", False)
    if deps.camera_worker is None:
        return {"error": "Camera not available for face tracking"}
    try:
        if deps.camera_worker.head_tracker is None:
            return {"error": "Face tracking not available - no head tracker initialized"}
        deps.camera_worker.set_head_tracking_enabled(enabled)
        return {"status": "success", "face_tracking": enabled}
    except Exception as e:
        return {"error": str(e)}


async def _handle_dance(args: dict, deps: ToolDependencies) -> dict:
    dance_name = args.get("dance_name", "happy")

    # Prefer the real dance library if available
    if DanceMove is not None:
        try:
            move = DanceMove(dance_name)
            deps.movement_manager.queue_move(move)
            return {"status": "success", "dance": dance_name, "source": "dance_library"}
        except KeyError:
            return {
                "error": (
                    f"Unknown dance '{dance_name}'. Available: "
                    "groovy_sway_and_roll, headbanger_combo, simple_nod, "
                    "yeah_nod, chicken_peck, side_to_side_sway, "
                    "pendulum_swing, dizzy_spin, jackson_square, "
                    "neck_recoil, stumble_and_recover, grid_snap"
                )
            }
        except Exception as e:
            return {"error": str(e)}

    # Fallback: treat as emotion
    return await _handle_emotion({"emotion_name": dance_name}, deps)


async def _handle_emotion(args: dict, deps: ToolDependencies) -> dict:
    emotion_name = args.get("emotion_name", "happy")

    # Prefer prerecorded emotion animations if available
    if RECORDED_MOVES is not None and EmotionQueueMove is not None:
        # Check if it's an exact emotion name (e.g. "scared1", "enthusiastic1")
        try:
            available = RECORDED_MOVES.list_moves()
        except Exception:
            available = []

        if emotion_name in available:
            # Exact match — play it directly
            try:
                move = EmotionQueueMove(emotion_name, RECORDED_MOVES)
                deps.movement_manager.queue_move(move)
                return {"status": "success", "emotion": emotion_name, "source": "recorded"}
            except Exception as e:
                return {"error": str(e)}

        # Try the simple-name → recorded-name mapping
        mapped = _EMOTION_MAP.get(emotion_name.lower())
        if mapped and mapped in available:
            try:
                move = EmotionQueueMove(mapped, RECORDED_MOVES)
                deps.movement_manager.queue_move(move)
                return {
                    "status": "success",
                    "emotion": emotion_name,
                    "played": mapped,
                    "source": "recorded",
                }
            except Exception as e:
                return {"error": str(e)}

        # Mapping didn't work — tell the model what's available
        return {
            "error": (
                f"Unknown emotion '{emotion_name}'. Available simple names: "
                + ", ".join(sorted(_EMOTION_MAP.keys()))
                + f". Or exact: {available[:8]}..."
            )
        }

    # Fallback: crude HeadLookMove sequences (sim or no library)
    from hermes_body.moves import HeadLookMove

    emotion_sequences: dict[str, list[str]] = {
        "happy": ["up", "front"],
        "sad": ["down"],
        "surprised": ["up", "front"],
        "curious": ["right", "left", "front"],
        "thinking": ["up", "left"],
        "confused": ["left", "right", "front"],
        "excited": ["up", "down", "up", "front"],
        "scared": ["down", "left", "right"],
        "shy": ["down", "left"],
    }
    sequence = emotion_sequences.get(emotion_name, ["front"])
    try:
        for direction in sequence:
            _, current_ant = deps.robot.get_current_joint_positions()
            current_head = deps.robot.get_current_head_pose()
            move = HeadLookMove(
                direction=direction,
                start_pose=current_head,
                start_antennas=tuple(current_ant),
                duration=0.8,
            )
            deps.movement_manager.queue_move(move)
        return {"status": "success", "emotion": emotion_name, "source": "head_look"}
    except Exception as e:
        return {"error": str(e)}


async def _handle_stop_moves(args: dict, deps: ToolDependencies) -> dict:
    deps.movement_manager.clear_move_queue()
    return {"status": "success", "message": "All movements stopped"}


async def _handle_idle(args: dict, deps: ToolDependencies) -> dict:
    return {"status": "success", "message": "Staying idle"}


async def _handle_antennas(args: dict, deps: ToolDependencies) -> dict:
    """Move antennas to an expressive preset position."""
    from hermes_body.moves import AntennaMove

    preset = args.get("preset", "neutral")
    try:
        current_head = deps.robot.get_current_head_pose()
        _, current_ant = deps.robot.get_current_joint_positions()
        move = AntennaMove(
            preset=preset,
            start_pose=current_head,
            start_antennas=tuple(current_ant),
            duration=0.6,
        )
        deps.movement_manager.queue_move(move)
        return {"status": "success", "preset": preset}
    except Exception as e:
        return {"error": str(e)}
