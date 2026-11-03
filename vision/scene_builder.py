"""
vision/scene_builder.py
=======================
Builds natural language prompts and immediate spoken descriptions from spatial obstacles.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from vision.spatial_reasoning import SpatialObstacle, ThreatLevel, Direction


SYSTEM_PROMPT = """You are an AI navigation assistant embedded into a smart walking stick for a visually impaired user.
Your instructions must be direct, concise, and focused on safety.
Rules:
1. Speak in 1 to 2 short sentences.
2. Immediately state any hazard directly ahead in the center path.
3. If the path is clear, state which direction is open.
4. Name specific items (e.g., 'chair ahead', 'stairs to your left').
5. Never say 'I see' or 'The picture shows'. Speak as direct verbal navigation guidance."""


def build_llm_prompt(
    obstacles: List[SpatialObstacle],
    ultrasonic_distance_cm: Optional[float] = None,
    user_query: Optional[str] = None
) -> str:
    """
    Format the current spatial perception into an LLM navigation prompt.
    """
    lines = []
    if ultrasonic_distance_cm is not None:
        lines.append(f"- Ultrasonic proximity sensor: {ultrasonic_distance_cm:.1f} cm directly ahead.")

    if not obstacles:
        lines.append("- Visual detector: No major obstacles detected in the immediate field of view.")
    else:
        for obs in obstacles:
            dist_str = f" (~{obs.estimated_distance_cm:.0f}cm)" if obs.estimated_distance_cm else ""
            lines.append(
                f"- {obs.label.capitalize()} on the {obs.direction.value}, {obs.proximity.value} range{dist_str} (Threat: {obs.threat_level.value.upper()})."
            )

    scene_summary = "\n".join(lines)
    query = user_query or "Provide clear, immediate navigation guidance."

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CURRENT SENSOR TELEMETRY:\n"
        f"{scene_summary}\n\n"
        f"USER REQUEST: {query}\n"
        f"VERBAL GUIDANCE:"
    )


def build_fast_alert_phrase(obstacles: List[SpatialObstacle], ultrasonic_distance_cm: Optional[float] = None) -> str:
    """
    Fast rule-based phrase generator for ultra-low-latency warnings (<50ms).
    Used as an immediate verbal alert before/without heavy LLM inference.
    """
    if ultrasonic_distance_cm is not None and ultrasonic_distance_cm < 40.0:
        return f"Warning! Obstacle {int(ultrasonic_distance_cm)} centimeters directly ahead. Stop."

    if not obstacles:
        return "Path ahead is clear."

    critical = [o for o in obstacles if o.threat_level == ThreatLevel.CRITICAL]
    if critical:
        top = critical[0]
        return f"Caution: {top.label} directly ahead, close."

    warnings = [o for o in obstacles if o.threat_level == ThreatLevel.WARNING]
    if warnings:
        top = warnings[0]
        return f"{top.label.capitalize()} to your {top.direction.value}."

    return "Path ahead is relatively clear."


def obstacles_to_dict(obstacles: List[SpatialObstacle]) -> List[Dict[str, Any]]:
    """Convert dataclasses to JSON-serializable dictionaries."""
    return [
        {
            "label": obs.label,
            "direction": obs.direction.value,
            "proximity": obs.proximity.value,
            "confidence": obs.confidence,
            "threat_level": obs.threat_level.value,
            "bbox": obs.bbox,
            "estimated_distance_cm": obs.estimated_distance_cm,
        }
        for obs in obstacles
    ]
