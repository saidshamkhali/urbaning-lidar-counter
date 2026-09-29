"""Class constants and the ``tracks`` / ``counts`` / ``unique`` blocks of detections.json and gt.json.

See ``docs/DATA_FORMAT.md`` for the contract. A *frame* is a list of object dicts with at least
``id``, ``cls`` and optionally ``moving`` and ``speed``.
"""

from __future__ import annotations

from collections import Counter

__all__ = [
    "FINE_CLASSES",
    "GROUPS",
    "GROUP_OF",
    "VEHICLE_CLASSES",
    "MOVING_SPEED",
    "normalize_class",
    "group_of",
    "build_tracks",
    "summarize",
]

FINE_CLASSES: tuple[str, ...] = (
    "car", "van", "truck", "bus", "trailer", "motorcycle",
    "cyclist", "escooter", "pedestrian", "other",
)
GROUPS: tuple[str, ...] = ("vehicle", "vru", "other")
GROUP_OF: dict[str, str] = {
    "car": "vehicle", "van": "vehicle", "truck": "vehicle", "bus": "vehicle",
    "trailer": "vehicle", "motorcycle": "vehicle",
    "cyclist": "vru", "escooter": "vru", "pedestrian": "vru",
    "other": "other",
}
VEHICLE_CLASSES: tuple[str, ...] = tuple(c for c in FINE_CLASSES if GROUP_OF[c] == "vehicle")

#: A track/object is "moving" when its speed exceeds this (m/s).
MOVING_SPEED = 1.0

_ALIASES = {
    "e-scooter": "escooter", "e_scooter": "escooter", "e scooter": "escooter", "scooter": "escooter",
    "motorbike": "motorcycle", "motorcyclist": "motorcycle",
    "bicycle": "cyclist", "bike": "cyclist", "bicyclist": "cyclist",
    "person": "pedestrian", "ped": "pedestrian",
}


def normalize_class(name: str) -> str:
    """Map a dataset/detector class name (e.g. ``"EScooter"``) to a fine class; unknown -> ``"other"``."""
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    return key if key in GROUP_OF else "other"


def group_of(cls: str) -> str:
    """Counting group (``vehicle`` / ``vru`` / ``other``) of a (possibly unnormalised) class name."""
    return GROUP_OF[normalize_class(cls)]


def build_tracks(frames: list[list[dict]]) -> dict[str, dict]:
    """Build the ``tracks`` block (keyed by ``str(id)``) from per-frame objects.

    ``cls`` is the majority vote over frames (ties go to the class seen first), ``moving`` is
    ``max_speed > MOVING_SPEED``.
    """
    votes: dict[str, Counter] = {}
    info: dict[str, dict] = {}
    for f, objects in enumerate(frames):
        for obj in objects:
            key = str(obj["id"])
            speed = obj.get("speed")
            speed = float(speed) if speed is not None else 0.0
            votes.setdefault(key, Counter())[normalize_class(obj["cls"])] += 1
            t = info.setdefault(key, {"first": f, "last": f, "max_speed": 0.0, "n_frames": 0})
            t["last"] = f
            t["n_frames"] += 1
            t["max_speed"] = max(t["max_speed"], speed)

    def sort_key(k: str) -> tuple[int, int | str]:
        return (0, int(k)) if k.lstrip("-").isdigit() else (1, k)

    tracks = {}
    for key in sorted(info, key=sort_key):
        cls = votes[key].most_common(1)[0][0]
        t = info[key]
        tracks[key] = {
            "cls": cls,
            "group": GROUP_OF[cls],
            "first": t["first"],
            "last": t["last"],
            "moving": bool(t["max_speed"] > MOVING_SPEED),
            "max_speed": round(float(t["max_speed"]), 3),
            "n_frames": t["n_frames"],
        }
    return tracks


def _ordered_by_class(counter: Counter) -> dict[str, int]:
    return {c: int(counter[c]) for c in FINE_CLASSES if counter[c] > 0}


def summarize(frames: list[list[dict]], tracks: dict[str, dict]) -> tuple[list[dict], dict]:
    """Compute the per-frame ``counts`` list and the sequence-level ``unique`` block.

    Per frame: objects per group, ``moving`` / ``parked`` vehicles (from each object's
    ``moving`` flag) and ``by_class`` counts of every class present. ``unique`` counts tracks
    by group and class.
    """
    counts = []
    for objects in frames:
        classes = [normalize_class(o["cls"]) for o in objects]
        groups = Counter(GROUP_OF[c] for c in classes)
        moving = sum(
            1 for o, c in zip(objects, classes) if GROUP_OF[c] == "vehicle" and o.get("moving", False)
        )
        entry = {g: int(groups[g]) for g in GROUPS}
        entry.update(moving=moving, parked=entry["vehicle"] - moving, by_class=_ordered_by_class(Counter(classes)))
        counts.append(entry)

    track_classes = Counter(normalize_class(t["cls"]) for t in tracks.values())
    unique: dict = {g: 0 for g in GROUPS}
    for cls, n in track_classes.items():
        unique[GROUP_OF[cls]] += n
    unique["by_class"] = _ordered_by_class(track_classes)
    return counts, unique
