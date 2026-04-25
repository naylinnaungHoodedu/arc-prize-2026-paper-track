"""Object-centric frame encoding for ARC-style frames."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable

import numpy as np


Pixel = tuple[int, int]


def normalize_frame(frame: Any) -> np.ndarray:
    """Return a 2-D integer color grid from an ARC frame-like object.

    ARC-AGI-3 frames are described in the blueprint as 64x64 frames with a
    small color vocabulary. The encoder accepts either a 2-D color-id grid or
    a 3-D RGB/RGBA array and converts it into stable integer color ids.
    """

    arr = np.asarray(frame)
    if arr.ndim == 2:
        return arr.astype(np.int64, copy=False)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0].astype(np.int64, copy=False)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.int64, copy=False)
        return (rgb[..., 0] << 16) + (rgb[..., 1] << 8) + rgb[..., 2]
    raise ValueError(f"Expected a 2-D grid or RGB/RGBA frame, got shape {arr.shape}.")


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class ObjectComponent:
    """A connected component with stable, symbolic ARC attributes."""

    color: int
    pixels: tuple[Pixel, ...]
    frame_shape: tuple[int, int]
    motion_delta: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.pixels)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]
        return min(rows), min(cols), max(rows), max(cols)

    @property
    def centroid(self) -> tuple[float, float]:
        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]
        return float(sum(rows) / len(rows)), float(sum(cols) / len(cols))

    @property
    def shape_signature(self) -> str:
        min_r, min_c, _, _ = self.bbox
        relative = sorted((r - min_r, c - min_c) for r, c in self.pixels)
        return _stable_hash(relative)

    @property
    def component_id(self) -> str:
        return _stable_hash(
            {
                "color": self.color,
                "bbox": self.bbox,
                "shape": self.shape_signature,
                "size": self.size,
            }
        )

    def moved(self, dy: int, dx: int) -> "ObjectComponent":
        height, width = self.frame_shape
        moved_pixels = tuple(
            sorted(
                (r + dy, c + dx)
                for r, c in self.pixels
                if 0 <= r + dy < height and 0 <= c + dx < width
            )
        )
        if not moved_pixels:
            moved_pixels = self.pixels
        return ObjectComponent(
            color=self.color,
            pixels=moved_pixels,
            frame_shape=self.frame_shape,
            motion_delta=(float(dy), float(dx)),
            metadata=dict(self.metadata),
        )

    def recolored(self, color: int) -> "ObjectComponent":
        return ObjectComponent(
            color=int(color),
            pixels=self.pixels,
            frame_shape=self.frame_shape,
            motion_delta=self.motion_delta,
            metadata=dict(self.metadata),
        )

    def signature(self) -> tuple[Any, ...]:
        return (self.color, self.size, self.bbox, self.shape_signature)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.component_id,
            "color": int(self.color),
            "size": self.size,
            "bbox": list(self.bbox),
            "centroid": [round(x, 3) for x in self.centroid],
            "shape_signature": self.shape_signature,
            "motion_delta": list(self.motion_delta) if self.motion_delta is not None else None,
        }


@dataclass(frozen=True)
class ObjectSet:
    """A serializable object-centric abstraction of a frame."""

    objects: tuple[ObjectComponent, ...]
    frame_shape: tuple[int, int]
    background: int

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def colors(self) -> tuple[int, ...]:
        return tuple(sorted({obj.color for obj in self.objects}))

    def signature(self) -> tuple[Any, ...]:
        return tuple(sorted(obj.signature() for obj in self.objects))

    def primitive_signature(self) -> dict[str, Any]:
        return {
            "frame_shape": list(self.frame_shape),
            "background": int(self.background),
            "object_count": self.object_count,
            "colors": list(self.colors),
            "sizes": sorted(obj.size for obj in self.objects),
            "shape_signatures": sorted(obj.shape_signature for obj in self.objects),
        }

    def translated(self, dy: int, dx: int, color: int | None = None) -> "ObjectSet":
        objects = tuple(
            obj.moved(dy, dx) if color is None or obj.color == color else obj
            for obj in self.objects
        )
        return ObjectSet(objects=objects, frame_shape=self.frame_shape, background=self.background)

    def with_objects(self, objects: Iterable[ObjectComponent]) -> "ObjectSet":
        return ObjectSet(
            objects=tuple(sorted(objects, key=lambda obj: obj.signature())),
            frame_shape=self.frame_shape,
            background=self.background,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_shape": list(self.frame_shape),
            "background": int(self.background),
            "object_count": self.object_count,
            "colors": list(self.colors),
            "objects": [obj.to_dict() for obj in self.objects],
        }


def infer_background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors(r: int, c: int, height: int, width: int) -> Iterable[Pixel]:
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _connected_components(grid: np.ndarray, background: int) -> list[ObjectComponent]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[ObjectComponent] = []
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if visited[r, c] or color == background:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            pixels: list[Pixel] = []
            while stack:
                pr, pc = stack.pop()
                pixels.append((pr, pc))
                for nr, nc in _neighbors(pr, pc, height, width):
                    if not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(
                ObjectComponent(
                    color=color,
                    pixels=tuple(sorted(pixels)),
                    frame_shape=(height, width),
                )
            )
    return components


def _attach_motion(objects: list[ObjectComponent], previous: ObjectSet | None) -> list[ObjectComponent]:
    if previous is None:
        return objects
    previous_pool = list(previous.objects)
    enriched: list[ObjectComponent] = []
    for obj in objects:
        candidates = [
            prev
            for prev in previous_pool
            if prev.color == obj.color and prev.shape_signature == obj.shape_signature
        ]
        if not candidates:
            enriched.append(obj)
            continue
        cur_r, cur_c = obj.centroid
        best = min(
            candidates,
            key=lambda prev: abs(prev.centroid[0] - cur_r) + abs(prev.centroid[1] - cur_c),
        )
        prev_r, prev_c = best.centroid
        enriched.append(
            ObjectComponent(
                color=obj.color,
                pixels=obj.pixels,
                frame_shape=obj.frame_shape,
                motion_delta=(round(cur_r - prev_r, 3), round(cur_c - prev_c, 3)),
                metadata=dict(obj.metadata),
            )
        )
        previous_pool.remove(best)
    return enriched


def encode_frame(frame: Any, previous: ObjectSet | None = None) -> ObjectSet:
    """Encode a raw frame into connected components and symbolic attributes."""

    grid = normalize_frame(frame)
    background = infer_background(grid)
    objects = _connected_components(grid, background)
    objects = _attach_motion(objects, previous)
    objects = sorted(objects, key=lambda obj: obj.signature())
    return ObjectSet(objects=tuple(objects), frame_shape=tuple(grid.shape), background=background)


def object_distance(left: ObjectSet, right: ObjectSet) -> float:
    """A small symbolic mismatch score used by the hypothesis likelihood."""

    unmatched = list(right.objects)
    total = abs(left.object_count - right.object_count) * 4.0
    for obj in left.objects:
        if not unmatched:
            total += 4.0
            continue
        best_idx = min(
            range(len(unmatched)),
            key=lambda i: _component_distance(obj, unmatched[i]),
        )
        total += _component_distance(obj, unmatched[best_idx])
        unmatched.pop(best_idx)
    total += len(unmatched) * 4.0
    return float(total)


def _component_distance(left: ObjectComponent, right: ObjectComponent) -> float:
    color_penalty = 0.0 if left.color == right.color else 2.0
    size_penalty = abs(left.size - right.size) / max(left.size, right.size, 1)
    bbox_penalty = sum(abs(a - b) for a, b in zip(left.bbox, right.bbox)) / max(
        left.frame_shape + right.frame_shape
    )
    shape_penalty = 0.0 if left.shape_signature == right.shape_signature else 1.0
    return color_penalty + size_penalty + bbox_penalty + shape_penalty
