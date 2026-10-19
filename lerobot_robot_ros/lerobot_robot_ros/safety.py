# Copyright 2025-2026 University of Augsburg, Chair of Mechatronics by Noah Böckmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Safety checks for Cartesian pose commands. Pure functions without ROS dependencies."""

import numpy as np
from scipy.spatial.transform import Rotation as R

_EPS = 1e-6


def rotation_from_6d(col1, col2) -> np.ndarray | None:
    """Build a rotation matrix from its first two columns (the "6D" representation).

    The columns do not need to be orthonormal (policy outputs rarely are); they are
    orthonormalized with Gram-Schmidt. Returns None if they are degenerate (~zero or parallel).
    """
    col1 = np.asarray(col1, dtype=float)
    col2 = np.asarray(col2, dtype=float)
    norm1 = np.linalg.norm(col1)
    if norm1 < _EPS:
        return None
    c1 = col1 / norm1
    u2 = col2 - np.dot(c1, col2) * c1
    norm2 = np.linalg.norm(u2)
    if norm2 < _EPS:
        return None
    c2 = u2 / norm2
    return np.column_stack((c1, c2, np.cross(c1, c2)))


def limit_position(
    target,
    current=None,
    workspace_min=None,
    workspace_max=None,
    max_step: float | None = None,
) -> np.ndarray:
    """Clip a target position into the workspace box, then to within `max_step` of `current`.

    The step limit is applied last, so a single command never jumps further than `max_step`,
    even when the robot currently is outside the box: it is then pulled back towards it step by step.
    """
    position = np.asarray(target, dtype=float).copy()
    if workspace_min is not None or workspace_max is not None:
        lower = -np.inf if workspace_min is None else np.asarray(workspace_min, dtype=float)
        upper = np.inf if workspace_max is None else np.asarray(workspace_max, dtype=float)
        position = np.clip(position, lower, upper)
    if max_step is not None:
        if current is None:
            raise ValueError("The current position is required to apply a step limit.")
        current = np.asarray(current, dtype=float)
        step = position - current
        dist = np.linalg.norm(step)
        if dist > max_step:
            position = current + step * (max_step / dist)
    return position


def limit_rotation(target: R, current: R, max_angle: float) -> R:
    """Limit the rotation from `current` to `target` to at most `max_angle` radians (along the geodesic)."""
    delta = current.inv() * target
    angle = delta.magnitude()
    if angle <= max_angle:
        return target
    return current * R.from_rotvec(delta.as_rotvec() * (max_angle / angle))
