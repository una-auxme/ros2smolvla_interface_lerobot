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

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from lerobot_robot_ros.config import ROS2InterfaceConfig
from lerobot_robot_ros.safety import limit_position, limit_rotation, rotation_from_6d, rotation_from_quat


def test_rotation_from_6d_roundtrip():
    rot = R.from_euler("xyz", [0.3, -1.2, 2.0]).as_matrix()
    np.testing.assert_allclose(rotation_from_6d(rot[:, 0], rot[:, 1]), rot, atol=1e-12)


def test_rotation_from_6d_orthonormalizes():
    rot = rotation_from_6d([2.0, 0.0, 0.0], [0.5, 3.0, 0.0])
    np.testing.assert_allclose(rot, np.eye(3), atol=1e-12)


@pytest.mark.parametrize(("col1", "col2"), [([0, 0, 0], [0, 1, 0]), ([1, 0, 0], [-2, 0, 0])])
def test_rotation_from_6d_degenerate(col1, col2):
    assert rotation_from_6d(col1, col2) is None


def test_rotation_from_quat_normalizes():
    rot = R.from_euler("xyz", [0.3, -1.2, 2.0])
    assert rotation_from_quat(3.0 * rot.as_quat()).approx_equal(rot)
    assert rotation_from_quat([0.0, 0.0, 0.0, 0.0]) is None


def test_limit_position_box_then_step():
    current = np.array([0.0, 0.0, 0.3])
    # Clipped into the box first ...
    np.testing.assert_allclose(
        limit_position([0.5, 0.0, 0.3], current, workspace_min=(-1, -1, 0.25), workspace_max=(0.1, 1, 1)),
        [0.1, 0.0, 0.3],
    )
    # ... then pulled onto the step sphere around the current position
    np.testing.assert_allclose(
        limit_position([0.5, 0.0, 0.3], current, (-1, -1, 0.25), (0.1, 1, 1), max_step=0.02),
        [0.02, 0.0, 0.3],
    )


def test_limit_position_one_sided_box():
    np.testing.assert_allclose(limit_position([0.0, 0.0, -1.0], workspace_min=(-1, -1, 0.2)), [0, 0, 0.2])


def test_limit_position_never_jumps_into_box():
    # Robot outside the box: it moves back towards it by at most max_step
    out = limit_position([0.0, 0.0, 0.5], current=[0.0, 0.0, 0.0], workspace_min=(-1, -1, 0.3), max_step=0.05)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.05])


def test_limit_position_step_requires_current():
    with pytest.raises(ValueError):
        limit_position([0, 0, 0], max_step=0.1)


def test_limit_rotation():
    current = R.from_euler("z", 0.0)
    target = R.from_euler("z", np.pi * 0.9)
    limited = limit_rotation(target, current, max_angle=0.1)
    assert limited.magnitude() == pytest.approx(0.1)
    np.testing.assert_allclose(limited.as_rotvec(), [0, 0, 0.1], atol=1e-12)
    assert limit_rotation(target, current, max_angle=3.0).approx_equal(target)


@pytest.mark.parametrize(
    "overrides",
    [
        {"workspace_min": (0, 0, 1), "workspace_max": (1, 1, 0)},
        {"workspace_min": (0, 0)},
        {"max_cartesian_step": 0.0},
        {"max_rotation_step": -0.1},
    ],
)
def test_invalid_safety_config_rejected(overrides):
    with pytest.raises(ValueError):
        ROS2InterfaceConfig(**overrides)
