"""Tests for Gamepad6DOFPoseTeleop with a fake gamepad and injected pose messages, without ROS spinning."""

import time

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from lerobot.teleoperators.utils import make_teleoperator_from_config
from scipy.spatial.transform import Rotation as R

from lerobot_teleoperator_devices import Gamepad6DOFPoseTeleop, Gamepad6DOFPoseTeleopConfig

QUAT_KEYS = ["pose.x", "pose.y", "pose.z", "pose.quat_x", "pose.quat_y", "pose.quat_z", "pose.quat_w"]


class FakeGamepad:
    def __init__(self):
        self.deltas = (0.0,) * 6

    def update(self):
        pass

    def get_6dof_deltas(self):
        return self.deltas

    def gripper_command(self):
        return 1.0

    def stop(self):
        pass


def pose_msg(xyz=(0.4, 0.0, 0.3), rot=None):
    msg = PoseStamped()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = xyz
    o = msg.pose.orientation
    o.x, o.y, o.z, o.w = (rot or R.identity()).as_quat()
    return msg


def make_teleop(**overrides):
    teleop = Gamepad6DOFPoseTeleop(Gamepad6DOFPoseTeleopConfig(**overrides))
    teleop.gamepad = FakeGamepad()
    teleop._pose_callback(pose_msg())
    return teleop


def step(teleop, deltas, dt=0.1):
    """Run _update_target with a fixed dt and a fresh pose."""
    teleop.gamepad.deltas = deltas
    teleop._update_target(np.array(deltas, dtype=float), dt, time.monotonic())
    return dict(zip(teleop._pose_keys, teleop._target_pose(), strict=True))


def test_factory_builds_pose_teleop():
    assert isinstance(make_teleoperator_from_config(Gamepad6DOFPoseTeleopConfig()), Gamepad6DOFPoseTeleop)


def test_starts_at_measured_pose():
    teleop = make_teleop()
    action = teleop.get_action()
    assert list(action) == QUAT_KEYS + ["gripper.pos"]
    np.testing.assert_allclose([action[k] for k in QUAT_KEYS], [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
    assert action["gripper.pos"] == 1.0
    assert list(teleop.action_features["names"]) == QUAT_KEYS + ["gripper.pos"]


def test_sticks_move_target_with_velocity():
    teleop = make_teleop(max_linear_velocity=0.05, max_angular_velocity=0.3)
    action = step(teleop, (1.0, 0.0, -0.5, 0.0, 0.0, 1.0))
    np.testing.assert_allclose([action["pose.x"], action["pose.z"]], [0.405, 0.2975])
    quat = [action[f"pose.quat_{k}"] for k in "xyzw"]
    np.testing.assert_allclose(R.from_quat(quat).as_rotvec(), [0.0, 0.0, 0.03], atol=1e-12)


def test_target_stays_within_lead_of_measured_pose():
    teleop = make_teleop(max_position_lead=0.02, max_rotation_lead=0.1)
    for _ in range(100):  # the robot does not move
        action = step(teleop, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    assert action["pose.x"] == pytest.approx(0.42)
    quat = [action[f"pose.quat_{k}"] for k in "xyzw"]
    assert R.from_quat(quat).magnitude() == pytest.approx(0.1)


def test_stale_pose_holds_target():
    teleop = make_teleop(max_pose_age_s=0.5)
    position, rotation, _ = teleop._measured
    teleop._measured = (position, rotation, time.monotonic() - 1.0)
    assert step(teleop, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0))["pose.x"] == 0.4


def test_6d_rotation_keys():
    teleop = make_teleop(use_6d_rotation=True, use_gripper=False)
    teleop._measured = None
    rot = R.from_euler("xyz", [0.1, 0.2, 0.3])
    teleop._pose_callback(pose_msg(rot=rot))
    action = teleop.get_action()
    m = rot.as_matrix()
    assert list(action) == ["pose.x", "pose.y", "pose.z"] + [f"pose.r{i}{j}" for j in (1, 2) for i in (1, 2, 3)]
    np.testing.assert_allclose(list(action.values())[3:], [*m[:, 0], *m[:, 1]], atol=1e-12)


def test_quaternion_sign_stays_continuous():
    teleop = make_teleop(max_angular_velocity=1.0, max_rotation_lead=10.0)
    previous = None
    for _ in range(40):  # yaw through more than pi, where scipy's quaternion sign would flip
        action = step(teleop, (0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        quat = np.array([action[f"pose.quat_{k}"] for k in "xyzw"])
        if previous is not None:
            assert np.dot(quat, previous) > 0
        previous = quat
        # Let the robot follow, so the lead limit does not stop the rotation
        teleop._pose_callback(pose_msg(rot=R.from_quat(quat)))


@pytest.mark.parametrize("name", ["max_linear_velocity", "max_position_lead", "max_pose_age_s"])
def test_config_rejects_non_positive_limits(name):
    with pytest.raises(ValueError, match=name):
        Gamepad6DOFPoseTeleopConfig(**{name: 0.0})
