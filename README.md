# ros2smolvla_interface_lerobot

This is a fork of [lerobot-ros](https://github.com/ycheng517/lerobot-ros) made by Yifei Cheng and contains specific tweaks and additions to make it applicable to our [ROS2SmolVLA](https://una-auxme.github.io/en/projects/ros2smolvla/) project.


This repository provides a generic ROS 2 interface for the [LeRobot](https://github.com/huggingface/lerobot) framework. It acts as a lightweight wrapper to connect any [ros2_control](https://control.ros.org/rolling/index.html) or [MoveIt](https://moveit.ai/) compatible robot arm with the LeRobot ecosystem.

The fork adds the possibility to simply send ROS2 TwistStamped messages to a rostopic. This makes it easy to ingest into other control nodes. Also, it implements an option to control a parallel gripper and adds a list of observation names to easily configure the observation states to LeRobot.


A gamepad teleoperator for 6-DoF end-effector control and a keyboard teleoperator for joint position control is also provided.

**Supported control modes:**

- Joint position with ros2_control
  - Using [joint_trajectory_controller](https://control.ros.org/rolling/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html)
  - Using [position_controllers](https://control.ros.org/rolling/doc/ros2_controllers/position_controllers/doc/userdoc.html)
- End-effector velocity
  - Using [Moveit Servo](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html)
  - Using a TwistStamped ROS message topic
- Gripper control with ros2_control
  - Using [joint_trajectory_controller](https://control.ros.org/rolling/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html)
  - Using [Gripper Action Controller](https://control.ros.org/jazzy/doc/ros2_controllers/gripper_controllers/doc/userdoc.html)
  - Using [Parallel Gripper Action Controller](https://control.ros.org/jazzy/doc/ros2_controllers/parallel_gripper_controller/doc/userdoc.html)

## Video Demo

[![lerobot-ros](https://markdown-videos-api.jorgenkh.no/url?url=https%3A%2F%2Fyoutu.be%2F8U8vDyi5IAs)](https://youtu.be/8U8vDyi5IAs)

## Prerequisites

### Software Requirements

Before getting started, ensure you have the following installed:

- [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html) - This repo is only tested on Jazzy.
- [ros2_control](https://control.ros.org/rolling/index.html)
- If end-effector control is desired, then [MoveIt2](https://moveit.ai/install-moveit2/binary) needs to be installed or you have to have additional control nodes for processing the ROS topic output.
- Our [ros2smolvla_interface_camera](https://github.com/una-auxme/ros2smolvla_interface_camera) package for interfacing with a camera via ROS image messages. You can do without, but have to remove our custom camera configurations in [config.py](lerobot_robot_ros/lerobot_robot_ros/config.py)

## Usage Example of the Fork

Check out our project overview page to get an example up and running: [ros2smolvla_docker usage](https://github.com/una-auxme/ros2smolvla_docker#using-the-containers)

Once you have teleoperation working, you can use all standard LeRobot features as usual.

## Robot Integration Guide

This section describes how to integrate other ROS-based robots with Lerobot.

### Arm Control Modes

Currently the repo supports the following arm control modes:

**Option 1: Joint Position Control**

This option uses [position_controllers](https://control.ros.org/rolling/doc/ros2_controllers/position_controllers/doc/userdoc.html) in `ros2_control`. It requires the robot to have:

- `position_controllers/JointGroupPositionController` for the robot arm joints
- `joint_state_broadcaster/JointStateBroadcaster` for joint state feedback

This option is enabled by setting `action_type` to `ActionType.JOINT_POSITION` in robot config.

**Option 2: Joint Trajectory Control**

This option uses [joint_trajectory_controller](https://control.ros.org/rolling/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html) in `ros2_control`. It requires the robot to have:

- `joint_trajectory_controller/JointTrajectoryController` for the robot arm joints
- `joint_state_broadcaster/JointStateBroadcaster` for joint state feedback

This option is enabled by setting `action_type` to `ActionType.JOINT_TRAJECTORY` in robot config.

**Option 3: End-Effector Control using MoveIt 2**

This option uses [Moveit Servo](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html) in MoveIt. It requires the robot to have:

- The `moveit_servo` node for real-time end-effector control
- `joint_trajectory_controller/JointTrajectoryController` for robot arm control
- `joint_state_broadcaster/JointStateBroadcaster` for joint state feedback

This option is enabled by setting `action_type` to `ActionType.CARTESIAN_VELOCITY` in robot config. See: [ar4_ros_driver](https://github.com/ycheng517/ar4_ros_driver) for an example of using `moveit_servo`.

**Option 4: End-Effector Control using TwistStamped Messages**

This option uses a ROS topic to publish TwistStamped messages under ```/servo_node/delta_twist_cmds```. It also subscribes to a PoseStamped topic called ```/cartesian_motion_controller/current_pose``` for position feedback. You need an additional ROS node to ingest and process the output further.


This option is enabled by setting `action_type` to `ActionType.CARTESIAN_VELOCITY_TWIST_MSG` in robot config.

### Gripper Control Modes

The repo supports three gripper control modes that can be configured via the `gripper_action_type` setting:

#### Trajectory Control (`GripperActionType.TRAJECTORY`)

- Uses `JointTrajectoryController` from ros2_control
- Publishes `JointTrajectory` messages to `/gripper_controller/joint_trajectory`

#### Action Control (`GripperActionType.ACTION`)

Normal Gripper
- Uses `GripperActionController` from ros2_control when `gripper_type` is set to `GripperType.GRIPPER`
- Sends action goals to `/gripper_controller/gripper_cmd`
- Provides feedback on whether the gripper reached its target position

Parallel Gripper
- Uses `ParallelGripperActionController` from ros2_control when `gripper_type` is set to `GripperType.PARALLEL_GRIPPER`
- Sends action goals to `/gripper_action_controller/gripper_cmd`
- Provides feedback on whether the gripper reached its target position

### Code Changes to Lerobot-ros

Extend the `ROS2Robot` class in [robot.py](./lerobot_robot_ros/lerobot_robot_ros/robot.py).
This class can be a simple pass-through. It just is needed to satisfy lerobot device discovery requirements.

```python
class MyRobot(ROS2Robot):
  pass
```

Then, create a config class for your robot by sub-classing `ROS2Config` in [config.py](./lerobot_robot_ros/lerobot_robot_ros/config.py).
The name of this class must be the same as your robot class, suffixed by `Config`.
You may override joint names, gripper configurations, and other parameters as needed.

The fork adds various configurations for our robot, depending on which robot state observations and camera settings are required. 

An example config class for joint velocity control may look like this:

```python
@RobotConfig.register_subclass("my_ros2_robot")
@dataclass
class MyRobotConfig(ROS2Config):
    action_type: ActionType = ActionType.JOINT_POSITION

    observation_names: list[str] = field(
      default_factory=lambda: [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper_joint",
      ]
    )

    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            base_link="base_link",
            arm_joint_names=[
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
            ],
            gripper_joint_name="gripper_joint",
            gripper_open_position=0.0,
            gripper_close_position=1.0,
            max_linear_velocity=0.05,  # m/s
            max_angular_velocity=0.25,  # rad/s
        )
    )
```
## License

This repository is licensed under Apache-2.0 as listed in [LICENSE](LICENSE). Please also refer to the attributions in [NOTICE](NOTICE.md).