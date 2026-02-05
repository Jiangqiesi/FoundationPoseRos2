#!/usr/bin/env python3
"""
FoundationPose MoveIt2 Controller for Gazebo Simulation

This controller integrates FoundationPose 6D pose estimation with MoveIt2 motion planning
for robot manipulation in Gazebo simulation environment.
"""
"""
使用方法示例：一一对应工件 ID 和模型名称，实现抓取与放置功能，如果不传模型名称，则使用默认的硬编码模型映射。
python foundationpose_moveit2_controller_place.py \
  --objects 1 2 5 \
  --model-names g0801 g0802 g0101
"""

import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from control_msgs.action import GripperCommand
from linkattacher_msgs.srv import AttachLink, DetachLink
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg
from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse
import time
import threading


class FoundationPoseMoveIt2Controller(Node):
    """
    Controller for robot manipulation using FoundationPose and MoveIt2 in Gazebo simulation.

    Subscribes to object pose topics from FoundationPose and provides motion planning
    and gripper control for grasping operations.
    """

    def __init__(self, object_ids, model_map, offset_z, approach_distance, lift_height,
                gripper_open_pos, gripper_close_pos,
                auto_move=False, enable_grasp=True):
        # Initialize node with use_sim_time enabled to sync with Gazebo simulation clock
        # Disable automatic_declare to avoid QoS parameter conflicts with MoveItPy's C++ nodes
        super().__init__(
            'foundationpose_moveit2_controller',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True),
                rclpy.parameter.Parameter('start_type_description_service', rclpy.parameter.Parameter.Type.BOOL, False),
                # Guard against invalid QoS override values coming from external parameter files.
                rclpy.parameter.Parameter(
                    'qos_overrides./clock.subscription.durability',
                    rclpy.parameter.Parameter.Type.STRING,
                    'system_default'
                ),
                rclpy.parameter.Parameter(
                    'qos_overrides./clock.subscription.history',
                    rclpy.parameter.Parameter.Type.STRING,
                    'keep_last'
                ),
                rclpy.parameter.Parameter(
                    'qos_overrides./clock.subscription.depth',
                    rclpy.parameter.Parameter.Type.INTEGER,
                    10
                ),
                rclpy.parameter.Parameter(
                    'qos_overrides./clock.subscription.reliability',
                    rclpy.parameter.Parameter.Type.STRING,
                    'reliable'
                ),
            ]
        )

        # 初始化时定义工件相对目标工件的硬编码偏移量
        self.layout_config = {
            1: (-0.02,  0.0495), # x, y
            2: (0.02, 0.0495),
            3: (0.0,    0.0),
            4: (0.0,   -0.0495),
        }
        # 放置偏移高度（比如每个物体都放在目标上方 2 cm 处）
        self.stacking_height = 0.02

        # Store parameters
        self.object_ids = object_ids
        self.model_map = model_map # [新增] 存储映射关系
        self.auto_move = auto_move
        self.offset_z = offset_z
        self.enable_grasp = enable_grasp
        self.lift_height = lift_height
        self.approach_distance = approach_distance
        self.gripper_open_pos = gripper_open_pos
        self.gripper_close_pos = gripper_close_pos

        # 如果命令行没传模型名称，就用默认的硬编码模型映射（方便调试）
        if not self.model_map and 1 in self.object_ids:
             self.model_map = {
                 1: "g0801", 
                 2: "g0802", 
                 3: "g0601", 
                 4: "g0701",
                 5: "g0101" # 假设 ID 5 是底座
             }
             self.get_logger().info("Using default hardcoded model map.")

        # State tracking
        self.latest_poses = {}
        self.subscribers = {}
        self.arm = None
        self.current_joint_positions = {}
        self.joint_state_lock = threading.Lock()

        # Callback group for concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()

        # Subscribe to joint states from Gazebo
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10,
            callback_group=self.callback_group
        )

        # Gripper action client
        self.gripper_action_client = ActionClient(
            self,
            GripperCommand,
            '/robot_hand_controller/gripper_cmd',
            callback_group=self.callback_group
        )

        # IFRA LinkAttacher service clients
        self.attach_client = self.create_client(AttachLink, '/ATTACHLINK')
        self.detach_client = self.create_client(DetachLink, '/DETACHLINK')

        # LinkAttacher configuration (adjust these names to match your Gazebo models)
        self.robot_model_name = 'rm75_robot'  # Gazebo model name for the robot
        self.robot_ee_link_name = 'Link7'  # End-effector link name
        self.object_link_name = 'link'  # Default link name for objects

        # Initialize MoveIt2
        try:
            self.get_logger().info('Initializing MoveIt2...')

            # Build MoveIt config using the same pattern as gazebo_moveit.launch.py
            moveit_config = (
                MoveItConfigsBuilder(
                    robot_name="rm75_gripper_cameras",
                    package_name="robot_moveit_config"
                )
                .robot_description()
                .robot_description_semantic()
                .robot_description_kinematics()
                .moveit_cpp(
                    file_path="config/motion_planning_python_api_tutorial.yaml"
                )
                .to_moveit_configs()
            )
            config_dict = moveit_config.to_dict()
            config_dict['use_sim_time'] = True 
            # Ensure MoveItPy's internal node does not fail on invalid /clock QoS overrides.
            config_dict['qos_overrides./clock.subscription.durability'] = 'system_default'
            config_dict['qos_overrides./clock.subscription.history'] = 'keep_last'
            config_dict['qos_overrides./clock.subscription.depth'] = 10
            config_dict['qos_overrides./clock.subscription.reliability'] = 'reliable'

            # Initialize MoveItPy
            self.get_logger().info('Initializing MoveItPy...')
            self.moveit = MoveItPy(
                node_name="moveit_py_node",
                config_dict=config_dict
            )

            self.get_logger().info('MoveIt2 initialized successfully')

            # Planning group name for Gazebo simulation
            self.group_name = "robot_arm"
            self.arm = self.moveit.get_planning_component(self.group_name)
            self.robot_model = self.moveit.get_robot_model()

            # JointModelGroup
            self.jmg = self.robot_model.get_joint_model_group(self.group_name)
            self.joint_names = self.jmg.active_joint_model_names

            if not self.joint_names:
                raise RuntimeError(f"Cannot get joint names from JointModelGroup({self.group_name})")

            self.get_logger().info(f'Planning group: {self.group_name}')
            self.get_logger().info(f'Joint names: {self.joint_names}')

        except Exception as e:
            self.get_logger().error(f'MoveIt2 initialization failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            raise

        # Wait for gripper action server
        self.get_logger().info('Waiting for gripper action server...')
        if not self.gripper_action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn('Gripper action server not available, gripper control disabled')
            self.gripper_available = False
        else:
            self.gripper_available = True
            self.get_logger().info('Gripper action server connected')

        # Wait for LinkAttacher services
        self.get_logger().info('Waiting for IFRA LinkAttacher services...')
        self.link_attacher_available = True
        if not self.attach_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/ATTACHLINK service not available, link attacher disabled')
            self.link_attacher_available = False
        if not self.detach_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/DETACHLINK service not available, link attacher disabled')
            self.link_attacher_available = False
        if self.link_attacher_available:
            self.get_logger().info('IFRA LinkAttacher services connected')

        # Subscribe to object pose topics
        for obj_id in self.object_ids:
            topic_name = f'/Current_OBJ_position_{obj_id}'
            self.subscribers[obj_id] = self.create_subscription(
                PoseStamped,
                topic_name,
                lambda msg, id=obj_id: self.pose_callback(msg, id),
                10,
                callback_group=self.callback_group
            )
            self.get_logger().info(f'Subscribed to object {obj_id} pose topic: {topic_name}')

        # Display timer
        self.timer = self.create_timer(2.0, self.display_poses)

        self.get_logger().info('Controller initialization complete')
        self.get_logger().info(f'Auto move: {"enabled" if self.auto_move else "disabled"}')
        self.get_logger().info(f'Grasp function: {"enabled" if self.enable_grasp else "disabled"}')

    def _joint_state_callback(self, msg: JointState):
        """Callback to track current joint positions from /joint_states topic."""
        with self.joint_state_lock:
            for name, position in zip(msg.name, msg.position):
                self.current_joint_positions[name] = position

    def get_current_joint_positions(self):
        """Get current joint positions thread-safely."""
        with self.joint_state_lock:
            return dict(self.current_joint_positions)

    def open_gripper(self, wait=True) -> bool:
        """Open the gripper."""
        self.get_logger().info('Opening gripper...')
        return self._send_gripper_command(self.gripper_open_pos, wait)

    def close_gripper(self, wait=True, max_effort: float = 5.0) -> bool:
        """Close the gripper with controlled force."""
        self.get_logger().info(f'Closing gripper with max_effort={max_effort}...')
        return self._send_gripper_command(self.gripper_close_pos, wait, max_effort)

    def _send_gripper_command(self, position: float, wait: bool, max_effort: float = 5.0) -> bool:
        """Send a gripper command via action client."""
        if not self.gripper_available:
            self.get_logger().warn('Gripper action server not available')
            return False

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort  # Reduced force to prevent knocking objects

        self.get_logger().info(f'Sending gripper command: position={position:.3f}')

        send_goal_future = self.gripper_action_client.send_goal_async(goal)

        if wait:
            rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=5.0)

            if not send_goal_future.done():
                self.get_logger().error('Gripper goal send timeout')
                return False

            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Gripper goal rejected')
                return False

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=10.0)

            if not result_future.done():
                self.get_logger().error('Gripper action timeout')
                return False

            self.get_logger().info('Gripper command completed')
            return True
        else:
            return True

    def attach_object(self, object_model_name: str) -> bool:
        """Attach object to robot end-effector using IFRA LinkAttacher."""
        if not self.link_attacher_available:
            self.get_logger().warn('LinkAttacher service not available')
            return False

        self.get_logger().info(f'Attaching object {object_model_name} to {self.robot_ee_link_name}...')

        req = AttachLink.Request()
        req.model1_name = self.robot_model_name
        req.link1_name = self.robot_ee_link_name
        req.model2_name = object_model_name
        req.link2_name = self.object_link_name

        future = self.attach_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        if future.done():
            result = future.result()
            self.get_logger().info(f'Attach result: {result}')
            return True
        else:
            self.get_logger().error('Attach service call timeout')
            return False

    def detach_object(self, object_model_name: str) -> bool:
        """Detach object from robot end-effector using IFRA LinkAttacher."""
        if not self.link_attacher_available:
            self.get_logger().warn('LinkAttacher service not available')
            return False

        self.get_logger().info(f'Detaching object {object_model_name} from {self.robot_ee_link_name}...')

        req = DetachLink.Request()
        req.model1_name = self.robot_model_name
        req.link1_name = self.robot_ee_link_name
        req.model2_name = object_model_name
        req.link2_name = self.object_link_name

        future = self.detach_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        if future.done():
            result = future.result()
            self.get_logger().info(f'Detach result: {result}')
            return True
        else:
            self.get_logger().error('Detach service call timeout')
            return False

    def pose_callback(self, msg, object_id):
        """Callback for receiving object poses from FoundationPose."""
        self.latest_poses[object_id] = msg

        if self.auto_move:
            self.move_to_object(object_id)

    def sync_moveit_start_state(self):
        """Synchronize MoveIt start state with current joint positions from /joint_states."""
        if self.arm is None:
            return

        current_positions = self.get_current_joint_positions()
        self.get_logger().info(f'Current joint positions: {current_positions}')

        if not current_positions:
            self.get_logger().warn('No joint state data available yet')
            return

        # Build joint positions array in the correct order
        joint_positions = []
        for name in self.joint_names:
            if name in current_positions:
                joint_positions.append(current_positions[name])
            else:
                self.get_logger().warn(f'Joint {name} not found in joint states')
                return

        # Create RobotState and set positions
        robot_state = RobotState(self.robot_model)
        robot_state.set_joint_group_active_positions(
            self.group_name,
            np.asarray(joint_positions, dtype=float)
        )
        robot_state.update()

        try:
            robot_state.enforce_bounds()
        except Exception:
            pass

        ok = self.arm.set_start_state(robot_state=robot_state)
        if not ok:
            self.get_logger().warn('set_start_state() returned False')

    def create_pose_stamped(self, x, y, z, quat_xyzw):
        """Create a PoseStamped message."""
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = float(quat_xyzw[0])
        pose.pose.orientation.y = float(quat_xyzw[1])
        pose.pose.orientation.z = float(quat_xyzw[2])
        pose.pose.orientation.w = float(quat_xyzw[3])
        return pose

    def get_downward_grasp_quat(self):
        """Return quaternion so the gripper points down (world -Z)."""
        # Assumes Link7 +X is the gripper approach axis; rotate +X -> -Z.
        # If your tool frame differs, adjust this rotation accordingly.
        quat = R.from_euler('y', -180.0, degrees=True).as_quat()
        return quat.tolist()

    def plan_to_pose(self, target_pose_stamped):
        """Plan a trajectory to the target pose using MoveIt."""
        try:
            self.sync_moveit_start_state()
            self.arm.set_goal_state(pose_stamped_msg=target_pose_stamped, pose_link="Link7")

            plan_result = self.arm.plan()

            if plan_result:
                self.get_logger().info('MoveIt2 planning successful')
                return plan_result
            else:
                self.get_logger().warn('MoveIt2 planning failed')
                return None

        except Exception as e:
            self.get_logger().error(f'MoveIt2 planning error: {e}')
            return None

    def execute_plan(self, plan):
        """Execute a planned trajectory using MoveIt."""
        if plan is None or plan.trajectory is None:
            self.get_logger().error('Empty plan or no trajectory')
            return False

        try:
            # Use MoveIt's execute method
            success = self.moveit.execute(plan.trajectory, controllers=[])

            if success:
                self.get_logger().info('Trajectory execution completed')
                return True
            else:
                self.get_logger().error('Trajectory execution failed')
                return False

        except Exception as e:
            self.get_logger().error(f'Trajectory execution error: {e}')
            return False

    def move_with_moveit(self, target_pose_stamped):
        """Plan and execute motion to target pose using MoveIt."""
        self.get_logger().info('Planning motion with MoveIt2...')

        plan = self.plan_to_pose(target_pose_stamped)

        if plan:
            return self.execute_plan(plan)
        else:
            self.get_logger().warn('MoveIt2 planning failed')
            return False

    def simple_grasp_sequence(self, object_id) -> bool:
        """
        Execute a simple vertical approach grasp sequence.

        Steps:
        1. Open gripper
        2. Move above object (approach height)
        3. Move down to grasp position
        4. Close gripper
        5. Lift object
        """
        if object_id not in self.latest_poses:
            self.get_logger().warn(f'No pose available for object {object_id}')
            return False

        pose_msg = self.latest_poses[object_id]
        pos = pose_msg.pose.position

        # Calculate positions
        approach_z = pos.z + self.offset_z + self.approach_distance
        grasp_z = pos.z + self.offset_z
        lift_z = grasp_z + self.lift_height

        # Use a vertical grasp orientation (gripper pointing down)
        # Quaternion for gripper pointing down along world -Z axis
        grasp_quat = self.get_downward_grasp_quat()

        self.get_logger().info(f'Starting grasp sequence for object {object_id}')
        self.get_logger().info(f'  Object position: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})')
        self.get_logger().info(f'  Approach Z: {approach_z:.3f}, Grasp Z: {grasp_z:.3f}, Lift Z: {lift_z:.3f}')

        # Step 1: Open gripper
        self.get_logger().info('Step 1: Opening gripper')
        if not self.open_gripper():
            self.get_logger().error('Failed to open gripper')
            return False
        time.sleep(0.5)

        # Step 2: Move to approach position (above object)
        self.get_logger().info(f'Step 2: Moving to approach position (z={approach_z:.3f}m)')
        approach_pose = self.create_pose_stamped(pos.x, pos.y, approach_z, grasp_quat)
        if not self.move_with_moveit(approach_pose):
            self.get_logger().error('Failed to reach approach position')
            return False
        time.sleep(0.5)

        # Step 3: Move down to grasp position
        self.get_logger().info(f'Step 3: Moving to grasp position (z={grasp_z:.3f}m)')
        grasp_pose = self.create_pose_stamped(pos.x, pos.y, grasp_z, grasp_quat)
        if not self.move_with_moveit(grasp_pose):
            self.get_logger().error('Failed to reach grasp position')
            # Return to approach position for safety
            self.move_with_moveit(approach_pose)
            return False
        # Wait for robot to settle and physics to stabilize
        time.sleep(1.5)

        # Step 4: Close gripper gently with low force
        self.get_logger().info('Step 4: Closing gripper gently')
        if not self.close_gripper(max_effort=-0.1):
            self.get_logger().error('Failed to close gripper')
            return False
        # Wait for gripper to firmly grasp
        time.sleep(0.5)

        # Step 4.5: Attach object to end-effector using IFRA LinkAttacher
        # [修改] 从字典获取模型名，而不是写死
        object_model_name = self.model_map.get(object_id)
        if object_model_name:
            self.get_logger().info(f'Step 4.5: Attaching object {object_model_name} (ID {object_id}) to gripper')
            if not self.attach_object(object_model_name):
                self.get_logger().warn('Failed to attach object (continuing anyway)')
        else:
            self.get_logger().warn(f'No model name found for ID {object_id}, skipping attachment.')
        time.sleep(0.5)

        # Step 5: Lift object
        self.get_logger().info(f'Step 5: Lifting object (z={lift_z:.3f}m)')
        lift_pose = self.create_pose_stamped(pos.x, pos.y, lift_z, grasp_quat)
        if not self.move_with_moveit(lift_pose):
            self.get_logger().error('Failed to lift object')
            self.open_gripper()
            return False

        self.get_logger().info(f'Grasp sequence completed for object {object_id}')
        return True

    def move_to_object(self, object_id):
        """Move to object position, optionally with grasp."""
        if object_id not in self.latest_poses:
            self.get_logger().warn(f'No pose found for object {object_id}')
            return False

        try:
            if self.enable_grasp:
                return self.simple_grasp_sequence(object_id)
            else:
                # Just move above the object without grasping
                pose_msg = self.latest_poses[object_id]
                pos = pose_msg.pose.position

                target_z = pos.z + self.offset_z + self.approach_distance
                grasp_quat = self.get_downward_grasp_quat()

                target_pose = self.create_pose_stamped(pos.x, pos.y, target_z, grasp_quat)
                return self.move_with_moveit(target_pose)

        except Exception as e:
            self.get_logger().error(f'Error moving to object: {e}')
            return False

    def move_to_target(self, source_obj_id, target_obj_id, use_layout):
        """
        将当前抓取的物体移动并放置在指定的 target_obj_id 工件位置。
        :param use_layout: 如果为 True, 则根据 layout_config 应用偏移；否则放置在中心。
        """
        try:
            if target_obj_id not in self.latest_poses:
                self.get_logger().warn(f'No pose found for target object {target_obj_id}')
                return False

            # 获取目标位置
            target_pose_msg = self.latest_poses[target_obj_id]
            target_pos = target_pose_msg.pose.position

            # 使用与抓取相同的姿态（手爪指向下方）
            grasp_quat = self.get_downward_grasp_quat()

            # 计算放置位置
            if use_layout:
                # 只有当开启布局模式时，才去查表
                offset_x, offset_y = self.layout_config.get(source_obj_id, (0.0, 0.0))
                self.get_logger().info(f"Using layout offset for OBJ {source_obj_id}: ({offset_x}, {offset_y})")
            else:
                # 否则偏移量为 0，直接放中心
                offset_x, offset_y = 0.0, 0.0
                self.get_logger().info(f"No layout offset applied (Center placement)")

            place_x = target_pos.x + offset_x
            place_y = target_pos.y + offset_y
            place_z = target_pos.z + self.offset_z + self.stacking_height # stacking_height可能需要微调，防止碰撞卡死

            self.get_logger().info(f'Starting place sequence to target object {target_obj_id}')
            self.get_logger().info(f'Place position: ({place_x:.3f}, {place_y:.3f}, {place_z:.3f})')

            # 1. 移动到目标上方位置 (Approach)
            approach_z = target_pos.z + self.offset_z + self.approach_distance
            self.get_logger().info(f'Moving to place approach position (z={approach_z:.3f}m)')
            approach_pose = self.create_pose_stamped(place_x, place_y, approach_z, grasp_quat)
            if not self.move_with_moveit(approach_pose):
                self.get_logger().error('Failed to reach place approach position')
                return False
            time.sleep(0.5)

            # 2. 移动到放置位置 (Place Down)
            self.get_logger().info(f'Moving to place position (z={place_z:.3f}m)')
            place_pose = self.create_pose_stamped(place_x, place_y, place_z, grasp_quat)
            if not self.move_with_moveit(place_pose):
                self.get_logger().error('Failed to reach place position')
                # 如果下不去，尝试回到上方
                self.move_with_moveit(approach_pose)
                return False
            time.sleep(1.5) # 等待稳定

            # Step 2.5: [新增] 解除物体绑定 (Detach)
            source_model_name = self.model_map.get(source_obj_id)
            if source_model_name:
                self.get_logger().info(f'Detaching object {source_model_name} before release')
                if not self.detach_object(source_model_name):
                    self.get_logger().warn('Failed to detach object (continuing anyway)')
            else:
                self.get_logger().warn(f'No model name found for ID {source_obj_id}, skipping detachment.')
            time.sleep(0.5) # 给仿真物理引擎一点时间结算分离

            # 3. 打开手爪释放物体 (Release)
            self.get_logger().info('Opening gripper to release object')
            if not self.open_gripper():
                self.get_logger().error('Failed to open gripper')
                return False
            time.sleep(1.0)

            # 4. 提起手臂离开物体 (Lift)
            self.get_logger().info('Lifting arm after placing object')
            lift_z = approach_z
            lift_pose = self.create_pose_stamped(place_x, place_y, lift_z, grasp_quat)
            if not self.move_with_moveit(lift_pose):
                self.get_logger().error('Failed to lift arm after placing object')
                return False
            time.sleep(0.5)

            self.get_logger().info(f'Successfully completed place sequence onto target object {target_obj_id}')
            return True
        
        except Exception as e:
            self.get_logger().error(f'Error placing object: {e}')
            return False

    def move_to_home(self):
            """移动回原始姿态 (Home)"""
            self.get_logger().info('Moving to HOME position...')
            home_pose = self.create_pose_stamped(0, 0, 0.9, [0, 0, 0, 1])
            
            return self.move_with_moveit(home_pose)

    def display_poses(self):
        """Display current object poses (called by timer)."""
        if not self.latest_poses:
            return

        # Uncomment for debugging:
        # for obj_id in sorted(self.latest_poses.keys()):
        #     msg = self.latest_poses[obj_id]
        #     pos = msg.pose.position
        #     self.get_logger().info(f'Object {obj_id}: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})')

    def manual_control_loop(self):
        """Interactive manual control loop."""
        print("\nManual control mode:")
        print("- Enter IDs (e.g. '1 2 3'): Move 1 to 3, then 2 to 3 (Last ID is target)")
        print("- Enter single ID (e.g. '1'): Move and grasp 1 only")
        print("- Enter 'status' to view current state")
        print("- Enter 'gripper open' to open gripper")
        print("- Enter 'gripper close' to close gripper")
        print("- Enter 'poses' to show detected object poses")
        print("- Enter 'joints' to show current joint positions")
        print("- Enter 'home' to move to home position")
        print("- Enter 'q' to quit")

        while rclpy.ok():
            try:
                user_input = input("\nEnter command > ").strip()

                if user_input.lower() in ['q', 'quit']:
                    print("Exiting program")
                    break

                elif user_input.lower() == 'status':
                    print(f"Objects tracked: {list(self.latest_poses.keys())}")
                    print(f"Gripper available: {self.gripper_available}")
                    continue

                elif user_input.lower() == 'gripper open':
                    if self.open_gripper():
                        print("Gripper opened")
                    else:
                        print("Failed to open gripper")
                    continue

                elif user_input.lower() == 'gripper close':
                    if self.close_gripper():
                        print("Gripper closed")
                    else:
                        print("Failed to close gripper")
                    continue

                elif user_input.lower() == 'poses':
                    if not self.latest_poses:
                        print("No object poses detected")
                    else:
                        for obj_id in sorted(self.latest_poses.keys()):
                            msg = self.latest_poses[obj_id]
                            pos = msg.pose.position
                            orient = msg.pose.orientation
                            print(f"Object {obj_id}:")
                            print(f"  Position: ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})")
                            print(f"  Orientation: ({orient.x:.4f}, {orient.y:.4f}, {orient.z:.4f}, {orient.w:.4f})")
                    continue

                elif user_input.lower() == 'joints':
                    positions = self.get_current_joint_positions()
                    if not positions:
                        print("No joint state data available")
                    else:
                        print("Current joint positions:")
                        for name in self.joint_names:
                            if name in positions:
                                print(f"  {name}: {positions[name]:.4f} rad ({math.degrees(positions[name]):.2f} deg)")
                    continue

                try:
                    # 1. 解析输入的数字列表
                    input_ids = [int(x) for x in user_input.split()]
                    
                    if not input_ids:
                        continue # 空输入，跳回循环，等待下次输入

                    # 情况 A: 只输入了一个 ID (e.g. "1") -> 只抓取，不放置
                    if len(input_ids) == 1:
                        obj_id = input_ids[0]
                        print(f"Single object mode: Grasping object {obj_id}")
                        if self.move_to_object(obj_id):
                            print(f"Done grasping {obj_id}.")
                        else:
                            print(f"Failed to grasp {obj_id}.")

                    # 情况 B: 输入了多个 ID (e.g. "1 2 3") -> 最后一个是目标，前面的是待抓取物体
                    else:
                        target_id = input_ids[-1]      # 最后一个是放置位置 (e.g. 3)
                        source_ids = input_ids[:-1]    # 前面的是要抓的物体 (e.g. [1, 2])

                        print(f"Sequence Mode: Moving {source_ids} onto Target {target_id}")

                        # 是否启用布局模式
                        if len(input_ids) == 5:
                            use_layout = True
                        else:
                            use_layout = False

                        # 检查目标位姿是否存在
                        if target_id not in self.latest_poses:
                            print(f"Error: Target object {target_id} pose not detected!")
                            continue

                        # 依次循环处理源物体
                        for i, obj_id in enumerate(source_ids):
                            print(f"\n--- Sequence {i+1}/{len(source_ids)}: Processing Object {obj_id} ---")
                            
                            # 1. 检查源物体位姿
                            if obj_id not in self.latest_poses:
                                print(f"Skipping Object {obj_id}: Pose not detected.")
                                continue # 跳过当前物体，尝试下一个

                            # 2. 抓取 (Pick)
                            print(f"STEP 1: Picking up object {obj_id}...")
                            if not self.move_to_object(obj_id):
                                print(f"Failed to pick object {obj_id}. Aborting this item.")
                                self.open_gripper()
                                continue # 跳过当前物体，尝试下一个
                            
                            # 3. 放置 (Place)
                            print(f"STEP 2: Placing onto target {target_id}...")
                            if not self.move_to_target(obj_id, target_id, use_layout):
                                print(f"Failed to place object {obj_id}.")
                                self.open_gripper()
                            else:
                                print(f"Successfully moved {obj_id} to {target_id}.")
                        
                        print("\nAll tasks in sequence completed.")

                except ValueError:
                    print("Invalid input. Enter object IDs or command.")

            except KeyboardInterrupt:
                print("\nReceived exit signal")
                break
            except Exception as e:
                print(f"Error: {e}")

    def cleanup(self):
        """Clean up resources."""
        self.get_logger().info("Cleaning up resources...")


def main():
    parser = argparse.ArgumentParser(description='FoundationPose MoveIt2 Controller for Gazebo')
    parser.add_argument('--objects', nargs='+', type=int, default=[1],
                       help='Object IDs to subscribe to (default: [1])')
    # [新增] 传入对应的 Gazebo 模型名称
    parser.add_argument('--model-names', nargs='+', type=str, default=[],
                       help='Gazebo model names corresponding to the object IDs (order matters!)')
    parser.add_argument('--auto-move', action='store_true',
                       help='Enable automatic movement mode')
    parser.add_argument('--offset-z', type=float, default=0.145,
                       help='Z offset for grasp (default: 0.15m)')
    parser.add_argument('--approach-distance', type=float, default=0.1,
                       help='Approach distance above object (default: 0.1m)')
    parser.add_argument('--lift-height', type=float, default=0.1,
                       help='Lift height after grasp (default: 0.1m)')
    parser.add_argument('--disable-grasp', action='store_true',
                       help='Disable grasping (only move to position)')
    parser.add_argument('--gripper-open-pos', type=float, default=0.8,
                       help='Gripper open position (default: 0.8)')
    parser.add_argument('--gripper-close-pos', type=float, default=0.25,
                       help='Gripper close position (default: 0.0)')

    args = parser.parse_args()

    # [新增] 创建物体 ID 到模型名称的映射字典
    model_map = {}
    if args.model_names:
        if len(args.objects) != len(args.model_names):
            print(f"Error: Number of objects ({len(args.objects)}) and model names ({len(args.model_names)}) do not match!")
            return
        # 创建字典 {1: "g0801", 2: "g0802", ...}
        model_map = dict(zip(args.objects, args.model_names))
    else:
        print("Warning: No model names provided. LinkAttacher might fail if names are not set manually.")

    rclpy.init()

    controller = None
    try:
        controller = FoundationPoseMoveIt2Controller(
            object_ids=args.objects,
            model_map=model_map,  # [新增] 将映射传入控制器
            auto_move=args.auto_move,
            offset_z=args.offset_z,
            enable_grasp=not args.disable_grasp,
            lift_height=args.lift_height,
            approach_distance=args.approach_distance,
            gripper_open_pos=args.gripper_open_pos,
            gripper_close_pos=args.gripper_close_pos,
        )

        print(f"Listening for object {args.objects} poses...")
        print("Press Ctrl+C to exit")

        if args.auto_move:
            rclpy.spin(controller)
        else:
            # Run ROS in background thread for manual control
            ros_thread = threading.Thread(target=lambda: rclpy.spin(controller))
            ros_thread.daemon = True
            ros_thread.start()

            controller.manual_control_loop()

    except KeyboardInterrupt:
        print("\nReceived exit signal, shutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if controller:
            controller.cleanup()
            controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
