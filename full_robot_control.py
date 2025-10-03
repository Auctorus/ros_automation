#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from yolo_msgs.msg import DetectionArray
from pymoveit2 import MoveIt2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, LookupException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import cv2


class FullRobotTask(Node):
    def __init__(self):
        super().__init__('full_robot_task_node')

        self.target_shelf = None
        self.target_medicine = None
        self.task_received = False

        self.declare_parameter("camera_frame", "head_front_camera_rgb_optical_frame")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("depth_topic", "/head_front_camera/depth/image_raw")
        self.declare_parameter("det_topic", "/yolo/detections")
        self.declare_parameter("info_topic", "/head_front_camera/depth/camera_info")

        self.camera_frame = self.get_parameter("camera_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/gripper_controller/joint_trajectory', 10)
        self.create_subscription(String, '/task_command', self.task_command_callback, 10)
        self.create_subscription(CameraInfo, self.get_parameter("info_topic").value, self.info_cb, 1)
        self.create_subscription(LaserScan, '/scan_raw', self.lidar_callback, 10)

        det_sub = Subscriber(self, DetectionArray, self.get_parameter("det_topic").value, qos_profile=10)
        depth_sub = Subscriber(self, Image, self.get_parameter("depth_topic").value, qos_profile=10)
        self.sync = ApproximateTimeSynchronizer([det_sub, depth_sub], 10, 0.1)
        self.sync.registerCallback(self.synced_cb)

        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.moveit2 = MoveIt2(
            node=self,
            group_name="arm_torso",
            end_effector_name="arm_tool_link",
            base_link_name="torso_lift_link",
            joint_names=[
                "torso_lift_joint", "arm_1_joint", "arm_2_joint", "arm_3_joint",
                "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint"
            ],
        )

        self.object_detected = False
        self.last_bbox = None
        self.last_pose = None
        self.last_detection_time = self.get_clock().now()
        self.lidar_ranges = []
        self.arm_motion_triggered = False
        self.state = 'ROTATING'
        self.rotation_start_time = self.get_clock().now()
        self.movement_start_time = None
        self.search_strategy_queue = [
            'MOVE_FORWARD_AND_ROTATE',
            'MOVE_BACKWARD_AND_ROTATE',
            'SHIFT_LEFT_AND_ROTATE',
            'SHIFT_RIGHT_AND_ROTATE'
        ]

        self.OFFSET_THRESHOLD = 40
        self.LIDAR_STOP_THRESHOLD = 0.7
        self.FALLBACK_TIMEOUT = 2.0
        self.ROTATION_TIMEOUT = 10.0
        self.MOVEMENT_DURATION = 3.0

        self.timer = self.create_timer(0.1, self.control_loop)
        self.create_timer(10.0, self.set_default_task)

        self.get_logger().info("🤖 Full robot task node initialized.")

    def task_command_callback(self, msg: String):
        parts = msg.data.split(',')
        if len(parts) == 2:
            self.target_shelf = parts[0].strip().lower()
            self.target_medicine = parts[1].strip().lower()
            self.arm_motion_triggered = False
            self.last_pose = None
            self.task_received = True
            self.state = 'ROTATING'
            self.rotation_start_time = self.get_clock().now()
            self.get_logger().info(f"🧾 Task: shelf='{self.target_shelf}', medicine='{self.target_medicine}'")

    def set_default_task(self):
        if not self.task_received:
            self.target_shelf = "shelf1"
            self.target_medicine = "paracetamol"
            self.arm_motion_triggered = False
            self.last_pose = None
            self.state = 'ROTATING'
            self.rotation_start_time = self.get_clock().now()
            self.get_logger().warn("⚠️ No external task received in 10s. Using default: shelf1, paracetamol.")

    def info_cb(self, msg: CameraInfo):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]

    def lidar_callback(self, msg: LaserScan):
        self.lidar_ranges = msg.ranges

    def get_front_lidar_distance(self):
        if not self.lidar_ranges:
            return float('inf')
        center = len(self.lidar_ranges) // 2
        front = self.lidar_ranges[center - 5:center + 5]
        front = [r for r in front if r > 0.05]
        return min(front) if front else float('inf')

    def synced_cb(self, det_msg: DetectionArray, depth_msg: Image):
        if not self.target_shelf or not self.target_medicine or self.fx is None:
            return

        for det in det_msg.detections:
            label = det.class_name.lower()
            cx = det.bbox.center.position.x

            if (self.target_shelf in label) or (self.target_medicine in label):
                if cx > 0:
                    self.object_detected = True
                    self.last_bbox = det.bbox
                    self.last_detection_time = self.get_clock().now()
                    self.state = 'FORWARD'
                    self.get_logger().info(f"✅ Detected: {label}, x={cx:.2f}, w={det.bbox.size.x:.2f}")
                else:
                    self.get_logger().warn(f"⚠️ Ignored {label}: bbox center x={cx}")

            if self.target_medicine in label:
                u = int(cx)
                v = int(det.bbox.center.position.y)
                depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                depth_val = float(np.nan_to_num(depth_img[v, u], nan=0.0))

                if 0.05 < depth_val < 5.0:
                    X = (u - self.cx) * depth_val / self.fx
                    Y = (v - self.cy) * depth_val / self.fy
                    Z = depth_val

                    pose = PoseStamped()
                    pose.header.stamp = depth_msg.header.stamp
                    pose.header.frame_id = self.camera_frame
                    pose.pose.position.x = X
                    pose.pose.position.y = Y
                    pose.pose.position.z = Z
                    pose.pose.orientation.w = 1.0

                    try:
                        tf = self.tf_buffer.lookup_transform(self.base_frame, pose.header.frame_id, rclpy.time.Time())
                        pose_base = do_transform_pose(pose, tf)
                        self.last_pose = pose_base
                        self.get_logger().info("📌 Pose transformed.")
                    except LookupException:
                        self.get_logger().warn("TF unavailable.")

    def control_loop(self):
        twist = Twist()
        now = self.get_clock().now()
        time_since_last = (now - self.last_detection_time).nanoseconds / 1e9
        lidar_dist = self.get_front_lidar_distance()

        if self.arm_motion_triggered:
            return

        if self.state == 'ROTATING':
            elapsed = (now - self.rotation_start_time).nanoseconds / 1e9
            self.get_logger().info(f"🔄 ROTATING | Detected={self.object_detected}, BBox={self.last_bbox is not None}")
            if self.object_detected and self.last_bbox:
                offset = self.last_bbox.center.position.x - 320
                self.get_logger().info(f"🎯 Offset={offset:.2f}")
                if abs(offset) < self.OFFSET_THRESHOLD:
                    self.state = 'FORWARD'
                    return
                twist.angular.z = -0.004 * offset
            elif elapsed > self.ROTATION_TIMEOUT:
                if self.search_strategy_queue:
                    self.state = self.search_strategy_queue.pop(0)
                    self.movement_start_time = now
                    return
                else:
                    self.reset_search()
                    return
            else:
                twist.angular.z = 0.4

        elif self.state == 'FORWARD':
            if lidar_dist < self.LIDAR_STOP_THRESHOLD:
                self.get_logger().info(f"🚩 Close to object: {lidar_dist:.2f}m")
                if self.last_pose:
                    self.arm_motion_triggered = True
                    self.trigger_arm_motion()
                self.state = 'STOP'
                return
            if time_since_last > self.FALLBACK_TIMEOUT:
                self.get_logger().warn("❌ Lost detection.")
                self.state = 'STOP'
                return
            offset = self.last_bbox.center.position.x - 320 if self.last_bbox else 0
            twist.linear.x = 0.2
            twist.angular.z = -0.002 * offset

        elif self.state == 'STOP':
            return

        elif self.state in [
            'MOVE_FORWARD_AND_ROTATE', 'MOVE_BACKWARD_AND_ROTATE',
            'SHIFT_LEFT_AND_ROTATE', 'SHIFT_RIGHT_AND_ROTATE'
        ]:
            move_elapsed = (now - self.movement_start_time).nanoseconds / 1e9
            if move_elapsed < self.MOVEMENT_DURATION:
                if self.state == 'MOVE_FORWARD_AND_ROTATE':
                    twist.linear.x = 0.2
                elif self.state == 'MOVE_BACKWARD_AND_ROTATE':
                    twist.linear.x = -0.2
                elif self.state == 'SHIFT_LEFT_AND_ROTATE':
                    twist.linear.y = 0.2
                elif self.state == 'SHIFT_RIGHT_AND_ROTATE':
                    twist.linear.y = -0.2
            else:
                self.reset_search()
                return

        self.cmd_pub.publish(twist)

    def reset_search(self):
        self.state = 'ROTATING'
        self.rotation_start_time = self.get_clock().now()
        self.search_strategy_queue = [
            'MOVE_FORWARD_AND_ROTATE',
            'MOVE_BACKWARD_AND_ROTATE',
            'SHIFT_LEFT_AND_ROTATE',
            'SHIFT_RIGHT_AND_ROTATE'
        ]

    def trigger_arm_motion(self):
        self.get_logger().info("🢾 Executing MoveIt2 arm motion...")
        self.moveit2.plan_kinematic_path(
            target_pose=self.last_pose,
            end_effector_link="arm_tool_link"
        )
        success = self.moveit2.execute()
        if success:
            self.get_logger().info("✅ Arm reached goal. Gripping.")
            self.close_gripper()
        else:
            self.get_logger().warn("❌ Motion plan failed.")

    def close_gripper(self):
        traj = JointTrajectory()
        traj.joint_names = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
        point = JointTrajectoryPoint()
        point.positions = [0.7, 0.7]
        point.time_from_start.sec = 1
        traj.points.append(point)
        traj.header.stamp = self.get_clock().now().to_msg()
        self.gripper_pub.publish(traj)
        self.get_logger().info("🧰 Gripper command sent.")


def main(args=None):
    rclpy.init(args=args)
    node = FullRobotTask()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

