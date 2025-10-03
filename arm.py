#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from yolo_msgs.msg import DetectionArray
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from pymoveit2 import MoveIt2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener, LookupException, TransformException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import time
import subprocess
import threading


class ArmExecutor(Node):
    def __init__(self):
        super().__init__('arm_node')

        self.target_label = 'paracetamol'
        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.last_pose = None
        self.arm_triggered = False
        self.finalized = False
        self.pose_ready = False
        self.execution_failed = False
        self.start_time = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        self.moveit2 = MoveIt2(
            node=self,
            group_name="arm_torso",
            end_effector_name="arm_tool_link",
            base_link_name="torso_lift_link",
            joint_names=[
                "torso_lift_joint", "arm_1_joint", "arm_2_joint", "arm_3_joint",
                "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint"
            ]
        )

        self.create_subscription(CameraInfo, "/head_front_camera/depth/camera_info", self.info_cb, 1)

        det_sub = Subscriber(self, DetectionArray, "/yolo/detections", qos_profile=10)
        depth_sub = Subscriber(self, Image, "/head_front_camera/depth/image_raw", qos_profile=10)
        self.sync = ApproximateTimeSynchronizer([det_sub, depth_sub], queue_size=10, slop=0.3)
        self.sync.registerCallback(self.synced_cb)

        self.create_timer(0.5, self.try_motion)
        self.get_logger().info("🤖 Arm node initialized and waiting for 'paracetamol'...")

    def info_cb(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info("📸 Camera intrinsics received.")

    def synced_cb(self, det_msg, depth_msg):
        if self.fx is None:
            return

        for det in det_msg.detections:
            if det.class_name.lower() == self.target_label:
                cx = int(det.bbox.center.position.x)
                cy = int(det.bbox.center.position.y)

                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                depth_val = float(np.nan_to_num(depth_image[cy, cx], nan=0.0))

                if 0.1 < depth_val < 2.5:
                    X = round((cx - self.cx) * depth_val / self.fx, 3)
                    Y = round((cy - self.cy) * depth_val / self.fy, 3)
                    Z = round(depth_val, 3)

                    pose = PoseStamped()
                    pose.header.stamp = depth_msg.header.stamp
                    pose.header.frame_id = "head_front_camera_color_optical_frame"
                    pose.pose.position.x = X
                    pose.pose.position.y = Y
                    pose.pose.position.z = Z
                    pose.pose.orientation.w = 1.0

                    try:
                        tf = self.tf_buffer.lookup_transform("base_footprint", pose.header.frame_id, rclpy.time.Time())
                        pose_base = PoseStamped()
                        pose_base.header = tf.header
                        pose_base.pose = do_transform_pose(pose.pose, tf)

                        pos = pose_base.pose.position
                        self.last_pose = pose_base

                        if self.is_reachable_pose(pos):
                            self.pose_ready = True
                            self.get_logger().info(f"📌 Good pose: x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f}")
                        else:
                            self.pose_ready = False
                            self.get_logger().warn(f"❌ Unreachable pose: x={pos.x:.2f}, z={pos.z:.2f}")
                    except (LookupException, TransformException) as e:
                        self.get_logger().warn(f"❌ TF failed: {str(e)}")
                else:
                    self.get_logger().warn("⚠️ Invalid depth value.")

    def is_reachable_pose(self, pos):
        return pos.x > 0.1 and pos.z > 0.3

    def try_motion(self):
        if self.arm_triggered or self.finalized or self.execution_failed:
            return

        if self.pose_ready:
            position = self.last_pose.pose.position
            orientation = self.last_pose.pose.orientation
            self.get_logger().info(
                f"🎯 Executing MoveIt to: x={position.x:.3f}, y={position.y:.3f}, z={position.z:.3f}, "
                f"q=({orientation.x:.2f}, {orientation.y:.2f}, {orientation.z:.2f}, {orientation.w:.2f})"
            )
            try:
                self.moveit2.move_to_pose(self.last_pose)  # CORRECTED
                self.arm_triggered = True
                self.start_time = time.time()
                self.create_timer(0.5, self.monitor_execution)
            except Exception as e:
                self.get_logger().error(f"🚨 MoveIt2 execution error: {str(e)}")
                self.execution_failed = True
                self.ask_user_for_next_step()
        else:
            self.rotate_robot()

    def rotate_robot(self):
        twist = Twist()
        twist.angular.z = 0.3
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info("🔁 Rotating to search for reachable pose...")

    def monitor_execution(self):
        if not self.arm_triggered or self.finalized:
            return

        if self.moveit2.is_executed():
            self.get_logger().info("✅ MoveIt executed. Closing gripper.")
            self.close_gripper()
            self.finalize_task()
        elif time.time() - self.start_time > 10:
            self.get_logger().warn("❌ MoveIt execution timeout.")
            self.arm_triggered = False  # allow retry
            self.execution_failed = True
            self.ask_user_for_next_step()

    def ask_user_for_next_step(self):
        def prompt():
            print("❓ MoveIt execution timed out or failed.")
            ans = input("Would you like to continue execution of table.py? (y/n): ").strip().lower()
            if ans == 'y':
                self.get_logger().info("📥 User chose to proceed with table.py")
                self.finalize_task()
            elif ans == 'n':
                self.get_logger().info("⏸️ Execution paused. You can retry or kill manually.")
            else:
                self.get_logger().warn("⚠️ Invalid input. Staying idle.")

        threading.Thread(target=prompt, daemon=True).start()

    def close_gripper(self):
        traj = JointTrajectory()
        traj.joint_names = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
        point = JointTrajectoryPoint()
        point.positions = [0.01, 0.01]
        point.time_from_start.sec = 1
        traj.points.append(point)
        traj.header.stamp = self.get_clock().now().to_msg()
        self.gripper_pub.publish(traj)
        self.get_logger().info("🤏 Gripper command sent.")

    def finalize_task(self):
        if self.finalized:
            return
        self.finalized = True

        try:
            self.get_logger().info("🧭 Launching table.py as next step...")
            subprocess.Popen(["python3", "table.py"])
        except Exception as e:
            self.get_logger().error(f"❌ Failed to launch table.py: {str(e)}")

        self.get_logger().info("🛑 Task complete. Shutting down.")
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ArmExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.finalize_task()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()

