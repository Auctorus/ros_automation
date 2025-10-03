#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, PoseStamped
from yolo_msgs.msg import DetectionArray
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, LookupException, TransformException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import time


class TableApproacher(Node):
    def __init__(self):
        super().__init__('table_node')

        self.target_label = 'table'
        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.last_pose = None
        self.last_detection_time = 0
        self.search_index = 0
        self.reverse_start_time = time.time()
        self.reverse_duration = 4.0  # seconds to move backward
        self.reversing_done = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(CameraInfo, "/head_front_camera/depth/camera_info", self.info_cb, 1)

        det_sub = Subscriber(self, DetectionArray, "/yolo/detections", qos_profile=10)
        depth_sub = Subscriber(self, Image, "/head_front_camera/depth/image_raw", qos_profile=10)
        self.sync = ApproximateTimeSynchronizer([det_sub, depth_sub], queue_size=10, slop=0.3)
        self.sync.registerCallback(self.synced_cb)

        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("🪑 Smart TableApproacher Node initialized...")

    def info_cb(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info("📷 Camera intrinsics received.")

    def synced_cb(self, det_msg, depth_msg):
        if self.fx is None or not self.reversing_done:
            return  # Skip detection during reversing phase

        for det in det_msg.detections:
            if det.class_name.lower() == self.target_label:
                cx = int(det.bbox.center.position.x)
                cy = int(det.bbox.center.position.y)

                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                depth_val = float(np.nan_to_num(depth_image[cy, cx], nan=0.0))

                if 0.1 < depth_val < 3.5:
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

                        self.last_pose = pose_base
                        self.last_detection_time = time.time()
                        self.search_index = 0
                        pos = pose_base.pose.position
                        self.get_logger().info(f"🟩 Table @ base: x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f}")
                    except (LookupException, TransformException) as e:
                        self.get_logger().warn(f"⚠️ TF failed: {str(e)}")

    def control_loop(self):
        now = time.time()
        twist = Twist()

        # 🔁 First phase: reverse for reverse_duration seconds
        if not self.reversing_done:
            elapsed = now - self.reverse_start_time
            if elapsed < self.reverse_duration:
                twist.linear.x = -0.2
                self.get_logger().info("↩️ Reversing before search...")
                self.cmd_pub.publish(twist)
                return
            else:
                self.reversing_done = True
                self.search_start_time = now
                self.get_logger().info("✅ Reverse complete. Starting search...")

        # ✅ Table was recently detected
        if self.last_pose and now - self.last_detection_time < 2.0:
            pos = self.last_pose.pose.position
            if pos.x < 0.7:
                self.get_logger().info("🛑 Reached table (x < 0.7m). Stopping.")
                self.cmd_pub.publish(Twist())
                return
            twist.linear.x = 0.2
            twist.angular.z = -0.6 * pos.y
            self.cmd_pub.publish(twist)
            self.get_logger().info(f"🎯 Approaching: x={pos.x:.2f}, y={pos.y:.2f}")
            return

        # 🔍 Search Strategy
        pattern_durations = [6.0, 2.5, 2.5, 6.0, 4.0, 3.0]
        current_duration = pattern_durations[self.search_index]
        time_since_search = now - self.search_start_time

        if time_since_search > current_duration:
            self.search_index = (self.search_index + 1) % len(pattern_durations)
            self.search_start_time = now

        if self.search_index == 0:
            twist.angular.z = 0.5
            self.get_logger().info("🔍 Searching [0]: rotating left ⟲")
        elif self.search_index == 1:
            twist.linear.y = 0.2
            self.get_logger().info("🔍 Searching [1]: sidestep left ⬅️")
        elif self.search_index == 2:
            twist.linear.y = -0.2
            self.get_logger().info("🔍 Searching [2]: sidestep right ➡️")
        elif self.search_index == 3:
            twist.angular.z = -0.5
            self.get_logger().info("🔍 Searching [3]: rotating right ⟳")
        elif self.search_index == 4:
            twist.linear.x = 0.15
            twist.angular.z = 0.2
            self.get_logger().info("🔍 Searching [4]: spiral forward 🔁")
        elif self.search_index == 5:
            twist.linear.x = -0.15
            self.get_logger().info("🔍 Searching [5]: stepping back ↩️")

        self.cmd_pub.publish(twist)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = TableApproacher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

