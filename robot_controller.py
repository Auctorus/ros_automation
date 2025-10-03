#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from yolo_msgs.msg import DetectionArray
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import subprocess

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(DetectionArray, '/yolo/detections', self.detection_callback, 10)
        self.create_subscription(LaserScan, '/scan_raw', self.lidar_callback, 10)
        self.create_subscription(String, '/task_command', self.task_callback, 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.lidar_ranges = []
        self.object_detected = False
        self.last_bbox = None
        self.last_detection_time = None
        self.last_label = None
        self.moveit_launched = False
        self.stable_detection_start = None

        self.state = 'ROTATING'
        self.rotation_start_time = self.get_clock().now()

        self.target_class_keywords = ['paracetamol']

        self.LIDAR_STOP_THRESHOLD = 0.7
        self.ROTATION_TIMEOUT = 10.0
        self.FALLBACK_TIMEOUT = 2.0
        self.OFFSET_THRESHOLD = 100  # relaxed
        self.DETECTION_STABLE_DURATION = 0.5

        self.search_strategy_queue = [
            'MOVE_FORWARD_AND_ROTATE',
            'MOVE_BACKWARD_AND_ROTATE',
            'SHIFT_LEFT_AND_ROTATE',
            'SHIFT_RIGHT_AND_ROTATE'
        ]
        self.movement_start_time = None
        self.MOVEMENT_DURATION = 3.0

    def detection_callback(self, msg):
        self.object_detected = False
        self.last_bbox = None
        now = self.get_clock().now()

        for detection in msg.detections:
            label = detection.class_name.lower()
            if any(keyword in label for keyword in self.target_class_keywords):
                self.object_detected = True
                self.last_bbox = detection.bbox
                self.last_detection_time = now
                self.last_label = label
                self.get_logger().info(f"📦 Detected {label} at x={self.last_bbox.center.position.x:.2f}")

                offset = self.last_bbox.center.position.x - 320
                if abs(offset) < self.OFFSET_THRESHOLD:
                    if not self.stable_detection_start:
                        self.stable_detection_start = now
                else:
                    self.stable_detection_start = None
                return

        self.stable_detection_start = None

    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges

    def task_callback(self, msg):
        task = msg.data.strip()
        self.get_logger().info(f"📥 Received external task command: {task}")
        try:
            shelf, medicine = [x.strip().lower() for x in task.split(',')]
            self.target_class_keywords = [shelf, medicine]
            self.get_logger().info(f"🎯 Updated target keywords: {self.target_class_keywords}")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to parse task command '{task}': {str(e)}")

    def get_front_lidar_distance(self):
        if not self.lidar_ranges:
            return float('inf')
        center_index = len(self.lidar_ranges) // 2
        spread = 10
        front_ranges = self.lidar_ranges[center_index - spread:center_index + spread + 1]
        front_ranges = [r for r in front_ranges if r > 0.05]
        return min(front_ranges) if front_ranges else float('inf')

    def control_loop(self):
        twist = Twist()
        now = self.get_clock().now()

        if self.moveit_launched:
            self.get_logger().info("✅ MoveIt triggered. Exiting controller node.")
            self.publisher.publish(Twist())  # Stop robot
            self.destroy_node()
            rclpy.shutdown()
            return

        lidar_distance = self.get_front_lidar_distance()

        # ✅ Trigger MoveIt if object seen recently and robot is close
        if self.object_detected and self.last_detection_time:
            stable_duration = (now - self.last_detection_time).nanoseconds / 1e9
            if stable_duration < 1.0 and lidar_distance < self.LIDAR_STOP_THRESHOLD:
                self.get_logger().warn(f'🛑 Object seen & Close ({lidar_distance:.2f} m). Triggering MoveIt.')
                self.call_moveit_script()
                return

        if self.state == 'ROTATING':
            elapsed = (now - self.rotation_start_time).nanoseconds / 1e9
            if self.object_detected and self.last_bbox:
                offset = self.last_bbox.center.position.x - 320
                self.get_logger().info(f"🎯 Object offset from center: {offset:.2f}")
                if abs(offset) < self.OFFSET_THRESHOLD:
                    self.get_logger().info('✅ Object centered. Switching to FORWARD.')
                    self.state = 'FORWARD'
                    return
                twist.angular.z = -0.004 * offset
            elif elapsed > self.ROTATION_TIMEOUT:
                if self.search_strategy_queue:
                    next_strategy = self.search_strategy_queue.pop(0)
                    self.get_logger().warn(f'⏰ Timeout. Switching to {next_strategy}')
                    self.state = next_strategy
                    self.movement_start_time = now
                    return
                else:
                    self.get_logger().error('💀 All search strategies failed. Restarting search.')
                    self.reset_search()
                    return
            else:
                twist.angular.z = 0.5

        elif self.state == 'FORWARD':
            time_since_last = (now - self.last_detection_time).nanoseconds / 1e9 if self.last_detection_time else float('inf')
            self.get_logger().info(f'📡 LiDAR front distance: {lidar_distance:.2f} m')

            if time_since_last > self.FALLBACK_TIMEOUT:
                self.get_logger().warn('❌ Detection lost. Stopping.')
                self.state = 'STOP'
                return

            if self.object_detected and self.last_bbox:
                offset = self.last_bbox.center.position.x - 320
                self.get_logger().info(f"🎯 Adjusting offset: {offset:.2f}")
                twist.linear.x = 0.25
                twist.angular.z = -0.002 * offset
            else:
                twist.linear.x = 0.1

        elif self.state == 'STOP':
            twist = Twist()

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
                self.get_logger().info('🔄 Movement done. Restarting ROTATING phase.')
                self.state = 'ROTATING'
                self.rotation_start_time = now
                return

        self.publisher.publish(twist)

    def reset_search(self):
        self.state = 'ROTATING'
        self.rotation_start_time = self.get_clock().now()
        self.search_strategy_queue = [
            'MOVE_FORWARD_AND_ROTATE',
            'MOVE_BACKWARD_AND_ROTATE',
            'SHIFT_LEFT_AND_ROTATE',
            'SHIFT_RIGHT_AND_ROTATE'
        ]

    def call_moveit_script(self):
        try:
            subprocess.Popen(["ros2", "run", "tiago_manual_goal", "arm"])
            self.get_logger().info("🚀 MoveIt arm node triggered!")
            self.moveit_launched = True
        except Exception as e:
            self.get_logger().error(f"❌ Failed to launch MoveIt task: {str(e)}")
            self.moveit_launched = False

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()

