#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math

class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher')
        self.publisher = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)
        self.timer = self.create_timer(0.1, self.ask_and_publish)
        self.sent = False

    def ask_and_publish(self):
        if self.sent:
            return
        x = float(input("Enter X position (meters): "))
        y = float(input("Enter Y position (meters): "))
        theta_deg = float(input("Enter θ (yaw in degrees): "))

        theta_rad = math.radians(theta_deg)
        z = math.sin(theta_rad / 2.0)
        w = math.cos(theta_rad / 2.0)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = z
        goal.pose.orientation.w = w

        self.publisher.publish(goal)
        self.get_logger().info(f"Goal sent: x={x}, y={y}, θ={theta_deg}°")
        cont = input("Send another goal? (y/n): ").strip().lower()
        if cont != 'y':
           rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

