#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from yolo_msgs.msg import Detection, DetectionArray  # your custom message
from cv_bridge import CvBridge
from ultralytics import YOLO
from yolo_msgs.msg import Detection, DetectionArray, BoundingBox2D, Pose2D, Point2D, Vector2
import cv2

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.bridge = CvBridge()
        self.model = YOLO('/home/auctorus/runs/detect/object_dictecting4/weights/best.pt')  # your trained model path
        self.sub = self.create_subscription(Image, '/head_front_camera/rgb/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(DetectionArray, '/yolo/detections', 10)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame)
        detections = DetectionArray()
        
        detections.header.stamp = msg.header.stamp  # use original image timestamp
        detections.header.frame_id = msg.header.frame_id  # or set fixed frame if needed

        for r in results:
            for box in r.boxes:
                d = Detection()
                d.class_name = self.model.names[int(box.cls[0])]

                # Fill in bbox.center.position
                d.bbox.center.position.x = float(box.xywh[0][0])
                d.bbox.center.position.y = float(box.xywh[0][1])

                # Corrected: fill in bbox.size.x and bbox.size.y
                d.bbox.size.x = float(box.xywh[0][2])
                d.bbox.size.y = float(box.xywh[0][3])

                detections.detections.append(d)

        self.pub.publish(detections)

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()

