#!/usr/bin/env python3
"""
yolo.py

이 노드는 다수의 환경에서 수신된 카메라 이미지를 처리합니다.
- 'camera/multi_image' 토픽을 구독하여, 각 환경의 sensor_msgs/Image 배열을 받습니다.
- 각 이미지에 대해 YOLO 모델을 이용해 'valve' 객체를 탐지합니다.
- 환경별 탐지 결과(1: valve 탐지, 0: 미탐지)를 Int32MultiArray 메시지로 'valve/detections' 토픽에 퍼블리시합니다.

※ 추가 기능:
- 받은 이미지를 'received_images/' 폴더에 저장합니다.
- 탐지된 경우, 바운딩 박스를 그려 'detections/' 폴더에 저장합니다.
"""

import rclpy
from rclpy.node import Node
import os

# 사용자 정의 메시지 (다중 환경용 이미지 배열)
from custom_message.msg import MultiImage  

from std_msgs.msg import Int32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.subscription = self.create_subscription(
            MultiImage,
            'camera/multi_image',
            self.multi_image_callback,
            10)
        self.subscription  

        self.detection_pub = self.create_publisher(Int32MultiArray, 'valve/detections', 10)
        self.bridge = CvBridge()

        self.yolo_model = YOLO('/home/vision/Downloads/minchan_yolo_320/train_franka2/weights/best.pt', verbose=False)

        # 이미지 저장 폴더 설정
        self.received_img_dir = "received_images"
        self.detected_img_dir = "detections"
        os.makedirs(self.received_img_dir, exist_ok=True)
        os.makedirs(self.detected_img_dir, exist_ok=True)

        self.get_logger().info("다중 환경 YOLO Detector 노드가 시작되었습니다.")

    def multi_image_callback(self, msg: MultiImage):
        detection_results = []  # 각 환경별 탐지 결과
        
        for idx, image_msg in enumerate(msg.images):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error(f"환경 {idx} 이미지 변환 실패: {e}")
                detection_results.append(0)
                continue

            # ✅ 1. 원본 이미지 저장
            received_img_path = os.path.join(self.received_img_dir, f"env_{idx}.jpg")
            cv2.imwrite(received_img_path, cv_image)

            # YOLO 예측 실행
            results = self.yolo_model.predict(cv_image, conf=0.8, verbose=False)
            valve_detected = 0  

            # ✅ 2. 탐지된 경우 바운딩 박스를 그려서 저장
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes.data:
                    class_id = int(box[-1])
                    class_name = self.yolo_model.names[class_id]
                    confidence = box[-2]

                    if class_name == "valve" and confidence >= 0.8:
                        valve_detected = 1

                        # 바운딩 박스 좌표
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name} {confidence:.2f}"
                        cv2.putText(cv_image, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # 탐지된 이미지 저장
                        detected_img_path = os.path.join(self.detected_img_dir, f"env_{idx}_detected.jpg")
                        cv2.imwrite(detected_img_path, cv_image)
                        break  

            detection_results.append(valve_detected)

        # 탐지 결과 퍼블리시
        detection_msg = Int32MultiArray()
        detection_msg.data = detection_results

        dim = MultiArrayDimension()
        dim.label = "environments"
        dim.size = len(detection_results)
        dim.stride = len(detection_results)
        detection_msg.layout.dim.append(dim)

        self.detection_pub.publish(detection_msg)
        self.get_logger().info(f"탐지 결과 퍼블리시: {detection_results}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
