#!/usr/bin/env python3
"""
yolo.py

이 노드는 다수의 환경에서 수신된 카메라 이미지를 처리합니다.
- 'camera/multi_image' 토픽을 구독하여, 각 환경의 sensor_msgs/Image 배열을 받습니다.
- 각 이미지에 대해 YOLO 모델을 이용해 'valve' 객체를 탐지합니다.
- 환경별 탐지 결과(1: valve 탐지, 0: 미탐지)를 Int32MultiArray 메시지로 'valve/detections' 토픽에 퍼블리시합니다.

※ MultiEnvImage.msg (사용자 정의 메시지)
---------------------------------
# MultiEnvImage.msg 예시
std_msgs/Header header
sensor_msgs/Image[] images
---------------------------------
"""

import rclpy
from rclpy.node import Node

# 사용자 정의 메시지 (다중 환경용 이미지 배열)
# 이 메시지는 패키지 내에 별도로 정의되어 있어야 합니다.
from custom_message.msg import MultiEnvImage  

from std_msgs.msg import Int32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        # 다수의 환경 이미지가 담긴 사용자 정의 메시지(MultiEnvImage)를 구독합니다.
        self.subscription = self.create_subscription(
            MultiEnvImage,
            'camera/multi_image',  # 다중 환경 카메라 이미지가 퍼블리시되는 토픽
            self.multi_image_callback,
            10)
        self.subscription  # 참조 유지

        # 각 환경의 탐지 결과(정수 배열: 1이면 탐지, 0이면 미탐지)를 퍼블리시할 토픽
        self.detection_pub = self.create_publisher(Int32MultiArray, 'valve/detections', 10)

        self.bridge = CvBridge()

        # YOLO 모델 로드 (모델 파일 경로를 실제 파일 위치에 맞게 수정)
        self.yolo_model = YOLO('/home/vision/Downloads/minchan_yolo_320/train_franka2/weights/best.pt', verbose=False)

        self.get_logger().info("다중 환경 YOLO Detector 노드가 시작되었습니다.")

    def multi_image_callback(self, msg: MultiEnvImage):
        """
        MultiEnvImage 메시지에는 여러 환경의 카메라 이미지(sensor_msgs/Image)가 포함되어 있습니다.
        각 이미지에 대해 YOLO 예측을 수행하고, valve 탐지 결과(1 또는 0)를 리스트에 저장합니다.
        """
        detection_results = []  # 각 환경별 탐지 결과 (길이 = num_envs)
        
        # msg.images는 sensor_msgs/Image 배열 (각 항목은 한 환경의 이미지)
        for idx, image_msg in enumerate(msg.images):
            try:
                # ROS Image 메시지를 OpenCV BGR 이미지로 변환
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error(f"환경 {idx} 이미지 변환 실패: {e}")
                detection_results.append(0)
                continue

            # YOLO 예측 실행 (신뢰도(confidence)가 0.8 이상인 경우만 고려)
            results = self.yolo_model.predict(cv_image, conf=0.8, verbose=False)
            valve_detected = 0  # 기본: 미탐지

            # 결과가 존재하면 박스 데이터를 순회하면서 valve 탐지를 확인
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes.data:
                    class_id = int(box[-1])
                    class_name = self.yolo_model.names[class_id]
                    confidence = box[-2]
                    if class_name == "valve" and confidence >= 0.8:
                        valve_detected = 1
                        break

            detection_results.append(valve_detected)

        # 탐지 결과를 Int32MultiArray 메시지로 퍼블리시
        detection_msg = Int32MultiArray()
        detection_msg.data = detection_results

        # MultiArrayDimension 설정 (옵션: 환경별 데이터임을 표시)
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
