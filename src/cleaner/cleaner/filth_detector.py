import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
import sys

class SegformerONNX:
    def __init__(self, model_path, target_size=(512, 512)):
        self.target_size = target_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def predict(self, image_bgr):
        original_h, original_w = image_bgr.shape[:2]
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # TODO: test without
        image_normalized = (image_resized / 255.0).astype(np.float32)
        image_normalized = (image_normalized - self.mean) / self.std
        
        input_tensor = np.transpose(image_normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        ort_inputs = {'pixel_values': input_tensor}
        logits = self.session.run(None, ort_inputs)[0][0]
        
        logits = np.transpose(logits, (1, 2, 0))
        upsampled_logits = cv2.resize(logits, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        pred_mask = np.argmax(upsampled_logits, axis=-1).astype(np.uint8)
        return pred_mask

    def draw_overlay(self, image_bgr, pred_mask, color=(0, 0, 255), alpha=0.5):
        mask_colored = np.zeros_like(image_bgr)
        mask_colored[pred_mask == 1] = color
        return cv2.addWeighted(image_bgr, 1.0, mask_colored, alpha, 0)

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.declare_parameter('model_path', '')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value

        if not model_path:
            self.get_logger().fatal("The 'model_path' parameter is strictly required.")
            raise ValueError("You must provide the ONNX model path via ROS arguments.")

        self.get_logger().info(f'Loading ONNX model from: {model_path}')

        self.segformer = SegformerONNX(model_path)

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10 
        )
        self.publisher = self.create_publisher(Image, '/segmentation_mask', 10)

        self.br = CvBridge()
        self.get_logger().info('Image subscriber node started.')

    def image_callback(self, msg):
        try:
            cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            pred_mask = self.segformer.predict(cv_image)

            mask_img = (pred_mask * 255).astype(np.uint8)
            mask_msg = self.br.cv2_to_imgmsg(mask_img, encoding="mono8")
            mask_msg.header = msg.header

            self.publisher.publish(mask_msg)

        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        image_subscriber = ImageSubscriber()
        rclpy.spin(image_subscriber)
    except ValueError as e:
        print(f"\n[FATAL] {e}\n")
    finally:
        if 'image_subscriber' in locals():
            image_subscriber.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
