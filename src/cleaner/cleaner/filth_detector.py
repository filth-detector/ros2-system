import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
import sys
from cleaner.srv import TriggerSegmentation

class ONNXSegmentationModel:
    def __init__(self, model_path, target_size=(512, 512), mean=None, std=None,
                 input_name='pixel_values', normalize_input=True):
        self.target_size = target_size
        self.normalize_input = normalize_input
        self.input_name = input_name

        if self.normalize_input:
            self.mean = np.array(mean if mean is not None else [0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array(std if std is not None else [0.229, 0.224, 0.225], dtype=np.float32)

        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def predict(self, image_bgr):
        original_h, original_w = image_bgr.shape[:2]

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)

        image_processed = (image_resized / 255.0).astype(np.float32)
        if self.normalize_input:
            image_processed = (image_processed - self.mean) / self.std

        input_tensor = np.transpose(image_processed, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        ort_inputs = {self.input_name: input_tensor}
        logits = self.session.run(None, ort_inputs)[0][0]

        logits = np.transpose(logits, (1, 2, 0))
        upsampled_logits = cv2.resize(logits, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        pred_mask = np.argmax(upsampled_logits, axis=-1).astype(np.uint8)
        return pred_mask

    def draw_overlay(self, image_bgr, pred_mask, color=(0, 0, 255), alpha=0.5):
        mask_colored = np.zeros_like(image_bgr)
        mask_colored[pred_mask == 1] = color
        return cv2.addWeighted(image_bgr, 1.0, mask_colored, alpha, 0)

class FilthDetectorNode(Node):
    def __init__(self):
        super().__init__('filth_detector_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('model_input_name', 'pixel_values')
        self.declare_parameter('model_target_width', 512)
        self.declare_parameter('model_target_height', 512)
        self.declare_parameter('model_normalize_input', True)
        self.declare_parameter('model_mean', [0.485, 0.456, 0.406])
        self.declare_parameter('model_std', [0.229, 0.224, 0.225])

        model_path = self.get_parameter('model_path').get_parameter_value().string_value

        if not model_path:
            self.get_logger().fatal("The 'model_path' parameter is strictly required.")
            raise ValueError("You must provide the ONNX model path via ROS arguments.")

        self.get_logger().info(f'Loading ONNX model from: {model_path}')

        input_name = self.get_parameter('model_input_name').get_parameter_value().string_value
        width = self.get_parameter('model_target_width').get_parameter_value().integer_value
        height = self.get_parameter('model_target_height').get_parameter_value().integer_value
        normalize = self.get_parameter('model_normalize_input').get_parameter_value().bool_value
        mean = self.get_parameter('model_mean').get_parameter_value().double_array_value
        std = self.get_parameter('model_std').get_parameter_value().double_array_value

        self.model = ONNXSegmentationModel(
            model_path,
            target_size=(width, height),
            mean=list(mean),
            std=list(std),
            input_name=input_name,
            normalize_input=normalize)

        self.latest_image_msg = None
        self.br = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(Image, '/segmentation_mask', 10)

        self._service = self.create_service(
            TriggerSegmentation,
            'trigger_segmentation',
            self.execute_callback)

        self.get_logger().info('Filth detector service is ready.')

    def image_callback(self, msg):
        self.latest_image_msg = msg

    def execute_callback(self, request, response):
        self.get_logger().info('Executing segmentation request...')

        if self.latest_image_msg is None:
            response.success = False
            response.message = "No image has been received from the camera yet."
            self.get_logger().error(response.message)
            return response

        try:
            cv_image = self.br.imgmsg_to_cv2(self.latest_image_msg, desired_encoding='bgr8')
            pred_mask = self.model.predict(cv_image)
            mask_img = (pred_mask * 255).astype(np.uint8)
            mask_msg = self.br.cv2_to_imgmsg(mask_img, encoding="mono8")
            mask_msg.header = self.latest_image_msg.header
            self.publisher.publish(mask_msg)

            response.success = True
            response.message = "Segmentation complete and mask published."
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f'Failed to process image: {e}'
            self.get_logger().error(response.message)

        return response

def main(args=None):
    rclpy.init(args=args)
    try:
        filth_detector_node = FilthDetectorNode()
        rclpy.spin(filth_detector_node)
    except ValueError as e:
        print(f"\n[FATAL] {e}\n")
    finally:
        if 'filth_detector_node' in locals():
            filth_detector_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
