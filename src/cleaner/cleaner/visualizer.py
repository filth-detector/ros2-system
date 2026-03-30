#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseArray
from cv_bridge import CvBridge

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFrame, QSlider, 
                             QDoubleSpinBox, QFormLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from enum import Enum

from std_srvs.srv import Trigger
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter as MsgParameter, ParameterType
from rclpy.parameter import Parameter as RclpyParameter

class RobotState(Enum):
    IDLE = 0
    PLANNING = 1
    CLEANING = 2

class ControlCenterNode(Node):
    def __init__(self):
        super().__init__('control_center_node')
        self.br = CvBridge()
        
        self.declare_parameter('resolution_m_per_px', 0.01)
        
        self.latest_image = None
        self.latest_mask = None
        self.latest_inflated_mask = None
        self.latest_centroids = None
        self.latest_path = None
        
        self.camera_fps = 0.0
        self.prev_frame_time = time.time()
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_cb, 10)
        self.create_subscription(Image, '/segmentation_mask', self.mask_cb, 10)
        self.create_subscription(Image, '/inflated_mask', self.inflated_cb, 10)
        self.create_subscription(PoseArray, '/filth_centroids', self.centroids_cb, 10)
        self.create_subscription(Path, '/cleaning_path', self.path_cb, 10)
        
        self.seg_client = self.create_client(Trigger, 'trigger_segmentation')
        self.blob_client = self.create_client(Trigger, 'extract_blobs')
        self.path_client = self.create_client(Trigger, 'generate_trajectory')
        self.clean_client = self.create_client(Trigger, 'start_cleaning')

        self.param_client = self.create_client(SetParameters, '/path_planner/set_parameters')

    def image_cb(self, msg): 
        self.latest_image = msg
        
        current_time = time.time()
        time_diff = current_time - self.prev_frame_time
        if time_diff > 0:
            current_fps = 1.0 / time_diff
            self.camera_fps = 0.9 * self.camera_fps + 0.1 * current_fps
        self.prev_frame_time = current_time

    def mask_cb(self, msg): self.latest_mask = msg
    def inflated_cb(self, msg): self.latest_inflated_mask = msg
    def centroids_cb(self, msg): self.latest_centroids = msg
    def path_cb(self, msg): self.latest_path = msg

    def publish_twist(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def push_parameters_to_planner(self, step_size, inflation, resolution):
        self.set_parameters([RclpyParameter('resolution_m_per_px', RclpyParameter.Type.DOUBLE, float(resolution))])

        if self.param_client.wait_for_service(timeout_sec=0.2):
            req = SetParameters.Request()
            
            p1 = MsgParameter()
            p1.name = 'step_size_px'
            p1.value.type = ParameterType.PARAMETER_INTEGER
            p1.value.integer_value = int(step_size)

            p2 = MsgParameter()
            p2.name = 'inflation_radius_px'
            p2.value.type = ParameterType.PARAMETER_INTEGER
            p2.value.integer_value = int(inflation)

            p3 = MsgParameter()
            p3.name = 'resolution_m_per_px'
            p3.value.type = ParameterType.PARAMETER_DOUBLE
            p3.value.double_value = float(resolution)

            req.parameters = [p1, p2, p3]
            
            future = self.param_client.call_async(req)
            future.add_done_callback(self._param_response_callback)
        else:
            self.get_logger().warn("Failed to connect to the planning module. Parameters not updated.")

    def _param_response_callback(self, future):
        try:
            response = future.result()
            all_successful = all(res.successful for res in response.results)
            if all_successful:
                self.get_logger().info("Planning parameters successfully updated!")
            else:
                self.get_logger().warn("The planning module rejected some parameters.")
        except Exception as e:
            self.get_logger().error(f"Error updating parameters: {e}")


class ControlCenterGUI(QMainWindow):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.state = RobotState.IDLE
        self.auto_sequence_active = False
        self.mask_check_attempts = 0
        self.FPS_CAP = 30
        self.FRAME_TIME = int(1000 / self.FPS_CAP)
        
        self.init_ui()
        
        self.ros_timer = QTimer()
        self.ros_timer.timeout.connect(self.spin_ros)
        self.ros_timer.start(self.FRAME_TIME) 

        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_feed)
        self.video_timer.start(self.FRAME_TIME) 

        self.mask_poll_timer = QTimer()
        self.mask_poll_timer.timeout.connect(self.verify_segmentation_result)

    def init_ui(self):
        self.setWindowTitle('Valymo Roboto Valdymo Centras')
        self.resize(1150, 750)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        video_layout = QVBoxLayout()
        self.video_label = QLabel("Laukiama kameros vaizdo...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        self.video_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_label)
        main_layout.addLayout(video_layout, stretch=2)

        control_layout = QVBoxLayout()
        self.status_label = QLabel("Būsena: BŪDĖJIMAS (IDLE)")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 18px; color: green;")
        control_layout.addWidget(self.status_label)
        
        self.btn_segment = QPushButton("1. Segmentuoti nešvarumus")
        self.btn_blobs = QPushButton("2. Analizuoti nešvarumų zonas")
        self.btn_path = QPushButton("3. Generuoti valymo trajektoriją")
        self.btn_clean = QPushButton("4. Valyti")
        
        self.btn_auto = QPushButton("Automatinis valymas")
        self.btn_auto.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 12px; font-size: 14px;")

        self.btn_segment.clicked.connect(self.cmd_segment)
        self.btn_blobs.clicked.connect(self.cmd_extract_blobs)
        self.btn_path.clicked.connect(self.cmd_generate_path)
        self.btn_clean.clicked.connect(self.cmd_clean)
        self.btn_auto.clicked.connect(self.cmd_auto_clean)

        control_layout.addWidget(self.btn_segment)
        control_layout.addWidget(self.btn_blobs)
        control_layout.addWidget(self.btn_path)
        control_layout.addWidget(self.btn_clean)
        
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        control_layout.addWidget(line1)
        control_layout.addWidget(self.btn_auto)
        control_layout.addSpacing(10)

        param_label = QLabel("Konfigūracija (Parametrai)")
        param_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        control_layout.addWidget(param_label)

        param_form = QFormLayout()

        self.lbl_step_val = QLabel("20 px")
        self.slider_step = QSlider(Qt.Horizontal)
        self.slider_step.setRange(5, 100)
        self.slider_step.setValue(20)
        self.slider_step.valueChanged.connect(lambda v: self.lbl_step_val.setText(f"{v} px"))
        self.slider_step.sliderReleased.connect(self.send_parameters)
        
        step_row = QHBoxLayout()
        step_row.addWidget(self.slider_step)
        step_row.addWidget(self.lbl_step_val)
        param_form.addRow("Žingsnis (Tankumas):", step_row)

        self.lbl_inf_val = QLabel("10 px")
        self.slider_inflation = QSlider(Qt.Horizontal)
        self.slider_inflation.setRange(0, 50)
        self.slider_inflation.setValue(10)
        self.slider_inflation.valueChanged.connect(lambda v: self.lbl_inf_val.setText(f"{v} px"))
        self.slider_inflation.sliderReleased.connect(self.send_parameters)
        
        inf_row = QHBoxLayout()
        inf_row.addWidget(self.slider_inflation)
        inf_row.addWidget(self.lbl_inf_val)
        param_form.addRow("Išplėtimo spindulys:", inf_row)

        self.spin_res = QDoubleSpinBox()
        self.spin_res.setRange(0.001, 0.100)
        self.spin_res.setSingleStep(0.001)
        self.spin_res.setDecimals(3)
        self.spin_res.setValue(0.01)
        self.spin_res.editingFinished.connect(self.send_parameters)
        
        param_form.addRow("Skiriamoji geba (m/px):", self.spin_res)
        
        control_layout.addLayout(param_form)
        
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        control_layout.addWidget(line2)

        move_label = QLabel("Rankinis valdymas (W, A, S, D, Q, E)")
        move_label.setStyleSheet("font-weight: bold;")
        control_layout.addWidget(move_label)

        grid_layout = QVBoxLayout()
        row1, row2, row3 = QHBoxLayout(), QHBoxLayout(), QHBoxLayout()
        
        self.btn_up = QPushButton("Pirmyn (W)")
        self.btn_rot_l = QPushButton("Sukti Kairėn (Q)")
        self.btn_left = QPushButton("Kairėn (A)")
        self.btn_right = QPushButton("Dešinėn (D)")
        self.btn_rot_r = QPushButton("Sukti Dešinėn (E)")
        self.btn_down = QPushButton("Atgal (S)")
        
        row1.addWidget(self.btn_up)
        row2.addWidget(self.btn_rot_l); row2.addWidget(self.btn_left); row2.addWidget(self.btn_right); row2.addWidget(self.btn_rot_r)
        row3.addWidget(self.btn_down)
        
        grid_layout.addLayout(row1); grid_layout.addLayout(row2); grid_layout.addLayout(row3)
        control_layout.addLayout(grid_layout)

        self.setup_move_btn(self.btn_up, 0.2, 0.0)
        self.setup_move_btn(self.btn_down, -0.2, 0.0)
        self.setup_move_btn(self.btn_left, 0.0, 0.5)
        self.setup_move_btn(self.btn_right, 0.0, -0.5)
        self.setup_move_btn(self.btn_rot_l, 0.0, 1.0)
        self.setup_move_btn(self.btn_rot_r, 0.0, -1.0)

        main_layout.addLayout(control_layout, stretch=1)
        self.setFocusPolicy(Qt.StrongFocus)
        self.update_button_states()

    def send_parameters(self):
        step = self.slider_step.value()
        inf = self.slider_inflation.value()
        res = self.spin_res.value()
        self.ros_node.push_parameters_to_planner(step, inf, res)

    def setup_move_btn(self, btn, linear, angular):
        btn.pressed.connect(lambda: self.attempt_move(linear, angular))
        btn.released.connect(lambda: self.ros_node.publish_twist(0.0, 0.0))

    def spin_ros(self):
        rclpy.spin_once(self.ros_node, timeout_sec=0.01)

    def set_state(self, new_state):
        self.state = new_state
        if self.state == RobotState.IDLE:
            self.status_label.setText("Būsena: BŪDĖJIMAS (IDLE)")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 18px; color: green;")
        elif self.state == RobotState.PLANNING:
            self.status_label.setText("Būsena: PLANUOJAMA...")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 18px; color: orange;")
        elif self.state == RobotState.CLEANING:
            self.status_label.setText("Būsena: VALOMA!")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 18px; color: red;")
        self.update_button_states()

    def update_button_states(self):
        params_enabled = (self.state == RobotState.IDLE and not self.auto_sequence_active)
        self.slider_step.setEnabled(params_enabled)
        self.slider_inflation.setEnabled(params_enabled)
        self.spin_res.setEnabled(params_enabled)

        if self.auto_sequence_active:
            self.btn_segment.setEnabled(False)
            self.btn_blobs.setEnabled(False)
            self.btn_path.setEnabled(False)
            self.btn_clean.setEnabled(False)
            self.btn_auto.setEnabled(False)
            return

        if self.state == RobotState.IDLE:
            self.btn_segment.setEnabled(True)
            self.btn_blobs.setEnabled(False)
            self.btn_path.setEnabled(False)
            self.btn_clean.setEnabled(False)
            self.btn_auto.setEnabled(True)
            
        elif self.state == RobotState.PLANNING:
            self.btn_segment.setEnabled(False)
            self.btn_auto.setEnabled(False)
            
            has_mask = self.ros_node.latest_mask is not None
            self.btn_blobs.setEnabled(has_mask)
            
            has_blobs = self.ros_node.latest_centroids is not None
            self.btn_path.setEnabled(has_blobs)
            
            has_path = self.ros_node.latest_path is not None
            self.btn_clean.setEnabled(has_path)
            
        elif self.state == RobotState.CLEANING:
            self.btn_segment.setEnabled(False)
            self.btn_blobs.setEnabled(False)
            self.btn_path.setEnabled(False)
            self.btn_clean.setEnabled(False)
            self.btn_auto.setEnabled(False)

    def attempt_move(self, linear, angular):
        if self.state != RobotState.IDLE:
            self.ros_node.get_logger().info("Manual control activated. Canceling current tasks.")
            self.cancel_operations()
        self.ros_node.publish_twist(linear, angular)

    def cancel_operations(self):
        self.auto_sequence_active = False
        self.mask_poll_timer.stop()
        self.ros_node.latest_mask = None
        self.ros_node.latest_inflated_mask = None
        self.ros_node.latest_centroids = None
        self.ros_node.latest_path = None
        self.set_state(RobotState.IDLE)

    def keyPressEvent(self, event):
        if event.isAutoRepeat(): return
        key = event.key()
        if key == Qt.Key_W: self.attempt_move(0.2, 0.0)
        elif key == Qt.Key_S: self.attempt_move(-0.2, 0.0)
        elif key == Qt.Key_A: self.attempt_move(0.0, 0.5)
        elif key == Qt.Key_D: self.attempt_move(0.0, -0.5)
        elif key == Qt.Key_Q: self.attempt_move(0.0, 1.0)
        elif key == Qt.Key_E: self.attempt_move(0.0, -1.0)
            
    def keyReleaseEvent(self, event):
        if event.isAutoRepeat(): return
        if event.key() in [Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D, Qt.Key_Q, Qt.Key_E]:
            self.ros_node.publish_twist(0.0, 0.0)

    def cmd_segment(self):
        if self.state != RobotState.IDLE and not self.auto_sequence_active: return
        self.set_state(RobotState.PLANNING)
        self.ros_node.latest_mask = None
        self.ros_node.latest_inflated_mask = None
        self.ros_node.latest_centroids = None
        self.ros_node.latest_path = None
        
        self.ros_node.get_logger().info("Requesting segmentation...")
        future = self.ros_node.seg_client.call_async(Trigger.Request())
        future.add_done_callback(self._cb_segment_done)

    def _cb_segment_done(self, future):
        try:
            response = future.result()
            if response.success:
                self.mask_check_attempts = 0
                self.mask_poll_timer.start(100) 
            else:
                self.cancel_operations()
        except Exception as e:
            self.ros_node.get_logger().error(f"Segmentation error: {e}")
            self.cancel_operations()

    def verify_segmentation_result(self):
        self.mask_check_attempts += 1
        
        if self.ros_node.latest_mask is not None:
            self.mask_poll_timer.stop()
            mask_cv = self.ros_node.br.imgmsg_to_cv2(self.ros_node.latest_mask, desired_encoding='mono8')
            
            if cv2.countNonZero(mask_cv) == 0:
                self.ros_node.get_logger().info("No filth found.")
                self.status_label.setText("Būsena: NERASTA NEŠVARUMŲ")
                self.status_label.setStyleSheet("font-weight: bold; font-size: 18px; color: #ff9800;")
                QTimer.singleShot(2500, self.cancel_operations)
            else:
                self.ros_node.get_logger().info("Filth found! Continuing planning.")
                if self.auto_sequence_active:
                    self.cmd_extract_blobs()
                    
        elif self.mask_check_attempts > 50:
            self.mask_poll_timer.stop()
            self.ros_node.get_logger().error("Failed to receive mask after segmentation.")
            self.cancel_operations()

    def cmd_extract_blobs(self):
        if self.state != RobotState.PLANNING and not self.auto_sequence_active: return
        self.ros_node.latest_inflated_mask = None
        self.ros_node.latest_centroids = None
        self.ros_node.latest_path = None
        
        self.ros_node.get_logger().info("Analyzing filth areas...")
        future = self.ros_node.blob_client.call_async(Trigger.Request())
        future.add_done_callback(self._cb_blobs_done)

    def _cb_blobs_done(self, future):
        try:
            response = future.result()
            if response.success:
                self.ros_node.get_logger().info(response.message)
                if self.auto_sequence_active:
                    self.cmd_generate_path()
            else:
                self.ros_node.get_logger().error(f"Blob extraction error: {response.message}")
                self.cancel_operations()
        except Exception as e:
            self.ros_node.get_logger().error(f"Error: {e}")
            self.cancel_operations()

    def cmd_generate_path(self):
        if self.state != RobotState.PLANNING and not self.auto_sequence_active: return
        self.ros_node.latest_path = None
        self.ros_node.get_logger().info("Planning trajectory...")
        
        future = self.ros_node.path_client.call_async(Trigger.Request())
        future.add_done_callback(self._cb_path_done)

    def _cb_path_done(self, future):
        try:
            response = future.result()
            if response.success and self.auto_sequence_active and self.state == RobotState.PLANNING:
                self.cmd_clean()
        except Exception as e:
            self.ros_node.get_logger().error(f"Planning error: {e}")
            self.cancel_operations()

    def cmd_clean(self):
        if self.state != RobotState.PLANNING and not self.auto_sequence_active: return
        self.set_state(RobotState.CLEANING)
        self.ros_node.get_logger().info("Starting cleaning...")
        
        future = self.ros_node.clean_client.call_async(Trigger.Request())
        future.add_done_callback(self._cb_clean_done)

    def _cb_clean_done(self, future):
        try:
            future.result()
            if self.state == RobotState.CLEANING:
                self.ros_node.get_logger().info("Cleaning finished!")
                self.status_label.setText("Būsena: VALYMAS BAIGTAS")
                self.status_label.setStyleSheet("font-weight: bold; font-size: 18px; color: green;")
                QTimer.singleShot(2500, self.cancel_operations)
        except Exception as e:
            self.ros_node.get_logger().error(f"Cleaning error: {e}")
            self.cancel_operations()

    def cmd_auto_clean(self):
        if self.state != RobotState.IDLE: return
        self.auto_sequence_active = True
        self.cmd_segment()

    def update_video_feed(self):
        self.update_button_states()
        if self.ros_node.latest_image is None: return

        try:
            display_img = self.ros_node.br.imgmsg_to_cv2(self.ros_node.latest_image, desired_encoding='bgr8')
            resolution = self.ros_node.get_parameter('resolution_m_per_px').value

            if self.ros_node.latest_mask is not None:
                mask = self.ros_node.br.imgmsg_to_cv2(self.ros_node.latest_mask, desired_encoding='mono8')
                overlay = np.zeros_like(display_img)
                overlay[mask > 0] = [0, 0, 255]
                display_img = cv2.addWeighted(display_img, 1.0, overlay, 0.4, 0)

            if self.ros_node.latest_inflated_mask is not None:
                inf_mask = self.ros_node.br.imgmsg_to_cv2(self.ros_node.latest_inflated_mask, desired_encoding='mono8')
                overlay = np.zeros_like(display_img)
                overlay[inf_mask > 0] = [0, 165, 255] 
                display_img = cv2.addWeighted(display_img, 1.0, overlay, 0.3, 0)

            if self.ros_node.latest_path is not None:
                waypoints = []
                for pose_stamped in self.ros_node.latest_path.poses:
                    px = int(pose_stamped.pose.position.x / resolution)
                    py = int(pose_stamped.pose.position.y / resolution)
                    waypoints.append([px, py])

                if len(waypoints) > 1:
                    pts = np.array(waypoints, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_img, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

            if self.ros_node.latest_centroids is not None:
                for i, pose in enumerate(self.ros_node.latest_centroids.poses):
                    px = int(pose.position.x / resolution)
                    py = int(pose.position.y / resolution)
                    cv2.circle(display_img, (px, py), 6, (255, 0, 0), -1)
                    cv2.putText(display_img, str(i + 1), (px + 8, py - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            h, w = display_img.shape[:2]
            fps_text = f"FPS: {int(self.ros_node.camera_fps)}"
            cv2.putText(display_img, fps_text, (10, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        except Exception as e:
            self.ros_node.get_logger().error(f"Rendering error: {e}")

def main(args=None):
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    node = ControlCenterNode()
    gui = ControlCenterGUI(node)
    gui.show()
    sys.exit(app.exec_())
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
