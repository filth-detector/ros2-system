#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import time
import math
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseArray, Point, PoseStamped
from std_msgs.msg import Empty, Float64
from cv_bridge import CvBridge

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFrame, QSlider, 
                             QFormLayout, QCheckBox, QDoubleSpinBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from enum import Enum

from std_srvs.srv import Trigger
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter as MsgParameter, ParameterType, ParameterValue
from rclpy.parameter import Parameter as RclpyParameter

class RobotState(Enum):
    IDLE = 0
    PLANNING = 1
    CLEANING = 2
    HOLDING = 3 

class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gui = None

    def mousePressEvent(self, event):
        if self.gui and getattr(self.gui, 'aim_mode_active', False):
            if self.pixmap() is None:
                return
            
            l_w, l_h = self.width(), self.height()
            p_w, p_h = self.pixmap().width(), self.pixmap().height()
            
            x_offset = (l_w - p_w) / 2.0
            y_offset = (l_h - p_h) / 2.0
            
            x_px = event.pos().x() - x_offset
            y_px = event.pos().y() - y_offset
            
            if 0 <= x_px <= p_w and 0 <= y_px <= p_h:
                if self.gui.ros_node.latest_image:
                    orig_w = self.gui.ros_node.latest_image.width
                    orig_h = self.gui.ros_node.latest_image.height
                    
                    final_x = int((x_px / p_w) * orig_w)
                    final_y = int((y_px / p_h) * orig_h)
                    
                    self.gui.execute_aim_and_hold(final_x, final_y)
                    
        super().mousePressEvent(event)


class VideoProcessorThread(QThread):
    """Handles all heavy OpenCV math off the main GUI thread."""
    change_pixmap_signal = pyqtSignal(QImage, int) 

    def __init__(self, ros_node, gui):
        super().__init__()
        self.ros_node = ros_node
        self.gui = gui
        self.running = True
        self.last_processed_frame = -1

    def run(self):
        while self.running:
            if self.ros_node.latest_image and self.ros_node.frame_counter != self.last_processed_frame:
                self.last_processed_frame = self.ros_node.frame_counter
                
                try:
                    display_img = self.ros_node.br.imgmsg_to_cv2(self.ros_node.latest_image, desired_encoding='bgr8')

                    if self.ros_node.latest_mask is not None:
                        mask = self.ros_node.br.imgmsg_to_cv2(self.ros_node.latest_mask, desired_encoding='mono8')
                        m_idx = mask > 0
                        display_img[m_idx] = display_img[m_idx] * 0.6 + np.array([0, 0, 255]) * 0.4 

                    if self.ros_node.latest_inflated_mask is not None:
                        inf_mask = self.ros_node.br.imgmsg_to_cv2(self.ros_node.latest_inflated_mask, desired_encoding='mono8')
                        inf_idx = (inf_mask > 0) & (mask == 0) if self.ros_node.latest_mask is not None else (inf_mask > 0)
                        display_img[inf_idx] = display_img[inf_idx] * 0.7 + np.array([0, 165, 255]) * 0.3 

                    if self.ros_node.latest_path is not None:
                        waypoints = [[int(p.pose.position.x), int(p.pose.position.y)] for p in self.ros_node.latest_path.poses]
                        if len(waypoints) > 1:
                            pts = np.array(waypoints, np.int32).reshape((-1, 1, 2))
                            cv2.polylines(display_img, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

                    if self.ros_node.latest_centroids is not None:
                        for i, pose in enumerate(self.ros_node.latest_centroids.poses):
                            px, py = int(pose.position.x), int(pose.position.y)
                            cv2.circle(display_img, (px, py), 6, (255, 0, 0), -1)
                            cv2.putText(display_img, str(i + 1), (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    if self.ros_node.current_aim_px is not None and self.gui.state in [RobotState.CLEANING, RobotState.HOLDING]:
                        cx, cy = self.ros_node.current_aim_px
                        if self.ros_node.previous_aim_px is not None:
                            px, py = self.ros_node.previous_aim_px
                            cv2.line(display_img, (px, py), (cx, cy), (0, 255, 255), 4) 

                        cv2.line(display_img, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 2)
                        cv2.line(display_img, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 2)
                        cv2.circle(display_img, (cx, cy), 4, (0, 0, 255), -1)

                    if self.gui.cb_show_center.isChecked():
                        h_img, w_img = display_img.shape[:2]
                        center_x, center_y = int(w_img / 2), int(h_img / 2)
                        cv2.line(display_img, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), 1)
                        cv2.line(display_img, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), 1)
                        cv2.circle(display_img, (center_x, center_y), 2, (0, 255, 0), -1)

                    h, w = display_img.shape[:2]
                    fps_val = int(self.ros_node.camera_fps)
                    cv2.putText(display_img, f"FPS: {fps_val}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                    self.change_pixmap_signal.emit(qt_image, fps_val)

                except Exception as e:
                    self.ros_node.get_logger().error(f"Video thread error: {e}")
                    
            time.sleep(0.005) 

    def stop(self):
        self.running = False
        self.wait()


class ControlCenterNode(Node):
    def __init__(self):
        super().__init__('control_center_node')
        self.br = CvBridge()
        
        self.latest_image = None
        self.frame_counter = 0 
        
        self.latest_mask = None
        self.latest_inflated_mask = None
        self.latest_centroids = None
        self.latest_path = None
        
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        
        self.yaw_joint_name = self.declare_parameter('yaw_joint_name', 'cleaning_head_yaw_joint').value
        self.pitch_joint_name = self.declare_parameter('pitch_joint_name', 'cleaning_head_pitch_joint').value
        
        self.current_aim_px = None  
        self.previous_aim_px = (320, 240) 
        
        self.camera_fps = 0.0
        self.prev_frame_time = time.time()
        self.cleaning_finished_flag = False
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/cleaning_path', 10)
        self.yaw_pub = self.create_publisher(Float64, '/cleaning_head_yaw_target_cmd', 10)
        self.pitch_pub = self.create_publisher(Float64, '/cleaning_head_pitch_target_cmd', 10)
        
        self.create_subscription(Image, '/camera/image_raw', self.image_cb, 10)
        self.create_subscription(Image, '/segmentation_mask', self.mask_cb, 10)
        self.create_subscription(Image, '/inflated_mask', self.inflated_cb, 10)
        self.create_subscription(PoseArray, '/filth_centroids', self.centroids_cb, 10)
        self.create_subscription(Path, '/cleaning_path', self.path_cb, 10)
        self.create_subscription(Point, '/current_aim_pixel', self.aim_cb, 10)  
        self.create_subscription(Empty, '/cleaning_done', self.done_cb, 10) 
        
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(Float64, '/cleaning_head_yaw_target_cmd', self.tgt_yaw_cb, 10)
        self.create_subscription(Float64, '/cleaning_head_pitch_target_cmd', self.tgt_pitch_cb, 10)
        
        self.seg_client = self.create_client(Trigger, 'trigger_segmentation')
        self.blob_client = self.create_client(Trigger, 'extract_blobs')
        self.path_client = self.create_client(Trigger, 'generate_trajectory')
        self.clean_client = self.create_client(Trigger, 'start_cleaning')
        self.cancel_client = self.create_client(Trigger, 'cancel_cleaning')  
        self.aim_hold_client = self.create_client(Trigger, 'aim_and_hold')
        
        self.planner_param_client = self.create_client(SetParameters, '/path_planner/set_parameters')
        self.controller_param_client = self.create_client(SetParameters, '/head_controller/set_parameters')

    def image_cb(self, msg): 
        self.latest_image = msg
        self.frame_counter += 1
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
    
    def aim_cb(self, msg):
        new_px = (int(msg.x), int(msg.y))
        if self.current_aim_px != new_px:
            if self.current_aim_px is not None:
                self.previous_aim_px = self.current_aim_px
            self.current_aim_px = new_px
        
    def done_cb(self, msg): self.cleaning_finished_flag = True

    def joint_cb(self, msg):
        try:
            if self.yaw_joint_name in msg.name:
                self.current_yaw = msg.position[msg.name.index(self.yaw_joint_name)]
            if self.pitch_joint_name in msg.name:
                self.current_pitch = msg.position[msg.name.index(self.pitch_joint_name)]
        except ValueError: pass 

    def tgt_yaw_cb(self, msg): self.target_yaw = msg.data
    def tgt_pitch_cb(self, msg): self.target_pitch = msg.data

    def publish_twist(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x, msg.angular.z = float(linear_x), float(angular_z)
        self.cmd_pub.publish(msg)

    def push_parameters_to_planner(self, step_size, inflation):
        if self.planner_param_client.wait_for_service(timeout_sec=0.2):
            req = SetParameters.Request()
            req.parameters = [
                MsgParameter(name='step_size_px', value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=int(step_size))),
                MsgParameter(name='inflation_radius_px', value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=int(inflation)))
            ]
            future = self.planner_param_client.call_async(req)
            future.add_done_callback(lambda f: self.get_logger().warn("Planner params rejected.") if not all(r.successful for r in f.result().results) else None)

    def push_parameters_to_controller(self, offset_y, offset_z, distance):
        if self.controller_param_client.wait_for_service(timeout_sec=0.2):
            req = SetParameters.Request()
            req.parameters = [
                MsgParameter(name='cam_offset_y', value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(offset_y))),
                MsgParameter(name='cam_offset_z', value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(offset_z))),
                MsgParameter(name='target_distance_m', value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(distance)))
            ]
            future = self.controller_param_client.call_async(req)
            future.add_done_callback(lambda f: self.get_logger().warn("Controller params rejected.") if not all(r.successful for r in f.result().results) else None)

class ControlCenterGUI(QMainWindow):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.state = RobotState.IDLE
        self.auto_sequence_active = False
        self.aim_mode_active = False
        self.holding_aim = False 
        self.mask_check_attempts = 0
        
        self.init_ui()
        
        self.video_thread = VideoProcessorThread(self.ros_node, self)
        self.video_thread.change_pixmap_signal.connect(self.update_video_feed)
        self.video_thread.start()
        
        self.ui_poll_timer = QTimer()
        self.ui_poll_timer.timeout.connect(self.poll_ui_state)
        self.ui_poll_timer.start(100) 
        
        self.mask_poll_timer = QTimer()
        self.mask_poll_timer.timeout.connect(self.verify_segmentation_result)

    def init_ui(self):
        self.setWindowTitle('Valymo Roboto Valdymo Centras')
        self.resize(1150, 800)
        main_widget = QWidget(); self.setCentralWidget(main_widget); main_layout = QHBoxLayout(main_widget)
        
        video_layout = QVBoxLayout()
        self.video_label = VideoLabel()
        self.video_label.gui = self
        self.video_label.setText("Laukiama kameros vaizdo...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        self.video_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_label)
        main_layout.addLayout(video_layout, stretch=2)

        control_layout = QVBoxLayout()
        self.status_label = QLabel("Būsena: BŪDĖJIMAS (IDLE)")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 18px; color: green;")
        control_layout.addWidget(self.status_label)
        
        tel_frame = QFrame()
        tel_frame.setStyleSheet("background-color: #2b2b2b; color: #4CAF50; border-radius: 4px; padding: 4px;")
        tel_layout = QVBoxLayout(tel_frame)
        self.lbl_tel_yaw = QLabel("YAW (Pan):   Cur:   +0.0°  |  Tgt:   +0.0°  |  Δ:   +0.0°")
        self.lbl_tel_pitch = QLabel("PITCH (Tilt): Cur:   +0.0°  |  Tgt:   +0.0°  |  Δ:   +0.0°")
        self.lbl_tel_yaw.setStyleSheet("font-family: monospace; font-size: 13px; font-weight: bold;")
        self.lbl_tel_pitch.setStyleSheet("font-family: monospace; font-size: 13px; font-weight: bold;")
        tel_layout.addWidget(self.lbl_tel_yaw); tel_layout.addWidget(self.lbl_tel_pitch)
        tel_layout.setContentsMargins(5, 5, 5, 5)
        control_layout.addWidget(tel_frame)
        
        self.btn_segment = QPushButton("1. Segmentuoti nešvarumus")
        self.btn_blobs = QPushButton("2. Analizuoti nešvarumų zonas")
        self.btn_path = QPushButton("3. Generuoti valymo trajektoriją")
        self.btn_clean = QPushButton("4. Valyti")
        self.btn_aim = QPushButton("Nutaikyti į tašką (Aim && Hold)"); self.btn_aim.setCheckable(True); self.btn_aim.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_reset_origin = QPushButton("Centruoti kamerą (Reset)")
        self.btn_cancel = QPushButton("5. Atšaukti valymą (Cancel)"); self.btn_cancel.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.btn_auto = QPushButton("Automatinis valymas"); self.btn_auto.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 12px; font-size: 14px;")

        self.btn_segment.clicked.connect(self.cmd_segment)
        self.btn_blobs.clicked.connect(self.cmd_extract_blobs)
        self.btn_path.clicked.connect(self.cmd_generate_path)
        self.btn_clean.clicked.connect(self.cmd_clean)
        self.btn_aim.clicked.connect(self.toggle_aim_mode)
        self.btn_reset_origin.clicked.connect(lambda: self.cancel_operations("UI_RESET_ORIGIN_CLICKED"))
        self.btn_cancel.clicked.connect(lambda: self.cancel_operations("UI_CANCEL_BUTTON_CLICKED")) 
        self.btn_auto.clicked.connect(self.cmd_auto_clean)

        control_layout.addWidget(self.btn_segment); control_layout.addWidget(self.btn_blobs); control_layout.addWidget(self.btn_path); control_layout.addWidget(self.btn_clean)
        aim_row = QHBoxLayout(); aim_row.addWidget(self.btn_aim); aim_row.addWidget(self.btn_reset_origin)
        control_layout.addLayout(aim_row)
        
        self.cb_show_center = QCheckBox("Rodyti centro žymeklį (Center Marker)")
        self.cb_show_center.setChecked(True); self.cb_show_center.setStyleSheet("font-weight: bold; color: #555;")
        control_layout.addWidget(self.cb_show_center)
        control_layout.addWidget(self.btn_cancel) 
        
        line1 = QFrame(); line1.setFrameShape(QFrame.HLine); control_layout.addWidget(line1)
        control_layout.addWidget(self.btn_auto); control_layout.addSpacing(10)

        param_label = QLabel("Konfigūracija (Parametrai)"); param_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        control_layout.addWidget(param_label)

        param_form = QFormLayout()
        
        self.slider_step = QSlider(Qt.Horizontal); self.slider_step.setRange(5, 100); self.slider_step.setValue(20)
        self.slider_inflation = QSlider(Qt.Horizontal); self.slider_inflation.setRange(0, 50); self.slider_inflation.setValue(10)
        self.slider_step.valueChanged.connect(lambda v: self.lbl_step_val.setText(f"{v} px")); self.slider_step.sliderReleased.connect(self.send_planner_parameters)
        self.slider_inflation.valueChanged.connect(lambda v: self.lbl_inf_val.setText(f"{v} px")); self.slider_inflation.sliderReleased.connect(self.send_planner_parameters)

        self.lbl_step_val = QLabel("20 px"); step_row = QHBoxLayout(); step_row.addWidget(self.slider_step); step_row.addWidget(self.lbl_step_val)
        param_form.addRow("Žingsnis (Tankumas):", step_row)
        self.lbl_inf_val = QLabel("10 px"); inf_row = QHBoxLayout(); inf_row.addWidget(self.slider_inflation); inf_row.addWidget(self.lbl_inf_val)
        param_form.addRow("Išplėtimo spindulys:", inf_row)
        
        self.spin_offset_y = QDoubleSpinBox(); self.spin_offset_y.setRange(-5.0, 5.0); self.spin_offset_y.setSingleStep(0.05); self.spin_offset_y.setValue(0.0)
        self.spin_offset_z = QDoubleSpinBox(); self.spin_offset_z.setRange(-5.0, 5.0); self.spin_offset_z.setSingleStep(0.05); self.spin_offset_z.setValue(1.1)
        self.spin_dist = QDoubleSpinBox(); self.spin_dist.setRange(-1.0, 50.0); self.spin_dist.setSingleStep(0.1); self.spin_dist.setValue(-1.0)
        
        self.spin_offset_y.editingFinished.connect(self.send_controller_parameters)
        self.spin_offset_z.editingFinished.connect(self.send_controller_parameters)
        self.spin_dist.editingFinished.connect(self.send_controller_parameters)
        
        param_form.addRow("Cam Offset Y (m):", self.spin_offset_y)
        param_form.addRow("Cam Offset Z (m):", self.spin_offset_z)
        param_form.addRow("Target Dist (m, -1=Inf):", self.spin_dist)
        
        control_layout.addLayout(param_form)
        
        line2 = QFrame(); line2.setFrameShape(QFrame.HLine); control_layout.addWidget(line2)
        move_label = QLabel("Rankinis valdymas (W, A, S, D, Q, E)"); move_label.setStyleSheet("font-weight: bold;"); control_layout.addWidget(move_label)

        grid_layout = QVBoxLayout()
        row1, row2, row3 = QHBoxLayout(), QHBoxLayout(), QHBoxLayout()
        self.btn_up = QPushButton("Pirmyn (W)"); self.btn_rot_l = QPushButton("Sukti Kairėn (Q)"); self.btn_left = QPushButton("Kairėn (A)")
        self.btn_right = QPushButton("Dešinėn (D)"); self.btn_rot_r = QPushButton("Sukti Dešinėn (E)"); self.btn_down = QPushButton("Atgal (S)")
        row1.addWidget(self.btn_up); row2.addWidget(self.btn_rot_l); row2.addWidget(self.btn_left); row2.addWidget(self.btn_right); row2.addWidget(self.btn_rot_r); row3.addWidget(self.btn_down)
        grid_layout.addLayout(row1); grid_layout.addLayout(row2); grid_layout.addLayout(row3)
        control_layout.addLayout(grid_layout)

        self.btn_up.pressed.connect(lambda: self.attempt_move(0.2, 0.0, "KEY_W")); self.btn_down.pressed.connect(lambda: self.attempt_move(-0.2, 0.0, "KEY_S"))
        self.btn_left.pressed.connect(lambda: self.attempt_move(0.0, 0.5, "KEY_A")); self.btn_right.pressed.connect(lambda: self.attempt_move(0.0, -0.5, "KEY_D"))
        self.btn_rot_l.pressed.connect(lambda: self.attempt_move(0.0, 1.0, "KEY_Q")); self.btn_rot_r.pressed.connect(lambda: self.attempt_move(0.0, -1.0, "KEY_E"))
        for btn in [self.btn_up, self.btn_down, self.btn_left, self.btn_right, self.btn_rot_l, self.btn_rot_r]: btn.released.connect(lambda: self.ros_node.publish_twist(0.0, 0.0))

        main_layout.addLayout(control_layout, stretch=1)
        self.setFocusPolicy(Qt.StrongFocus)
        self.update_button_states()

    def update_video_feed(self, qt_image, fps):
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def poll_ui_state(self):
        if self.ros_node.cleaning_finished_flag:
            self.ros_node.cleaning_finished_flag = False
            self.handle_cleaning_done()
            
        c_yaw = math.degrees(self.ros_node.current_yaw)
        t_yaw = math.degrees(self.ros_node.target_yaw)
        d_yaw = t_yaw - c_yaw
        
        c_pitch = math.degrees(self.ros_node.current_pitch)
        t_pitch = math.degrees(self.ros_node.target_pitch)
        d_pitch = t_pitch - c_pitch
        
        self.lbl_tel_yaw.setText(f"YAW (Pan):   Cur: {c_yaw:>+6.1f}°  |  Tgt: {t_yaw:>+6.1f}°  |  Δ: {d_yaw:>+6.1f}°")
        self.lbl_tel_pitch.setText(f"PITCH (Tilt): Cur: {c_pitch:>+6.1f}°  |  Tgt: {t_pitch:>+6.1f}°  |  Δ: {d_pitch:>+6.1f}°")
        self.update_button_states()

    def send_planner_parameters(self):
        self.ros_node.push_parameters_to_planner(self.slider_step.value(), self.slider_inflation.value())

    def send_controller_parameters(self):
        self.ros_node.push_parameters_to_controller(
            self.spin_offset_y.value(), 
            self.spin_offset_z.value(),
            self.spin_dist.value()
        )

    def set_state(self, new_state):
        self.state = new_state
        if self.state == RobotState.IDLE:
            self.status_label.setText("Būsena: BŪDĖJIMAS (IDLE)"); self.status_label.setStyleSheet("font-weight: bold; font-size: 18px; color: green;")
        elif self.state == RobotState.PLANNING:
            self.status_label.setText("Būsena: PLANUOJAMA..."); self.status_label.setStyleSheet("font-weight: bold; font-size: 18px; color: orange;")
        elif self.state == RobotState.CLEANING:
            self.status_label.setText("Būsena: VYKDOMA..."); self.status_label.setStyleSheet("font-weight: bold; font-size: 18px; color: blue;")
        elif self.state == RobotState.HOLDING:
            self.status_label.setText("Būsena: NUTAIKYTA (Hold)"); self.status_label.setStyleSheet("font-weight: bold; font-size: 18px; color: purple;")
        self.update_button_states()

    def update_button_states(self):
        params_enabled = (self.state in [RobotState.IDLE, RobotState.HOLDING] and not self.auto_sequence_active)
        self.slider_step.setEnabled(params_enabled)
        self.slider_inflation.setEnabled(params_enabled)
        self.spin_offset_y.setEnabled(params_enabled)
        self.spin_offset_z.setEnabled(params_enabled)
        self.spin_dist.setEnabled(params_enabled)

        if self.auto_sequence_active:
            self.btn_segment.setEnabled(False); self.btn_blobs.setEnabled(False); self.btn_path.setEnabled(False)
            self.btn_clean.setEnabled(False); self.btn_auto.setEnabled(False); self.btn_aim.setEnabled(False)
            self.btn_cancel.setEnabled(True)
            return

        if self.state in [RobotState.IDLE, RobotState.HOLDING]:
            self.btn_segment.setEnabled(True); self.btn_blobs.setEnabled(self.ros_node.latest_mask is not None)
            self.btn_path.setEnabled(self.ros_node.latest_centroids is not None); self.btn_clean.setEnabled(self.ros_node.latest_path is not None)
            self.btn_auto.setEnabled(True); self.btn_aim.setEnabled(True); self.btn_cancel.setEnabled(self.state == RobotState.HOLDING)
        elif self.state == RobotState.PLANNING:
            self.btn_segment.setEnabled(False); self.btn_auto.setEnabled(False); self.btn_aim.setEnabled(False)
            self.btn_cancel.setEnabled(True); self.btn_blobs.setEnabled(self.ros_node.latest_mask is not None)
            self.btn_path.setEnabled(self.ros_node.latest_centroids is not None); self.btn_clean.setEnabled(self.ros_node.latest_path is not None)
        elif self.state == RobotState.CLEANING:
            self.btn_segment.setEnabled(False); self.btn_blobs.setEnabled(False); self.btn_path.setEnabled(False)
            self.btn_clean.setEnabled(False); self.btn_auto.setEnabled(False); self.btn_aim.setEnabled(False); self.btn_cancel.setEnabled(True)

    def check_and_clear_hold(self, override_reason):
        if self.state == RobotState.HOLDING:
            self.ros_node.get_logger().info(f"Hold overridden by {override_reason}.")
            self.cancel_operations(f"HOLD_OVERRIDE_{override_reason}")

    def toggle_aim_mode(self, checked):
        self.check_and_clear_hold("TOGGLE_AIM_MODE")
        self.aim_mode_active = checked
        if checked:
            self.btn_aim.setStyleSheet("background-color: #ffeb3b; color: black; font-weight: bold;"); self.set_state(RobotState.IDLE)
        else:
            self.btn_aim.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")

    def execute_aim_and_hold(self, u, v):
        self.check_and_clear_hold("EXECUTE_NEW_AIM")
        self.btn_aim.setChecked(False); self.toggle_aim_mode(False)
        self.holding_aim = True
        self.ros_node.previous_aim_px = (320, 240) 
        
        path_msg = Path()
        pose = PoseStamped(); pose.pose.position.x = float(u); pose.pose.position.y = float(v); path_msg.poses.append(pose)
        
        self.ros_node.path_pub.publish(path_msg)
        self.set_state(RobotState.CLEANING)
        self.ros_node.get_logger().info(f"Aiming at pixel ({u}, {v})")
        self.ros_node.aim_hold_client.call_async(Trigger.Request()).add_done_callback(self._cb_clean_done)

    def reset_origin(self):
        self.ros_node.yaw_pub.publish(Float64(data=0.0)); self.ros_node.pitch_pub.publish(Float64(data=0.0))
        self.ros_node.get_logger().info("Camera origin reset to 0.0")

    def attempt_move(self, linear, angular, key_name="UNKNOWN_KEY"):
        if self.state != RobotState.IDLE: self.cancel_operations(f"MANUAL_OVERRIDE_{key_name}")
        self.ros_node.publish_twist(linear, angular)

    def cancel_operations(self, reason="UNKNOWN"):
        self.holding_aim = False
        self.ros_node.get_logger().error(f"====== ABORT INITIATED. REASON: {reason} ======")
        self.auto_sequence_active = False
        self.mask_poll_timer.stop()
        
        self.ros_node.latest_mask = None; self.ros_node.latest_inflated_mask = None
        self.ros_node.latest_centroids = None; self.ros_node.latest_path = None
        self.ros_node.current_aim_px = None; self.ros_node.previous_aim_px = (320, 240)
        
        if self.ros_node.cancel_client.wait_for_service(timeout_sec=0.2):
            self.ros_node.cancel_client.call_async(Trigger.Request())
        self.reset_origin()
        self.set_state(RobotState.IDLE)
        
    def handle_cleaning_done(self):
        if self.holding_aim:
            self.ros_node.get_logger().info("Aim reached. Holding position indefinitely.")
            self.set_state(RobotState.HOLDING)
            return

        self.ros_node.get_logger().info("Sequence successfully completed! Returning to IDLE.")
        self.auto_sequence_active = False
        self.ros_node.current_aim_px = None; self.ros_node.previous_aim_px = (320, 240) 
        self.ros_node.latest_mask = None; self.ros_node.latest_inflated_mask = None; self.ros_node.latest_centroids = None; self.ros_node.latest_path = None
        self.reset_origin() 
        self.set_state(RobotState.IDLE)
        self.status_label.setText("Būsena: VEIKSMAS BAIGTAS"); self.status_label.setStyleSheet("font-weight: bold; font-size: 18px; color: green;")
        QTimer.singleShot(3000, lambda: self.set_state(RobotState.IDLE) if self.state == RobotState.IDLE else None)

    def keyPressEvent(self, event):
        if event.isAutoRepeat(): return
        key = event.key()
        if key == Qt.Key_W: self.attempt_move(0.2, 0.0, "KEY_W")
        elif key == Qt.Key_S: self.attempt_move(-0.2, 0.0, "KEY_S")
        elif key == Qt.Key_A: self.attempt_move(0.0, 0.5, "KEY_A")
        elif key == Qt.Key_D: self.attempt_move(0.0, -0.5, "KEY_D")
        elif key == Qt.Key_Q: self.attempt_move(0.0, 1.0, "KEY_Q")
        elif key == Qt.Key_E: self.attempt_move(0.0, -1.0, "KEY_E")
            
    def keyReleaseEvent(self, event):
        if event.isAutoRepeat(): return
        if event.key() in [Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D, Qt.Key_Q, Qt.Key_E]: self.ros_node.publish_twist(0.0, 0.0)

    def cmd_segment(self):
        self.check_and_clear_hold("CMD_SEGMENT")
        if self.state != RobotState.IDLE and not self.auto_sequence_active: return
        self.set_state(RobotState.PLANNING)
        self.ros_node.latest_mask = None; self.ros_node.latest_inflated_mask = None; self.ros_node.latest_centroids = None; self.ros_node.latest_path = None
        self.ros_node.get_logger().info("Requesting segmentation...")
        self.ros_node.seg_client.call_async(Trigger.Request()).add_done_callback(self._cb_segment_done)

    def _cb_segment_done(self, future):
        try:
            response = future.result()
            if response.success:
                self.mask_check_attempts = 0; self.mask_poll_timer.start(100) 
            else: self.cancel_operations(f"SEGMENT_SERVICE_REJECTED: {response.message}")
        except Exception as e:
            self.cancel_operations(f"SEGMENT_SERVICE_EXCEPTION: {str(e)}")

    def verify_segmentation_result(self):
        self.mask_check_attempts += 1
        if self.ros_node.latest_mask is not None:
            self.mask_poll_timer.stop()
            mask_cv = self.ros_node.br.imgmsg_to_cv2(self.ros_node.latest_mask, desired_encoding='mono8')
            if cv2.countNonZero(mask_cv) == 0:
                self.ros_node.get_logger().info("No filth found."); self.status_label.setText("Būsena: NERASTA NEŠVARUMŲ"); self.status_label.setStyleSheet("font-weight: bold; font-size: 18px; color: #ff9800;")
                QTimer.singleShot(2500, lambda: self.cancel_operations("NO_FILTH_FOUND_TIMER"))
            else:
                self.ros_node.get_logger().info("Filth found! Continuing planning.")
                if self.auto_sequence_active: self.cmd_extract_blobs()
        elif self.mask_check_attempts > 50:
            self.mask_poll_timer.stop()
            self.cancel_operations("MASK_POLL_TIMEOUT")

    def cmd_extract_blobs(self):
        self.check_and_clear_hold("CMD_BLOBS")
        if self.state != RobotState.PLANNING and not self.auto_sequence_active: return
        self.ros_node.latest_inflated_mask = None; self.ros_node.latest_centroids = None; self.ros_node.latest_path = None
        self.ros_node.get_logger().info("Analyzing filth areas...")
        self.ros_node.blob_client.call_async(Trigger.Request()).add_done_callback(self._cb_blobs_done)

    def _cb_blobs_done(self, future):
        try:
            response = future.result()
            if response.success:
                self.ros_node.get_logger().info(response.message)
                if self.auto_sequence_active: self.cmd_generate_path()
            else: self.cancel_operations(f"BLOB_SERVICE_REJECTED: {response.message}")
        except Exception as e: self.cancel_operations(f"BLOB_SERVICE_EXCEPTION: {str(e)}")

    def cmd_generate_path(self):
        self.check_and_clear_hold("CMD_PATH")
        if self.state != RobotState.PLANNING and not self.auto_sequence_active: return
        self.ros_node.latest_path = None
        self.ros_node.get_logger().info("Planning trajectory...")
        self.ros_node.path_client.call_async(Trigger.Request()).add_done_callback(self._cb_path_done)

    def _cb_path_done(self, future):
        try:
            response = future.result()
            if response.success and self.auto_sequence_active and self.state == RobotState.PLANNING: self.cmd_clean()
            elif not response.success: self.cancel_operations(f"PATH_SERVICE_REJECTED: {response.message}")
        except Exception as e: self.cancel_operations(f"PATH_SERVICE_EXCEPTION: {str(e)}")

    def cmd_clean(self):
        self.check_and_clear_hold("CMD_CLEAN")
        if self.state != RobotState.PLANNING and not self.auto_sequence_active: return
        self.set_state(RobotState.CLEANING)
        self.ros_node.get_logger().info("Starting cleaning...")
        self.ros_node.clean_client.call_async(Trigger.Request()).add_done_callback(self._cb_clean_done)

    def _cb_clean_done(self, future):
        try:
            response = future.result()
            if self.state == RobotState.CLEANING:
                if response.success: self.ros_node.get_logger().info("Sequence is now executing!")
                else: self.cancel_operations(f"EXECUTION_REJECTED: {response.message}")
        except Exception as e: self.cancel_operations(f"EXECUTION_EXCEPTION: {str(e)}")

    def cmd_auto_clean(self):
        self.check_and_clear_hold("CMD_AUTO_CLEAN")
        if self.state != RobotState.IDLE: return
        self.auto_sequence_active = True
        self.cmd_segment()

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

def main(args=None):
    rclpy.init(args=args)
    node = ControlCenterNode()
    
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)
    gui = ControlCenterGUI(node)
    gui.show()
    sys.exit(app.exec_())
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
