'''
import sys
import cv2
import numpy as np
import pyautogui
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton,
                             QVBoxLayout, QWidget, QHBoxLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWebEngineWidgets import QWebEngineView
from processing_pipeline import process
from game_control import get_config_data

# Load gesture configuration
config = get_config_data()

# Webcam thread for real-time video processing
class WebcamThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_gesture_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.running = True
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                thresh_lower = 60  
                thresh_upper = 200
                try:
                    _, _, _, _, count_defects = process(frame, thresh_lower, thresh_upper)
                    fingers = count_defects + 1
                    self.update_gesture_signal.emit(fingers)
                    if config.get(str(fingers), "None") != "None":
                        pyautogui.press(config[str(fingers)])
                except Exception as e:
                    print("Error in processing:", e)
                    fingers = 0
                self.change_pixmap_signal.emit(frame)

    def stop(self):
        self.running = False
        self.cap.release()
        self.quit()

# Main UI
class DinoGameUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dino Game with Gesture Control")
        self.setGeometry(100, 100, 1500, 700)

        # Main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # === Title Label ===
        self.title_label = QLabel("Dino Game Using Hand Gestures", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.title_label.setStyleSheet("color: #333333; padding: 10px;")
        main_layout.addWidget(self.title_label)

        # === Horizontal Layout for Game & Webcam ===
        game_cam_layout = QHBoxLayout()
        main_layout.addLayout(game_cam_layout)

        # === Dino Game Web View (with better visibility) ===
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("https://chromedino.com/"))
        self.browser.setFixedSize(750, 450)  # Increased height to make game visible
        self.browser.loadFinished.connect(self.adjust_game_view)
        game_cam_layout.addWidget(self.browser)

        # === Webcam Feed ===
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(700, 500)
        game_cam_layout.addWidget(self.video_label)

        # === Gesture Count Label ===
        self.gesture_label = QLabel("Detected Fingers: 0\n\n2 Fingers = Dino on Ground\n1 Finger = Dino Jump", self)
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.gesture_label.setFont(QFont("Arial", 18))
        self.gesture_label.setStyleSheet("color: #000000; padding: 10px;")
        main_layout.addWidget(self.gesture_label)

        # === Button Layout ===
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)

        # Start/Stop Webcam Button
        self.toggle_camera_btn = QPushButton("Stop Webcam")
        self.toggle_camera_btn.setFont(QFont("Arial", 14))
        self.toggle_camera_btn.clicked.connect(self.toggle_camera)
        button_layout.addWidget(self.toggle_camera_btn)

        # Start Game Button
        self.start_game_btn = QPushButton("Start Game")
        self.start_game_btn.setFont(QFont("Arial", 14))
        self.start_game_btn.clicked.connect(self.start_game)
        button_layout.addWidget(self.start_game_btn)

        # Start the webcam automatically
        self.webcam_thread = WebcamThread()
        self.webcam_thread.change_pixmap_signal.connect(self.update_image)
        self.webcam_thread.update_gesture_signal.connect(self.update_gesture)
        self.webcam_thread.start()

    def adjust_game_view(self, ok):
        """ Adjusts the Dino game visibility by zooming in and removing scrollbars. """
        if ok:
            zoom_script = """
            document.body.style.zoom = "130%";  // Zoom in to make Dino more visible
            document.documentElement.style.overflow = 'hidden';  // Hide scrolling
            """
            self.browser.page().runJavaScript(zoom_script)

    def toggle_camera(self):
        if self.webcam_thread.isRunning():
            self.webcam_thread.stop()
            self.toggle_camera_btn.setText("Start Webcam")
        else:
            self.webcam_thread = WebcamThread()
            self.webcam_thread.change_pixmap_signal.connect(self.update_image)
            self.webcam_thread.update_gesture_signal.connect(self.update_gesture)
            self.webcam_thread.start()
            self.toggle_camera_btn.setText("Stop Webcam")


    def start_game(self):
        pyautogui.press("up")

    def update_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def update_gesture(self, fingers):
        self.gesture_label.setText(f"Detected Fingers: {fingers}\n\n2 Fingers = Dino on Ground\n1 Finger = Dino Jump")

    def closeEvent(self, event):
        self.webcam_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DinoGameUI()
    window.show()
    sys.exit(app.exec_())
'''
import pyautogui
pyautogui.FAILSAFE = False
import sys
import cv2
import numpy as np
import pyautogui
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWebEngineWidgets import QWebEngineView
from processing_pipeline import process
from game_control import get_config_data

# Load gesture configuration
config = get_config_data()

# Webcam thread for real-time gesture detection
class WebcamThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_gesture_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.running = True
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                try:
                    roi, drawing, thresh_img, crop_img, fingers = process(frame, 60, 200)

                    if fingers is None:  
                        fingers = 0  # Prevents NoneType errors

                    self.update_gesture_signal.emit(fingers)

                    if str(fingers) in config:
                        pyautogui.press(config[str(fingers)])

                except Exception as e:
                    print("Error in processing:", e)
                    fingers = 0

                self.change_pixmap_signal.emit(frame)

    def stop(self):
        self.running = False
        self.cap.release()
        self.quit()

# UI Class for Dino Game
class DinoGameUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dino Game with Gesture Control")
        self.setGeometry(100, 100, 1500, 700)

        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Title Label
        self.title_label = QLabel("Dino Game Using Hand Gestures", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.title_label.setStyleSheet("color: #333333; padding: 10px;")
        main_layout.addWidget(self.title_label)

        # Layout for Game & Webcam
        game_cam_layout = QHBoxLayout()
        main_layout.addLayout(game_cam_layout)

        # Dino Game Web View
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("https://chromedino.com/"))
        self.browser.setFixedSize(750, 450)
        self.browser.loadFinished.connect(self.adjust_game_view)
        game_cam_layout.addWidget(self.browser)

        # Webcam Feed
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(700, 500)
        game_cam_layout.addWidget(self.video_label)

        # Gesture Label
        self.gesture_label = QLabel("Detected Fingers: 0", self)
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.gesture_label.setFont(QFont("Arial", 18))
        self.gesture_label.setStyleSheet("color: #000000; padding: 10px;")
        main_layout.addWidget(self.gesture_label)

        # Start Game Button
        self.start_game_btn = QPushButton("Start Game")
        self.start_game_btn.setFont(QFont("Arial", 14))
        self.start_game_btn.clicked.connect(self.start_game)
        main_layout.addWidget(self.start_game_btn)

        # Start Webcam
        self.webcam_thread = WebcamThread()
        self.webcam_thread.change_pixmap_signal.connect(self.update_image)
        self.webcam_thread.update_gesture_signal.connect(self.update_gesture)
        self.webcam_thread.start()

    def adjust_game_view(self, _):
        """ Adjusts the Dino game visibility by zooming in and removing scrollbars. """
        js_script = """
        (function() {
            try {
                document.body.style.zoom = '130%';  // Zoom in to make Dino more visible
                document.documentElement.style.overflow = 'hidden';  // Hide scrolling
                let style = document.createElement('style');
                style.innerHTML = '::-webkit-scrollbar { display: none; }';
                document.head.appendChild(style);
            } catch (error) {
                console.log('JS Error:', error);
            }
        })();
        """
        self.browser.page().runJavaScript(js_script)


    def start_game(self):
        pyautogui.press("up")

    def update_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_image.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def update_gesture(self, fingers):
        self.gesture_label.setText(f"Detected Fingers: {fingers}")

    def closeEvent(self, event):
        self.webcam_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DinoGameUI()
    window.show()
    sys.exit(app.exec_())
