import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QProgressBar, QPushButton, QFileDialog, QComboBox

from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer
import cv2
import os

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Super Resolution")
        self.setGeometry(100, 100, 1000, 600)

        self.setStyleSheet("background-color: lightGray;")

        self.ups_label = QLabel(self)
        self.ups_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ups_label.setGeometry(520, 50, 400, 20)
        self.ups_label.setText(f"Выберете модель для апскейла: ")

        self.ups_label = QLabel(self)
        self.ups_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ups_label.setGeometry(495, 160, 400, 20)
        self.ups_label.setText(f"Выберете желаемый FPS: ")

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setGeometry(10, 50, 480, 270)

        self.file_label = QLabel(self)
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setGeometry(10, 330, 480, 20)

        self.button = QPushButton("Выбрать видео", self)
        self.button.setGeometry(10, 10, 160, 30)
        self.button.clicked.connect(self.open_file_dialog)

        self.button1 = QPushButton("Запустить обработку", self)
        self.button1.setGeometry(600, 310, 160, 30)
        self.button1.clicked.connect(self.open_file_dialog)
        # self.button1.setStyleSheet("QPushButton:pressed { background-color: #c0c0c0; border-style: inset; }") 

        self.combo_box = QComboBox(self)
        self.combo_box.setGeometry(600, 75, 160, 30)
        self.combo_box.addItem("BasicVSR++")
        self.combo_box.addItem("VRT")
        self.combo_box.addItem("ESRGAN")

        self.combo_box1 = QComboBox(self)
        self.combo_box1.setGeometry(600, 185, 160, 30)
        self.combo_box1.addItem("60")
        self.combo_box1.addItem("120")

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(800, 310, 160, 30)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.video_path = ""
        self.cap = None

        self.timer1 = QTimer()
        self.timer1.timeout.connect(self.update_progress)

    def open_file_dialog(self):
        self.timer1.start(100)
        dialog = QFileDialog(self)
        options = dialog.options()
        file_path, _ = dialog.getOpenFileName(self, "Выбрать видео", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)", options=options)
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)
            filename = os.path.basename(self.video_path)
            self.file_label.setText(f"Открыт файл: {filename}")
            self.start_video()

    def start_video(self):
        self.timer.start(100)
        if self.cap.isOpened():
            self.timer.start(1000 // self.get_fps())

    def get_fps(self):
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    def update_progress(self):
        value = self.progress_bar.value()
        if value < 100:
            value += 1
            self.progress_bar.setValue(value)
        else:
            self.timer1.stop()  # Останавливаем таймер, когда прогресс достиг максимального значения
            self.button.setEnabled(True)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoPlayer()
    window.show()
    sys.exit(app.exec())
