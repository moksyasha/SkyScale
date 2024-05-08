import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QMainWindow, QLabel, QSlider, QProgressBar, QPushButton, QFileDialog, QComboBox

from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
import cv2
import os

import torch
import torchvision
import numpy as np

from models.moksvsr.MoksVSR import MoksVSR, MoksPlus



class VideoUpscaling(QThread):
    frameProcessed = pyqtSignal(np.ndarray)
    procentProcessed = pyqtSignal(np.int8)

    def __init__(self, filePath, shape, reader, frames_lr):
        super().__init__()
        self.filePath = filePath
        self.video_shape = shape
        self.reader = reader
        self.frames_lr = frames_lr

    def inference(self, frames, model, save_path, index_img):
        from torchvision.utils import save_image
        with torch.no_grad():
            outputs = model(frames)
            # save upscaled images to jpg
            outputs = outputs.squeeze()
            img_copy = torch.clone(outputs[0])
            
            # self.frameProcessed.emit(img_copy.permute(1, 2, 0).cpu().numpy()*255.)
            for i in range(outputs.shape[0]):
                save_image(outputs[i], os.path.join(save_path + "temp/", f"{index_img:04d}_MoksPlus.jpg"))
                index_img += 1
        return index_img

    def run(self):
        frames_per_cycle = 10
        index_img = 0
        device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        curr_frames = torch.empty((frames_per_cycle, self.video_shape[0], self.video_shape[1], self.video_shape[2]), dtype=torch.float32, device=device_cuda)
        
        # set up model
        model = MoksPlus()
        checkpoint = torch.load("/home/moksyasha/Projects/SkyScale/MoksVSR/checkpoints/plus_resize64_29.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(device_cuda)

        save_path = '/home/moksyasha/Projects/SkyScale/MoksVSR/results/video/'
        os.makedirs(save_path + "temp/", exist_ok=True)

        while True:
            torch.cuda.empty_cache()
            try:
                frames = 0
                for i in range(frames_per_cycle):
                    curr_frames[i] = (next(self.reader))["data"] / 255.
                    frames += 1
            except Exception as e:
                print(e)
                if frames:
                    curr_frames = curr_frames[:frames].unsqueeze(0).permute(0, 1, 4, 2, 3)
                    index_img = self.inference(curr_frames, model, save_path, index_img)
                break
            else:
                #self.update_frame2(curr_frames[0].permute(1, 0, 2).cpu().numpy())
                img_copy = torch.clone(curr_frames[0])
                self.frameProcessed.emit((img_copy.cpu().numpy()*255.).astype(np.uint8))
                index_img = self.inference(curr_frames.unsqueeze(0).permute(0, 1, 4, 2, 3), model, save_path, index_img)
                print(f"{index_img}/{self.frames_lr} frames upscaled")
                self.procentProcessed.emit(np.ceil(index_img/self.frames_lr*100.).astype(np.int8))
        self.procentProcessed.emit(np.int8(100))
        os.chdir(save_path + "temp/")
        output_name = "ups_" + self.filePath
        os.system(f'ffmpeg -y -r 24 -pattern_type glob -i "*.jpg" -c:v libx264 -movflags +faststart ../{output_name}.mp4')
        #os.system('cd .. && rm -rf ./temp')


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Super Resolution")
        self.setGeometry(100, 100, 1000, 600)

        self.setStyleSheet("background-color: lightGray;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        layout.addWidget(self.slider)
        #self.slider.setTickInterval(1)
        #self.slider.setSingleStep(1)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        #self.slider.setRange(0, 10)
        self.slider.valueChanged.connect(self.updateFrameSlider)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBothSides) 

        self.pixmap_lr = QLabel(self)
        self.pixmap_lr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pixmap_lr.setGeometry(10, 100, 480, 270)

        self.pixmap_hr = QLabel(self)
        self.pixmap_hr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pixmap_hr.setGeometry(500, 100, 480, 270)

        self.file_label = QLabel(self)
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setGeometry(10, 500, 480, 20)
        self.file_label2 = QLabel(self)

        self.file_label2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label2.setGeometry(20, 80, 110, 20)
        self.file_label3 = QLabel(self)
        self.file_label3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label3.setGeometry(550, 80, 120, 20)
        self.file_label2.setText(f"Low Resolution:")
        self.file_label3.setText(f"High Resolution:")

        self.cadr_lr_label = QLabel(self)
        self.cadr_lr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cadr_lr_label.setGeometry(10, 520, 480, 20)

        self.buttonh = QPushButton("?", self)
        self.buttonh.setGeometry(180, 10, 30, 30)
        #self.buttonh.clicked.connect(self.open_file_dialog)

        self.button = QPushButton("Выбрать видео", self)
        self.button.setGeometry(10, 10, 160, 30)
        self.button.clicked.connect(self.open_file_dialog)

        self.button1 = QPushButton("Запустить обработку", self)
        self.button1.setGeometry(350, 550, 160, 30)
        self.button1.clicked.connect(self.upscale)
        # self.button1.setStyleSheet("QPushButton:pressed { background-color: #c0c0c0; border-style: inset; }") 

        self.combo_box = QComboBox(self)
        self.combo_box.setGeometry(10, 550, 160, 30)
        self.combo_box.addItem("MoksVSR")
        self.combo_box.addItem("VRT")
        self.combo_box.addItem("BasicVSR++")

        self.combo_box1 = QComboBox(self)
        self.combo_box1.setGeometry(180, 550, 160, 30)
        self.combo_box1.addItem("60")
        self.combo_box1.addItem("120")

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(550, 550, 160, 30)
        self.video_shape = []
        self.frames_lr = 0
        self.is_opened = 0
        self.memory_frame = []
        self.progress_bar.setValue(100)

    def open_file_dialog(self):
        # dialog = QFileDialog(self)
        # options = dialog.options()
        # file_path, _ = dialog.getOpenFileName(self, "Выбрать видео", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)", options=options)
        file_path = '/media/moksyasha/linux_data/datasets/own/test270x480_24_5sec.mp4'
        if file_path:
            self.is_opened = 1
            self.video_path = file_path
            torchvision.set_video_backend("cuda")
            self.reader = torchvision.io.VideoReader(self.video_path, "video")
            test_frame = (next(self.reader))["data"]
            dur = self.reader.get_metadata()['video']['duration']
            fps = self.reader.get_metadata()['video']['fps']
            self.frames_lr = np.ceil(dur * fps)

            
            # for i in range(100):
            #     test_frame = (next(self.reader))["data"]
            #     if i % 5 == 0:
            #         self.memory_frame.append(test_frame)

            self.cadr_lr_label.setText(f"Кадров в файле: {self.frames_lr}")

            self.filename = os.path.basename(self.video_path)
            self.video_shape = [test_frame.shape[0], test_frame.shape[1], test_frame.shape[2]]
            self.file_label.setText(f"Открыт файл: {self.filename}")
            self.update_frame1(test_frame.squeeze().cpu().numpy())

    def eventFilter(self, source, event):
        if event.type() == event.Enter:
            # показываем увеличенное изображение
            pixmap = source.pixmap()
            pixmap = pixmap.scaled(pixmap.width() * 2, pixmap.height() * 2, Qt.AspectRatioMode.KeepAspectRatio)
            self.imageLabel.setPixmap(pixmap)
        elif event.type() == event.Leave:
            # возвращаем изображение к обычному размеру
            pixmap = source.pixmap()
            pixmap = pixmap.scaled(pixmap.width() // 2, pixmap.height() // 2, Qt.AspectRatioMode.KeepAspectRatio)
            self.imageLabel.setPixmap(pixmap)

        return super().eventFilter(source, event)

    def upscale(self):
        if self.is_opened:
            self.videoProcessor = VideoUpscaling(self.filename, self.video_shape, self.reader, self.frames_lr)
            self.videoProcessor.frameProcessed.connect(self.update_frame2)
            self.videoProcessor.procentProcessed.connect(self.update_progress)
            self.videoProcessor.start()
        return

    def updateFrameSlider(self, frame_number):
        print(frame_number)
        self.update_frame1(self.memory_frame[frame_number].squeeze().cpu().numpy())
        self.update_frame2(self.memory_frame[frame_number].squeeze().cpu().numpy())

    def update_progress(self, value):
        # if value == 100:
        #     self.slider.show()
        self.progress_bar.setValue(value)

    def update_frame1(self, frame):
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.pixmap_lr.setPixmap(pixmap.scaled(self.pixmap_lr.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def update_frame2(self, frame):
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.pixmap_hr.setPixmap(pixmap.scaled(self.pixmap_hr.size(), Qt.AspectRatioMode.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoPlayer()
    window.show()
    sys.exit(app.exec())
