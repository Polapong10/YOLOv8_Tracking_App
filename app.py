# Import module
import cv2
from pathlib import Path
import numpy as np
import sys
from time import sleep

# Import module of ultralytics
from ultralytics import YOLO

# Import pyqt5 module
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QObject, QThread, pyqtSignal

# Add Path to the root folder of the project
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] # Path to the root folder of the project
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = str(ROOT)

# Initial UI
from ui import Ui_MainWindow
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()

# Initial Counter from OOP module
from customcounter import ObjectCounter
counter = None

# Initial Capture
capVideo = None

# QThread for load frame from video
class UpdateFrameCV2(QObject):
    frameread = pyqtSignal(np.ndarray)
    finished = pyqtSignal()

    def run(self):
        while capVideo.isOpened():
            success, frame = capVideo.read()
            if success:
                self.frameread.emit(frame)
                 # Delay 50 ms for next frame ( 1000 ms / 20 fps = 50 ms )
                sleep(0.05)
            else:
                self.finished.emit()
                break

class mainui(Ui_MainWindow):
    def __init__(self):
        super().setupUi(MainWindow)
        # path video variable
        self.videoDirectory = None
        # path model variable
        self.modelDirectory = None
        # Initial model
        self.model = None
        # Initial button interaction
        self.initialSignal()


    # Initial button interaction
    def initialSignal(self):
        # Load model button
        self.sel_model_bt.clicked.connect(self.getDirmodel)
        # Selected video
        self.sel_vdo_bt.clicked.connect(self.getVideoDir)
        # Start button
        self.start_bt.clicked.connect(self.startCounting)

    # Get the directory of the model
    def getDirmodel(self):
        dialog = QFileDialog()
        pathmodel = dialog.getOpenFileName(MainWindow, 'Selected weight file', ROOT, 'Weight file (*.pt)')[0]
        if pathmodel != '':
            # Store the directory of the model
            self.modelDirectory = pathmodel
        elif self.model_path.text() != '':
            pass       
        else:
            self.modelDirectory = None
            # Set model to None
            self.model = None
        # Set the text of the label
        self.model_path.setText(self.modelDirectory)

    # Get video directory
    def getVideoDir(self):
        # global capVideo
        dialog = QFileDialog()
        pathVideo = dialog.getOpenFileName(MainWindow, 'Selected video file', ROOT, 'Video file (*.mp4 *.mkv *.avi)')[0]
        if pathVideo != '':
            # Store video path
            self.videoDirectory = pathVideo
        elif self.vdo_path.text() != '':
            pass
        else:
            self.videoDirectory = None
            # capVideo = None
        # Set the text of the label
        self.vdo_path.setText(self.videoDirectory)

    # Start counting
    def startCounting(self):
        global capVideo
        global counter 
        counter = ObjectCounter()
        # Setup video
        capVideo = cv2.VideoCapture(self.videoDirectory)
        # Setup Model
        self.model = YOLO(self.modelDirectory)

        # Set QThread
        self.threadUpdateFrame = QThread()
        self.workerThreadUpdate = UpdateFrameCV2()
        self.workerThreadUpdate.moveToThread(self.threadUpdateFrame)
        self.threadUpdateFrame.started.connect(self.workerThreadUpdate.run)
        self.workerThreadUpdate.frameread.connect(self.UpdateCounting)
        self.workerThreadUpdate.finished.connect(self.finishedCounting)
        # Prepair counter arguments
        counter.set_arguments(classes_names = self.model.names,
                                # Point for draw line in counting process
                                reg_pts = [(700, 200), (1400, 200), (50, 200), (500,200)],
                                # Draw track or not
                                draw_tracks = True
                                ), 
        # Start detect      
        self.threadUpdateFrame.start()


    # Update counting
    def UpdateCounting(self, im0):
        # Tracking
        im0_tracks = self.model.track(im0, # Image
                                      persist = True, # persisting tracks between frames
                                      save = False, # save results
                                      conf = 0.3, # confidence threshold
                                      classes = 2, # class filter 2 = car
                                      show = False,
                                      device = '0',
                                      verbose = False # show results plot
                                      )
        # Counting
        im0_res = counter.start_counting(im0, im0_tracks)
        # Convert color
        im0_res = cv2.cvtColor(im0_res, cv2.COLOR_BGR2RGB)
        # Display result with Pixmap
        h, w, ch = im0_res.shape
        bytesPerLine = ch * w
        qImg = QImage(im0_res.data, w, h, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg).scaled(self.video_display.size(), QtCore.Qt.KeepAspectRatio)
        self.video_display.setPixmap(pixmap)
        # Update counting display
        self.UpdateCountingDisplay()

    # Update counting display
    def UpdateCountingDisplay(self):
        # Set display
        self.lcd_ent.display(int(counter.ent_road))
        self.lcd_ext.display(int(counter.ext_road))

    # Finished counting
    def finishedCounting(self):
        global capVideo
        global counter 
        # Release video
        capVideo.release()
        # Reset model 
        self.model = None
        # Stop QThread
        self.threadUpdateFrame.quit()
        self.threadUpdateFrame.deleteLater
        self.workerThreadUpdate.deleteLater
        # Reset counter
        counter.ent_road = counter.ext_road = 0

        # Set text of display result
        self.video_display.setText("Finished counting")
        

if __name__ == '__main__':
    obj = mainui()
    MainWindow.show()
    sys.exit(app.exec_())