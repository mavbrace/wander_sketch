# Displays SVGs (python3)

import os.path
import numpy as np
from PyQt5 import QtGui, QtSvg, QtWidgets, QtCore
import sys
from random import randint

NAME_OF_GROUP_DIR = "sketch"
NAME_OF_SVG_FILE = "sample"

IMG_WIDTH = 50
IMG_HEIGHT = 50

QT_WIDTH = 1920
QT_HEIGHT = 1080

CLOUD = 30

CYCLE_SECONDS = 30 # 30 seconds to gather info, then store, then repeat

image_prefix = "//Users//mavisbrace//Desktop//understanding_place//svg//"


class CreateMap(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.current_img_counter = 0
        self.initUI()

    def initUI(self):
        self.setGeometry(0,0,QT_WIDTH,QT_HEIGHT)
        self.setGeometry(0,0, QT_WIDTH, QT_HEIGHT)
        self.setWindowTitle('collection');

        # Set window background color
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(200,200,200))
        self.setPalette(p)

        SVG_LIMIT = 200
        all_svgWidgets = []

        # -- not necessary anymore; keeping for fun
        svgWidget = QtSvg.QSvgWidget(image_prefix + "temp_sample.svg",self)
        svg_x = randint(0, QT_WIDTH - svgWidget.width())
        svg_y = randint(0, QT_HEIGHT - svgWidget.height())
        svgWidget.setGeometry(svg_x,svg_y, IMG_WIDTH, IMG_HEIGHT)

        self.show()

        self.timer = QtCore.QTimer()
        self.timer.start(2000)
        self.timer.timeout.connect(self.grabAndShow)


    # w = window
    def grabAndShow(self):
        # 1. check to see if there's a new file
        nextDir = ("%ssketch%04d" %(image_prefix, self.current_img_counter))
        if os.path.isdir(nextDir):
            svg_x = randint(0, QT_WIDTH)
            svg_y = randint(0, QT_HEIGHT)
            num = len(os.listdir(nextDir))
            for name in os.listdir(nextDir):
                file_name = nextDir + "//" + name
                if os.path.isfile(file_name):
                    svgWidget_new = QtSvg.QSvgWidget(file_name, self)
                    svgWidget_new.setGeometry(svg_x + randint(0,CLOUD) - CLOUD/2, svg_y + randint(0,CLOUD) - CLOUD/2, IMG_WIDTH * num, IMG_HEIGHT * num)
                    svgWidget_new.show()
                    print("New SVG added to map: " + name)
                self.show()
            self.current_img_counter = self.current_img_counter + 1



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    the_map = CreateMap()

    sys.exit(app.exec_())
