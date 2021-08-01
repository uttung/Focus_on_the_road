from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os


folder = "1/"

for file_name in os.listdir(folder):
    filepath = os.path.join(folder, file_name)
    img = cv2.imread(filepath)
    frame = imutils.resize(img, width=20, height=20)
    cv2.imwrite(filepath,frame)
    
    
