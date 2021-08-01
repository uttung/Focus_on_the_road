import cv2
import csv
import os

Names = [['./training-images','train']]

for name in Names:
	

	FileList = []
	for dirname in os.listdir(name[0])[1:]: # [1:] Excludes .DS_Store from Mac OS
		path = os.path.join(name[0],dirname)
		for filename in os.listdir(path):
			if filename.endswith(".png"):
				FileList.append(os.path.join(name[0],dirname,filename))
