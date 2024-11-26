import cv2
import numpy as np

imgFolder = \
    './Data/layered_dress/silk_chamuse/5_Samba_Dancing/PD30/render_F/'
frame0 = 21
frame1 = 321

img = []
for i in range(frame0, frame1+1):
    imgName = imgFolder + str(i).zfill(7) + '.png'
    img.append(cv2.imread(imgName, cv2.IMREAD_COLOR))

height, width, layers = img[1].shape
print(img[1].shape)

vcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(imgFolder + 'video.mp4', vcc, 30.0, (width, height))

for j in img:
    videoWriter.write(j)

videoWriter.release()
