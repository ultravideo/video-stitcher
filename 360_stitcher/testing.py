import glob
import sys
import math
import cv2
import numpy as np

folder = 'static'
if len(sys.argv) > 1:
    folder = sys.argv[1]
skips = 0
if len(sys.argv) > 2:
    skips = int(sys.argv[2])

files = glob.glob(folder + "/*.mp4")
image_size = (288, 162)

images = []
for file_path in files:
    video = cv2.VideoCapture(file_path)
    for _ in range(skips):
        _, frame = video.read()
    images.append(frame)
    #frame = cv2.resize(frame, dsize=image_size)


def project(x, y, f, rot):
    x_ = f * (rot + math.atan(x / f))
    y_ = f * y / math.sqrt(x**2 + f**2)
    return x_, y_


height, width, channels = images[0].shape
new_img = np.zeros(shape=(height, width*5, channels), dtype=np.uint8)
mask = np.zeros(shape=(height, width*5, 1), dtype=np.uint8)
for i, img in enumerate(images):
    rot = 2 * math.pi / 6 * i
    f = 1000
    for (y, x, c), _ in np.ndenumerate(img):
        if c == 0:
            x_, y_ = project(x - width / 2, y - height / 2, f, rot)
            x_, y_ = int(x_), int(y_)
            x_ += width // 2
            y_ += height // 2
            new_img[y_, x_, :] += img[y, x, :]
            mask[y_, x_] += 1

for (y, x, c), val in np.ndenumerate(new_img):
    if mask[y, x]:
        new_img[y, x, c] = val / mask[y, x]

cv2.imwrite("stitch.png", new_img)
cv2.imshow("image", new_img)
cv2.waitKey(0)
