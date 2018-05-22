'''
Module for helping calibrate offsets for test material
'''
import cv2
import glob
import sys

folder = 'static'
if len(sys.argv) > 1:
    folder = sys.argv[1]

files = glob.glob(folder + "/*.mp4")
image_size = (288, 162)

# Read all frames to memory
all_frames = []
for file_path in files:
    video = cv2.VideoCapture(file_path)
    frames = []
    fails = 0
    while video.isOpened():
        cap, frame = video.read()
        if not cap:
            fails += 1
            if fails > 10:
                break
            continue
        frame = cv2.resize(frame, dsize=image_size)
        frames.append(frame)
    all_frames.append(frames)

print([len(frames) for frames in all_frames])
offsets = [0 for _ in range(len(all_frames))]

index = 0
step = 1
# Use numbers 1-6 to alter the offsets and 0 to alter current frame.
# When the videos seem to be in sync exit and the offsets are printed.
while True:
    for i, (frames, offset) in enumerate(zip(all_frames, offsets)):
        if index + offset < len(frames):
            cv2.imshow(str(i), frames[index + offset])
    key = cv2.waitKey(1)
    if key >= ord('1') and key < ord('1') + len(all_frames):
        offsets[key - ord('1')] += step
    if key == ord('0'):
        index += step
    if key == ord('q'):
        break
    if key == ord('w'):
        step = 1
    if key == ord('s'):
        step = -1

print(offsets)
