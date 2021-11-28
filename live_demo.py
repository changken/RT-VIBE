import cv2
from vibe.live.vibe_live import VibeLive

if __name__ == '__main__':
    vibe_live = VibeLive()
    cap = cv2.VideoCapture('sample_video.mp4')
    if not cap.isOpened():
        print("Error opening video stream or file")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            vibe_live(frame)
        else:
            break
