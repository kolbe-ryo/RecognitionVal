#import recog_value
import cv2

def capture():
    # image_quality = int(input("画質を選択=> 0: 最高, 1: 通常, 2: 低画質 = "))
    img_size = [[3200, 1800], [1920, 1080], [1280, 960]]

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[1][0])
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[1][1])
    ret, img = cap.read()
    # img = cv2.resize(img, (img_size[1][0], img_size[1][1]))
    
    return img