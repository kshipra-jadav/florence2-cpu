import cv2
import numpy as np
from PIL import Image

from colors import Color
from led_runner import operate_led


def get_capture():
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    return cap

def get_frame():
    cap = get_capture()

    for _ in range(20):
        cap.read()
    
    _, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    isDark = checkImageDark(frame)

    print(f"{isDark=}")

    if isDark:
        operate_led(led_color=Color.WHITE.value, sleep_duration=2)
        _, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        return frame

    return frame

def checkImageDark(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    meanpercent = np.mean(gray) * 100 / 255

    return True if meanpercent < 25 else False

def check_flash():
    cap = get_capture()

    while True:
        _, frame = cap.read()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        isDark = checkImageDark(frame)

        print(f"{isDark=}")

        if isDark:
            operate_led(led_color=Color.WHITE.value, sleep_duration=0)
            _, frame = cap.read()
        
        cv2.imshow("Frame", cv2.resize(frame, (0, 0), fx=0.25, fy=0.25))
        key = cv2.waitKey(1)

        if 0xFF & key == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    frame = get_frame()

    Image.fromarray(frame).show()

    

    


