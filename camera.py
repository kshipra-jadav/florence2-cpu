import cv2

def get_capture():
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    return cap

def get_frame():
    cap = get_capture()

    for _ in range(10):
        cap.read()
    
    _, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame
