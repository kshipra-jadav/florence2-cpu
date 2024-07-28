import time
from pprint import pp
import os

from PIL import Image

from florence_model import run_example, load_model
from camera import get_frame
from button import setup_gpio, BLUE_BTN1, PURPLE_BTN2, GPIO

BASE_PATH = "/home/gsfc-pi/dev/media"
model, processor = load_model()

def get_image(filename=None, resize_factor=100, show=False):
    if filename:
        image = Image.open(os.path.join(BASE_PATH, filename))
    
    if not filename:
        image = Image.fromarray(get_frame())

    print(f"Opened Image with dimensions :- {image.size} pixels")

    if resize_factor != 100:
        resized_dimensions = (round(image.size[0] * (resize_factor / 100)), round(image.size[1] * (resize_factor / 100)))

        print(f"Resized it to :- {resized_dimensions} pixels")

        image = image.resize(resized_dimensions, resample=Image.Resampling.LANCZOS)

    if show:
        image.show()

    return image    



def benchmark(image):
    model, processor = load_model()

    print("running simple caption")

    start = time.perf_counter()
    response = run_example(model=model, processor=processor, task_prompt='<CAPTION>', image=image)
    pp(f"Simple caption response - {response}")
    print(f"Inference time - {time.perf_counter() - start:.2f} seconds")

    print("------")
    print("\n")

    print("running detailed caption")

    start = time.perf_counter()
    response = run_example(model=model, processor=processor, task_prompt='<DETAILED_CAPTION>', image=image)
    pp(f"Detailed caption response - {response}")
    print(f"Inference time - {time.perf_counter() - start:.2f} seconds")

    print("------")
    print("\n")

    print("running more detailed caption")

    start = time.perf_counter()
    response = run_example(model=model, processor=processor, task_prompt='<MORE_DETAILED_CAPTION>', image=image)
    pp(f"More Detailed caption response - {response}")
    print(f"Inference time - {time.perf_counter() - start:.2f} seconds")

    print("------")
    print("\n")


def run_model(image, ocr=False):
    instruction = "<OCR>" if ocr else "<MORE_DETAILED_CAPTION>"

    start = time.perf_counter()
    response = run_example(model=model, image=image, processor=processor, task_prompt=instruction, text_input=None)
    print()
    pp(response[instruction])
    print(f"\nTime taken to perform {instruction} - {time.perf_counter() - start:.2f} seconds")



if __name__ == "__main__":
    setup_gpio()
    try:
        while True:
            if not GPIO.input(BLUE_BTN1):  
                print("scene detection")
                img = get_image(show=True)
                run_model(img)
            if not GPIO.input(PURPLE_BTN2):  
                print("ocr")
                img = get_image(show=True)
                run_model(img, ocr=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.cleanup()
