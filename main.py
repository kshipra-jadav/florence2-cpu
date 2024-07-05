import time
from pprint import pp
import os

from PIL import Image

from florence_model import run_example, load_model

BASE_PATH = "/home/gsfc-pi/dev/media"

def get_image(filename, resize_factor, show=False):
    image = Image.open(os.path.join(BASE_PATH, filename))

    print(f"Opened Image with dimensions :- {image.size} pixels")

    resized_dimensions = (round(image.size[0] * (resize_factor / 100)), round(image.size[1] * (resize_factor / 100)))

    print(f"Resized it to :- {resized_dimensions} pixels")

    resized_image = image.resize(resized_dimensions, resample=Image.Resampling.LANCZOS)

    if show:
        resized_image.show()

    return resized_image    



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


def run_model(image, instruction, prompt=None):
    model, processor = load_model()

    start = time.perf_counter()
    response = run_example(model=model, image=image, processor=processor, task_prompt=instruction, text_input=prompt)
    pp(response)
    print(f"\nTime taken to perform {instruction} - {time.perf_counter() - start:.2f} seconds")




if __name__ == "__main__":
    image = get_image(filename="wallpaper.jpg", resize_factor=100, show=True)


    benchmark(image)

    # run_model(image, "<pure_text>", "How many people are there in the image?")