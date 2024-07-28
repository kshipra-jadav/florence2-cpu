import RPi.GPIO as GPIO
import time

# Pin Definitions
BLUE_BTN1 = 27  # GPIO pin for the left button
PURPLE_BTN2 = 22  # GPIO pin for the right button

def setup_gpio():
    # Pin Setup
    GPIO.setmode(GPIO.BCM)  # Broadcom pin-numbering scheme
    GPIO.setup(BLUE_BTN1, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Set up pin as input with pull-up resistor
    GPIO.setup(PURPLE_BTN2, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Set up pin as input with pull-up resistor


