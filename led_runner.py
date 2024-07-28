import time
import board
import neopixel_spi

from colors import Color

LED_STRIP_LEDS = 16

led_strip = neopixel_spi.NeoPixel_SPI(board.SPI(), LED_STRIP_LEDS, brightness=1.0, auto_write=True)

def operate_led(led_color, sleep_duration):
    led_strip.fill(led_color)

    time.sleep(sleep_duration)

    led_strip.fill(Color.BLACK.value)