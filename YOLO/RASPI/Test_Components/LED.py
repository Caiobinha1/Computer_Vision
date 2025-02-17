import RPi.GPIO as GPIO
import time

# Define the GPIO pin for the LED
LED_PIN = 8

# Setup
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
GPIO.setup(LED_PIN, GPIO.OUT)

def blink_led(times=5, interval=0.5):
    """Blinks the LED a specified number of times."""
    for _ in range(times):
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(interval)
        GPIO.output(LED_PIN, GPIO.LOW)
        time.sleep(interval)

try:
    print("Blinking LED on GPIO 8...")
    blink_led()
    print("Test completed!")
except KeyboardInterrupt:
    print("Test interrupted by user.")
finally:
    GPIO.cleanup()  # Reset GPIO states
