# scripts/control_trueskate.py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from appium import webdriver
from appium.options.ios import XCUITestOptions
import time
from dotenv import load_dotenv

from src.trueskate_ai.sim.touch_actions import tap, swipe, long_press, curved_drag, reset_position

load_dotenv()


def connect_and_launch():
    udid = os.environ.get("IPHONE_UDID")
    if not udid:
        raise RuntimeError("IPHONE_UDID not set. Copy .env.example to .env and fill in your device UDID.")

    options = XCUITestOptions()
    options.platform_name = 'iOS'
    options.automation_name = 'XCUITest'
    options.bundle_id = 'com.trueaxis.skate'
    options.udid = udid
    options.wda_local_port = 8100
    options.use_prebuilt_wda = True
    options.skip_log_capture = True

    driver = webdriver.Remote('http://127.0.0.1:4723', options=options)
    print("True Skate launched")
    return driver


if __name__ == "__main__":
    driver = connect_and_launch()

    time.sleep(1.5)
    long_press(driver, 350, 752, duration=0.5)

    time.sleep(3)

    reset_position(driver)

    # curved_drag(driver, [(30, 500), (60, 400), (90, 300), (120, 200), (300, 150)], total_duration=2)
    # print("Long press")

    # swipe(driver, 100, 300, 100, 600, duration=0.001)
    print("Executed touch actions")

    # time.sleep(5)
    # driver.quit()