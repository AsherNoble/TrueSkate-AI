# scripts/control_trueskate.py
from appium import webdriver
from appium.options.ios import XCUITestOptions
import time
import os
from dotenv import load_dotenv

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

    # Example: tap screen
    driver.execute_script('mobile: tap', {'x': 200, 'y': 400})
    print("Tapped screen")

    time.sleep(5)
    driver.quit()