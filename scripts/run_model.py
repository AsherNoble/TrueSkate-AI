# scripts/control_trueskate.py
from appium import webdriver
from appium.options.ios import XCUITestOptions
import time


def connect_and_launch():
    options = XCUITestOptions()
    options.platform_name = 'iOS'
    options.automation_name = 'XCUITest'
    options.bundle_id = 'com.trueaxis.skate'
    options.udid = '00008030-0015490C0E52202E'
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