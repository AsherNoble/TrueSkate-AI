from appium import webdriver
from appium.options.ios import XCUITestOptions
import time

options = XCUITestOptions()
options.platform_name = 'iOS'
options.automation_name = 'XCUITest'
options.bundle_id = 'com.trueaxis.skate'
options.udid = '00008030-0015490C0E52202E'
options.wda_local_port = 8100
options.use_prebuilt_wda = True  # Use the WDA we already launched
options.skip_log_capture = True

driver = webdriver.Remote('http://127.0.0.1:4723', options=options)
print("Connected! True Skate should launch on your phone.")

# Test a tap
driver.execute_script('mobile: tap', {'x': 200, 'y': 400})

time.sleep(5)
driver.quit()