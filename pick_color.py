import cv2
import mss
import numpy as np

hsv_window_name = "HSV Inspector"

# Global variable to hold the image
current_image = None

def on_mouse_move(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        pixel = current_image[y, x]  # row = y, col = x
        bgr = np.uint8([[pixel]])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        print(f"({x}, {y}) → BGR: {tuple(pixel)}, HSV: {tuple(hsv)}")

# Grab one screenshot using mss
with mss.mss() as sct:
    monitor = sct.monitors[1]  # Full screen
    screenshot = sct.grab(monitor)
    img = np.array(screenshot)
    current_image = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# Set up the window and callback
cv2.namedWindow(hsv_window_name)
cv2.setMouseCallback(hsv_window_name, on_mouse_move)

# Display the image and wait for ESC
while True:
    cv2.imshow(hsv_window_name, current_image)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cv2.destroyAllWindows()

# import pyautogui
# screenshot = pyautogui.screenshot()
# screenshot.save("test.png")