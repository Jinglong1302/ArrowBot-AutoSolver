import subprocess
import cv2
import numpy as np

def capture_screenshot():
    cmd = ["adb", "exec-out", "screencap", "-p"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ADB Error: {err.decode('utf-8')}")
    img_arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

# 1. Capture screen directly from the connected phone
print("[*] Capturing screen via ADB...")
img = capture_screenshot()
cv2.imwrite("screen_victory.png", img)
print("[+] Captured and saved as 'screen_victory.png'.")

# 2. Configure a resizable window so high-res phone screens fit on your PC monitor
window_name = "Select Next/Continue Button (Drag box -> Press ENTER/SPACE)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Determine an appropriate display size (max 900px tall) to prevent overflowing the monitor
h, w, _ = img.shape
scale = min(1.0, 900.0 / h)
disp_w = int(w * scale)
disp_h = int(h * scale)
cv2.resizeWindow(window_name, disp_w, disp_h)

# 3. Interactive ROI selection
# cv2.selectROI returns coordinates relative to the original image dimensions
r = cv2.selectROI(window_name, img, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Check if a valid selection was made
if r[2] > 0 and r[3] > 0:
    x, y, crop_w, crop_h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
    btn_crop = img[y : y + crop_h, x : x + crop_w]
    cv2.imwrite("btn_next.png", btn_crop)
    print(f"[✓] Saved 'btn_next.png' with dimensions {crop_w}x{crop_h}px.")
else:
    print("[!] Selection canceled or empty. No template saved.")