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
cv2.imwrite("large_puzzle.png", img)
print("[+] Captured and saved as 'screen_victory.png'.")