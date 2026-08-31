import subprocess
import cv2
import numpy as np

def get_screenshot():
    """Captures screenshot directly from ADB stream into an OpenCV image."""
    cmd = ["adb", "exec-out", "screencap", "-p"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_image, err = process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError(f"ADB Error: {err.decode('utf-8')}")

    # Decode PNG byte array into OpenCV BGR format
    image_array = np.frombuffer(raw_image, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return frame

def send_tap(x: int, y: int):
    """Sends a touch event to the device at (x, y) coordinates."""
    subprocess.run(["adb", "shell", "input", "tap", str(x), str(y)], check=True)

def preprocess_board(frame):
    """Binarizes the game area to isolate arrows and lines."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Invert binary threshold: line paths become 255 (white), background becomes 0 (black)
    # The puzzle lines are dark, so thresholding below ~100 isolates them cleanly
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    return binary

if __name__ == "__main__":
    print("Capturing frame from Honor device...")
    frame = get_screenshot()
    print(f"Captured screen resolution: {frame.shape[1]}x{frame.shape[0]}")

    # Preprocess
    binary_grid = preprocess_board(frame)

    # Save to disk to inspect the output
    cv2.imwrite("screen_raw.png", frame)
    cv2.imwrite("screen_binary.png", binary_grid)
    print("Saved 'screen_raw.png' and 'screen_binary.png' for inspection.")