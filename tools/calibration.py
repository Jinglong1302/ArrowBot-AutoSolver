import cv2
import numpy as np

frame = cv2.imread("debug_corridors.png")
h, w, _ = frame.shape
roi = frame[int(h * 0.20):int(h * 0.95), int(w * 0.01):int(w * 0.99)]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

tmpl_img = cv2.imread("arrow_up_template.png", cv2.IMREAD_GRAYSCALE)
_, tmpl = cv2.threshold(tmpl_img, 80, 255, cv2.THRESH_BINARY_INV)

res = cv2.matchTemplate(binary, tmpl, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print(f"Top match score at default scale: {max_val:.3f}")