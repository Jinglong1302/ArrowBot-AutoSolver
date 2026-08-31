import cv2

# Load your captured screenshot
img = cv2.imread("screen_raw.png")

# Select an arrowhead pointing UP on screen
# A window will open: click and drag a tight box around ONE clean UP-pointing arrowhead, then press ENTER/SPACE
r = cv2.selectROI("Select UP Arrowhead", img, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Crop and save
arrow_crop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
cv2.imwrite("arrow_up_template.png", arrow_crop)
print(f"Saved template with size: {arrow_crop.shape[1]}x{arrow_crop.shape[0]}")