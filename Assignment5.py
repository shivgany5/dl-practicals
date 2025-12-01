# Assignment: Text Identification using OpenCV + Tesseract OCR

import cv2
import pytesseract

# ---------- 1. Set Tesseract Path (Windows only) ----------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- 2. Load Image ----------
img = cv2.imread("sample_text.jpg")   # change image name if needed

if img is None:
    raise FileNotFoundError("Image not found. Check filename/path.")

original = img.copy()

# ---------- 3. Preprocessing (OpenCV) ----------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ---------- 4. Run Tesseract OCR ----------
config = r"--oem 3 --psm 6"  # LSTM OCR model, assume block of text
text = pytesseract.image_to_string(thresh, config=config)

print("\n===== Extracted Text =====")
print(text)

# ---------- 5. Optional: Draw bounding boxes ----------
data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)

for i in range(len(data['text'])):
    if int(data['conf'][i]) > 50:  # filter weak detections
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        cv2.rectangle(original, (x, y), (x + w, y + h), (0,255,0), 2)

# ---------- 6. Show Results ----------
cv2.imshow("OCR Result - Detected Text", original)
cv2.imshow("Processed (Thresholded Image)", thresh)

print("\nPress any key to close windows...")
cv2.waitKey(0)
cv2.destroyAllWindows()
