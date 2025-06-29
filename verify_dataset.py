import cv2
import os

# Example: change this to match one of your file paths
image_path = 'model\dataset\A-\cluster_0_16.BMP'

# Check if file exists
if not os.path.exists(image_path):
    print(f"File not found: {image_path}")
else:
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Failed to load image. File may be corrupted or unreadable.")
    else:
        print("✅ Image loaded successfully! Shape:", img.shape)
        cv2.imshow('Fingerprint', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
