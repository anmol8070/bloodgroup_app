import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Adjust if your model input is different
    img = img / 255.0  # Adjust based on your training preprocessing
    img = np.expand_dims(img, axis=0)
    return img

