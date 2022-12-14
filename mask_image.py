from keras.applications.mobilenet_v2 import preprocess_input
from keras_preprocessing.image.utils import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True,
                    help="path to input image")
args = parser.parse_args()
imgPath = args.image

protxt = "face_detector\\deploy.prototxt"
faceWeights = "face_detector\\res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNet(protxt, faceWeights)

model = load_model("mask_detector.model")

image = cv2.imread(imgPath)
orig = image.copy()
(h, w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    # confidence for face detection
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:  # proceed only if detected face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        (mask, withoutMask) = model.predict(face)[0]

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 225)

        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

cv2.imshow("Output", image)
cv2.waitKey(0)
