from keras.applications.mobilenet_v2 import preprocess_input
from keras_preprocessing.image.utils import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2


def detectMasks(image, faceModel, maskModel):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceModel.setInput(blob)
    detections = faceModel.forward()

    faces = []
    locs = []
    preds = []

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

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        preds = maskModel.predict(faces)
    return (locs, preds)


protxt = "face_detector\\deploy.prototxt"
faceWeights = "face_detector\\res10_300x300_ssd_iter_140000.caffemodel"

faceModel = cv2.dnn.readNet(protxt, faceWeights)

maskModel = load_model("mask_detector.model")

vs = VideoStream(src=0).start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    (locs, preds) = detectMasks(frame, faceModel, maskModel)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
