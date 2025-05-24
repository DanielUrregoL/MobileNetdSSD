import cv2

ALL_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

def model(target_classes: list[str], stop_event=None):
    # Validar clases
    for cls in target_classes:
        if cls not in ALL_CLASSES:
            raise ValueError(f"Clase desconocida: {cls}")

    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    net = cv2.dnn.readNetFromCaffe("model/MobileNetSSD_deploy.prototxt", "model/MobileNetSSD_deploy.caffemodel")
    cap = cv2.VideoCapture(0)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                label = ALL_CLASSES[idx]

                if label not in target_classes:
                    continue

                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                color = COLORS[idx % len(COLORS)]
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, f"{label}: {confidence:.2f}", (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("MobileNet-SSD Detection", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def detect_objects_in_frame(frame, target_classes: list[str]):
    net = cv2.dnn.readNetFromCaffe(
        "model/MobileNetSSD_deploy.prototxt",
        "model/MobileNetSSD_deploy.caffemodel"
    )

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = ALL_CLASSES[idx]

            if label not in target_classes:
                continue

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            results.append({
                "label": label,
                "confidence": float(confidence),
                "x": int(startX),
                "y": int(startY),
                "w": int(endX - startX),
                "h": int(endY - startY),
            })

    return results

