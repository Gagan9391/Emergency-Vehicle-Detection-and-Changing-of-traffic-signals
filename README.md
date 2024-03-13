import cv2
import numpy as np
import serial

port = ''
baudrate = 9600
# ser=serial.Serial(port,baudrate)
# signal=1

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

# Initialize zoom factor and zoom flag
zoom_factor = 1
zoom_flag = False

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    # Check if the 'z' key is pressed
    key = cv2.waitKey(1)
    if key == ord('z'):
        zoom_factor = 2 if not zoom_flag else 1
        zoom_flag = not zoom_flag

    # Draw a red plus sign in the middle of the frame
    symbol_size = 40
    thickness = 4
    color = (0, 0, 255)  # Red color

    cv2.line(img, (width // 2 - symbol_size, height // 2),
             (width // 2 + symbol_size, height // 2), color, thickness)
    cv2.line(img, (width // 2, height // 2 - symbol_size),
             (width // 2, height // 2 + symbol_size), color, thickness)

    # Apply zoom if enabled
    img_zoomed = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

    blob = cv2.dnn.blobFromImage(img_zoomed, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width * zoom_factor)
                center_y = int(detection[1]*height * zoom_factor)
                w = int(detection[2]*width * zoom_factor)
                h = int(detection[3]*height * zoom_factor)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            # if(label=='truck'):
            # ser.write(signal)
            # ser.close()
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img_zoomed, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_zoomed, label + " " + confidence, (x, y+20), font, 2, (255, 255, 255), 2)

    cv2.imshow('Image', img_zoomed)

    if key == 13:
        break

cap.release()
cv2.destroyAllWindows()
