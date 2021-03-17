import cv2

min_blue, min_green, min_red = 0, 0, 180
max_blue, max_green, max_red = 30, 240, 255

v = cv2.__version__.split('.')[0]
print(cv2.__version__)

camera = cv2.VideoCapture(0)

while True:
    _, frame_BGR = camera.read()
    frame_HSV = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(frame_HSV, (min_blue, min_green, min_red),
        (max_blue, max_green, max_red))

    cv2.namedWindow('Binary Image with Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary Image with Mask', mask)

    if v == '3':
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if contours:
        (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])
        cv2.rectangle(frame_BGR, (x_min-15, y_min-15),
           (x_min + box_width + 15, y_min + box_height + 15),
           (0,255,0),3)
        label = 'Dectected Object'
        cv2.putText(frame_BGR, label, (x_min-5, y_min-25),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.namedWindow('Dectected Object', cv2.WINDOW_NORMAL)
    cv2.imshow('Dectected Object', frame_BGR)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()