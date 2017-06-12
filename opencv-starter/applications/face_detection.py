import cv2, sys

def detect(path):
    img = cv2.imread(path)
    cascade = cv2.CascadeClassifier("../packages/resources/haarcascades/haarcascade_frontalface_alt.xml")
    rects = cascade.detectMultiScale(img, 1.3, 4, cv2.CASCADE_SCALE_IMAGE, (20,20))

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

def box(rects, img):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    cv2.imwrite('../data/faces_detected.jpg', img)

rects, img = detect("E:\Projects\computer-vision\computer-vision-python\opencv-starter\data\\group.jpg")
print(rects)
# sys.exit(0)
box(rects, img)