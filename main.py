import cv2
import numpy as np
import math


# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Load Haar Cascade Parameters
faceVJ = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# faceVJ = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Initialize Parameters
ker_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
ker_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ker_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Initialize Background Subtractor MOG2
history = 200
var = 20
mog_rgb = cv2.BackgroundSubtractorMOG2(history, var)
mog_ybr = cv2.BackgroundSubtractorMOG2(history, var)

# Background Modelling
count = 0
while 1:

    count += 1
    # Capture Frame from Webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret is None:
        break
    # frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert Color Space
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ybr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

    # Train the MOG
    mask_rgb = mog_rgb.apply(rgb, learningRate=1.0/history)
    mask_ybr = mog_ybr.apply(ybr, learningRate=1.0/history)

    # Show Images
    cv2.imshow('Mask RGB - YCB_CR', np.hstack([mask_rgb, mask_ybr]))

    # Exit on ESC
    k = cv2.waitKey(10)
    if k == 27 or count > 100:
        cv2.destroyAllWindows()
        break

# Main Loop
count = 0
while 1:

    # Capture Frame from Webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret is None:
        break
    # frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Drawing Support
    draw = frame.copy()

    # Convert Color Space
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Viola Jones Face Detector
    # gray = cv2.equalizeHist(gray)
    faces = faceVJ.detectMultiScale(gray, 1.2, 2)  # 1.1 3

    # Extract Detected Face from the Image
    face = np.zeros_like(frame)
    max_area = -1
    xf, yf, wf, hf = 0, 0, 0, 0
    if len(faces):
        for (x, y, w, h) in faces:
            area = w*h
            if area > max_area:
                max_area = area
                xf, yf, wf, hf = x, y, w, h
        pad_wf, pad_hf = int(0.15*wf), int(0.15*hf)
        face = frame[yf + pad_hf:yf + hf - pad_hf,
                     xf + pad_wf:xf + wf - pad_wf]

    # Compute Back Projection
    face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    face_hist = cv2.calcHist([face_hsv], [0, 1], None, [64, 64], [0, 180, 0, 256])
    cv2.normalize(face_hist, face_hist, 0, 255, cv2.NORM_MINMAX)
    back_pro = cv2.calcBackProject([hsv], [0, 1], face_hist, [0, 180, 0, 256], 1)

    # Remove Face
    x_lim = frame.shape[1]
    y_lim = frame.shape[0]
    pad_wf, pad_hf = int(0.15*wf), int(0.55*hf)
    x_min = xf - pad_wf
    x_max = xf + wf + pad_wf
    y_min = yf - pad_hf
    y_max = yf + hf + pad_hf
    if x_min < 0:
        x_min = 0
    if x_max > x_lim:
        x_max = x_lim
    if y_min < 0:
        y_min = 0
    if y_max > y_lim:
        y_max = y_lim
    back_pro[y_min:y_max, x_min:x_max] = np.zeros((y_max - y_min, x_max - x_min), np.uint8)

    # Image Enhancement for Back Projection
    back_con = back_pro.copy()
    back_con = cv2.filter2D(back_con, -1, ker_5)
    # back_con = cv2.medianBlur(back_con, 5)
    # back_con = cv2.GaussianBlur(back_con, (5, 5), 0)

    # Obtain Background Mask
    mask_rgb = mog_rgb.apply(rgb, learningRate=0.0/history)

    # Apply Threshold to Background Mask
    thresh_rgb = cv2.threshold(mask_rgb, 126, 255, cv2.THRESH_BINARY)[1]
    # thresh_rgb = cv2.adaptiveThreshold(mask_rgb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 0)

    # Apply Morphology to Background Mask
    morph_rgb = thresh_rgb.copy()
    morph_rgb = cv2.morphologyEx(morph_rgb, cv2.MORPH_CLOSE, ker_5, iterations=1)
    morph_rgb = cv2.morphologyEx(morph_rgb, cv2.MORPH_OPEN, ker_3, iterations=1)

    # Obtain Hand Mask TODO test gaussian blur with threshold
    mask_hand = back_con & morph_rgb
    # mask_hand = cv2.GaussianBlur(mask_hand, (5, 5), 0)
    thresh_hand = cv2.threshold(mask_hand, 225, 255, cv2.THRESH_BINARY)[1]
    # thresh_hand = cv2.threshold(mask_hand, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Apply Morphology to Hand Mask
    morph_hand = thresh_hand.copy()
    morph_hand = cv2.morphologyEx(morph_hand, cv2.MORPH_CLOSE, ker_5, iterations=2)

    # Hand Contours and Convex Hull RETR_TREE RETR_CCOMP RETR_EXTERNAL RETR_LIST CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(morph_hand.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(gray, np.uint8)
    max_area = -1
    xc, yc, wc, hc = 0, 0, 0, 0
    ci = 0
    if len(contours):
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                ci = i
        cnt = contours[ci]
        cv2.drawContours(mask, [cnt], 0, 255, -1)

        # Distance Transform
        dist = cv2.distanceTransform(mask, cv2.cv.CV_DIST_L2, 5)
        y_dist, x_dist = np.unravel_index(dist.argmax(), dist.shape)
        cv2.circle(draw, (x_dist, y_dist), 5, [255, 0, 255], -1)

        # Find Convexity Defects
        hull = cv2.convexHull(cnt)
        cv2.drawContours(draw, [cnt], 0, (0, 255, 0), 1)
        cv2.drawContours(draw, [hull], 0, (255, 0, 0), 1)
        hull = cv2.convexHull(cnt, returnPoints=False)
        if hull.size > 6:
            defects = cv2.convexityDefects(cnt, hull)
            count_defects = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
                    if angle <= 90 and d > 1000:  # TODO work on depth d
                        count_defects += 1
                        cv2.circle(draw, far, 5, [0, 0, 255], -1)

    # # Show Images
    cv2.imshow('Foreground with Thresh', np.hstack([mask_rgb, thresh_rgb]))
    cv2.imshow('Backprojection with Morph', np.hstack([back_con, morph_rgb]))
    cv2.imshow('Extracted Hand', np.hstack([morph_hand, mask]))
    cv2.imshow('Face', face)
    cv2.imshow('Frame', draw)

    # Exit on ESC
    k = cv2.waitKey(10)
    if k == 27:
        cv2.destroyAllWindows()
        break
