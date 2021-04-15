# Libraries
###################################################################################################
import cv2
import numpy as np
import math
import statistics
import traceback

# Segmentation
DIGITS_LOOKUP_RH = {
    (1, 1, 0, 0, 1, 0, 1): 0,
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}

# Main
###################################################################################################
def valueRecognition(img_org):
    test = True
    # Read Image
    if test:
        cv2.imshow("Original Image", img_org)

    # Canny
    img_canny = cannyImg(img_org, test=test)

    # Recognitions
    for i in [-2, -1, 0]:
        # Find contours
        coordinates, bools = find_contour(img_canny, test=test)
        # Image Rotation
        img_rot = rotateImage(img_org, coordinates, test=test)

        # Processing for dilate
        img_canny_rot = cannyImg(img_rot, test=test)
        coordinates, bools = find_contour(img_canny_rot, test=test)
        img_thresh = processing(img_rot, const=i, test=test)

        if bools:
            try:
                # Perspective Image
                distortion_imgs = perspective_img(coordinates, img_thresh, test=test)
                cv2.waitKey(0)
                # Recognition number
                recognitionNumber, log = segment_recognition(coordinates, distortion_imgs, threshold=0.75, test=test)
                if recognitionNumber != 0:
                    if test:
                        print('=============================================================')
                        print('Threshould const: ', i)
                        print('Value: ', recognitionNumber)
                        print('=============================================================')
                        cv2.waitKey(0)

                    if len(recognitionNumber) == 6:
                        recognitionNumber.append(1)

                    strValue = map(str, recognitionNumber)
                    recognitionValue = ''.join(strValue)
                    return recognitionValue, log
            except:
                traceback.print_exc()
                continue

# Rotate Image
###################################################################################################  
def rotateImage(img_org, coordinates, test):
    # Polyfit
    xPosition, yPosition = [], []
    for coordinate in coordinates:
        xPosition.append(coordinate[0])
        yPosition.append(coordinate[1])
    y = np.array(yPosition)
    z = np.polyfit(xPosition, y, 1)

    rotateAngle = np.arctan2(z[0], 1)

    # Rotate image
    height, width = img_org.shape[0], img_org.shape[1]
    center = (int(0), int(height/2))
    rotateDegree = rotateAngle * 180 / math.pi
    
    trans = cv2.getRotationMatrix2D(center, rotateDegree, 1.0)
    img_rot = cv2.warpAffine(img_org, trans, (width, height))

    if test:
        print(rotateAngle)
        print(rotateDegree)
        cv2.imshow("Rotate Image", img_rot)

    return img_rot

# Canny for detection edges
################################################################################################### 
def cannyImg(img, test):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 0, 380)
    if test:
        cv2.imshow("Canny", img_edges)
    return img_edges

# Processing Image
###################################################################################################  
def processing(img_org, const, test):
    # Grayed
    img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

    # Max Contrast
    height, width = img_gray.shape
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_TopHat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    img_BlackHat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # cv2.imshow("TopHat Image", img_TopHat)
    # cv2.imshow("BlackHat Image", img_BlackHat)

    img_gray_TopHat = cv2.add(img_gray, img_TopHat)
    img_gray_TopHat_BlackHat = cv2.subtract(img_gray_TopHat, img_BlackHat)

    # cv2.imshow("Gray_TopHat Image", img_gray_TopHat)
    # cv2.imshow("Gray_TopHat_BlackHat Image", img_gray_TopHat_BlackHat)

    img_blurred = cv2.GaussianBlur(img_gray_TopHat_BlackHat, ksize=(15, 15), sigmaX=2)
    img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, const)

    # ret, img_thresh = cv2.threshold(img_dilate, 0, 255, cv2.THRESH_OTSU)

    
    img_erode = cv2.erode(img_thresh, kernel, iterations = 1)
    kernel = np.ones((3, 3), np.uint8)
    img_dilate = cv2.dilate(img_erode, kernel, iterations=2)
    
    if test:
        cv2.imshow("Blurred", img_blurred)
        cv2.imshow("Threshould", img_thresh)
        cv2.imshow("Erode", img_dilate)

    return img_dilate


# Find Contours
###################################################################################################
def find_contour(img_thresh, test):
    try:
        contours, npaHierarchy = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        height, width = img_thresh.shape
        img_contours = np.zeros((height, width, 3), np.uint8)

        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1)
        cv2.imshow("Contours Image", img_contours)

        # Selected contours 1
        selectedContours_1 = []
        for contour in contours:
            [x, y, w, h] = cv2.boundingRect(contour)
            aspectRatio = float(w) / float(h)
            area = w * h

            if 0.05 < aspectRatio < 0.85 and 500 < area < 5000:
                selectedContours_1.append(contour)
                if test:
                    print("x and y: ", x,y)
                    print("w and h: ", w,h)
                    print("Aspect Ratio: ", aspectRatio)
                    print("Area: ", area)

        # Selected contours 2
        selectedContours_2 = []
        for i, contour_1 in enumerate(selectedContours_1):
            for j, contour_2 in enumerate(selectedContours_1):
                if i >= j:
                    continue
                [x1, y1, w1, h1] = cv2.boundingRect(contour_1)
                [x2, y2, w2, h2] = cv2.boundingRect(contour_2)

                # Convert left side of coordinate to right side
                x1, x2 = x1 + w1, x2 + w2
                dist_x, dist_y = abs(x1-x2), abs(y1-y2)
                distance = math.sqrt((dist_x ** 2) + (dist_y ** 2))
                dist_x1_x2 = [distance, contour_1, contour_2]
                selectedContours_2.append(dist_x1_x2)

        # Selected Contours 3
        coordinate = []
        selectedContours_3 = []
        for i, contour_3 in enumerate(selectedContours_2):
            for j, contour_4 in enumerate(selectedContours_2):
                if i >= j:
                    continue
                [x1, y1, w1, h1] = cv2.boundingRect(contour_3[1])
                [x2, y2, w2, h2] = cv2.boundingRect(contour_3[2])
                [x3, y3, w3, h3] = cv2.boundingRect(contour_4[1])
                [x4, y4, w4, h4] = cv2.boundingRect(contour_4[2])
                # width < distance < height, distance/distance rate
                if w1 < contour_3[0] < h1 or w2 < contour_3[0] < h2:
                    if w3 < contour_4[0] < h3 or w4 < contour_4[0] < h4:
                        if 0.95 <= contour_3[0]/contour_4[0] <= 1.05:
                            selectedContours_3.append(contour_3[1])
                            selectedContours_3.append(contour_3[2])
                            selectedContours_3.append(contour_4[1])
                            selectedContours_3.append(contour_4[2])
                            coordinate.append([x1, y1, w1, h1])
                            coordinate.append([x2, y2, w2, h2])
                            coordinate.append([x3, y3, w3, h3])
                            coordinate.append([x4, y4, w4, h4])
            
        coordinate = list(map(list, set(map(tuple, coordinate))))
        coordinate = sorted(coordinate)

        # Rectangle positions filter(Over the same y_position 3)
        coordinates = []
        for pos_1 in coordinate:
            cnt = 0
            pos_y_1, pos_y_2 = pos_1[1], pos_1[1]+pos_1[3]
            for pos_2 in coordinate:
                pos_y_3, pos_y_4 = pos_2[1], pos_2[1]+pos_2[3]
                if pos_y_3 != 0 and pos_y_4 != 0:
                    if 0.9 < pos_y_1/pos_y_3 < 1.1 and 0.9 < pos_y_2/pos_y_4 < 1.1:
                        cnt += 1
            if cnt >= 3:
                coordinates.append(pos_1)
        
        coordinates = list(map(list, set(map(tuple, coordinates))))
        coordinates = sorted(coordinates)

        # x, y方向のずれが大きい座標値を削除する
        cnt = np.array(range(len(coordinates)))
        xPosition, yPosition = [], []
        for coordinate in coordinates:
            xPosition.append(coordinate[0] + coordinate[2])
            yPosition.append(coordinate[1])
        
        xPoints = np.array(xPosition)
        yPoints = np.array(yPosition)

        xLine = np.polyfit(cnt, xPoints, 1)
        yLine = np.polyfit(cnt, yPoints, 1)

        xGaps, yGaps = [], []
        for i, (x, y) in enumerate(zip(xPoints, yPoints)):
            xGap, yGap = 0, 0
            xGap = abs(x - (xLine[0]*i + xLine[1]))
            yGap = abs(y - (yLine[0]*i + yLine[1]))
            
            xGaps.append(xGap)
            yGaps.append(yGap)
        
        xStdev = statistics.pstdev(xGaps)
        yStdev = statistics.pstdev(yGaps)
        for i, (xGap, yGap) in enumerate(zip(xGaps, yGaps)):
            if xGap > np.average(xGaps)+xStdev*1.8 or yGap > np.average(yGaps)+yStdev*1.8:
                coordinates.pop(i)

        
        for contour in selectedContours_3:
            [x, y, w, h] = cv2.boundingRect(contour)
            img_rec = cv2.rectangle(img_thresh,(x,y),(x+w,y+h),(100,255,100),1)
        
        cv2.drawContours(img_contours, selectedContours_3, -1, (0, 255, 0), 1)
        
        for i, coordinate1 in enumerate(coordinates):
            for j, coordinate2 in enumerate(coordinates):
                if i >= j:
                    continue
                
                if abs(coordinate1[0] - coordinate2[0]) < 2 :
                    coordinates.pop(i)

        if test:
            cv2.imshow("Rectangle Image1", img_rec)
            print('Coordinate: ', coordinates)

        return coordinates, True

    except:
        traceback.print_exc()
        print(pos_y_1, pos_y_2, pos_y_3, pos_y_4)
        print(coordinate)
        return False, False


# Distortion
###################################################################################################
def perspective_img(coordinate, img_thresh, test):
    distortion_imgs = []
    for i, position in enumerate(coordinate):
        x,y,w,h = position
        img_trim = img_thresh[y:y+h , x:x+w]
        # Perspective Transform
        dst_val = 0 # Distortion value
        pos_org = np.float32([[dst_val+6,0],[w,0],[0,h],[w-dst_val-7,h]])
        pos_trs = np.float32([[0,0],[w,0],[0,h],[w,h]])
        M = cv2.getPerspectiveTransform(pos_org, pos_trs)
        img_perspect = cv2.warpPerspective(img_trim, M, (w, h))
        distortion_imgs.append(img_perspect)
    
    if test:
        for i, img in enumerate(distortion_imgs):
            cv2.imshow("Trim "+str(i), img)
            
    return distortion_imgs


# Segmentation
###################################################################################################
def segment_recognition(coordinates, distortion_imgs, threshold, test):
    # s = 6 # Width of segment
    digits = []
    log = ''
    # coordinate[x,y,w,h]
    for num, (coordinate, distortion_img) in enumerate(zip(coordinates, distortion_imgs)):
        ret, distortion_img = cv2.threshold(distortion_img, 0, 255, cv2.THRESH_OTSU)
        x, y, w, h = coordinate
        s = int(w/5)
        l, ls = int(w), int(h/2-s)
        aspectRatio = float(w) / float(h)

        # Numer is 1
        if aspectRatio < 0.5:
            segment = distortion_img[0:h, 0:w]
            on_pixel = cv2.countNonZero(segment)
            total_pixel = w*h
            on_pixel_ratio = on_pixel/total_pixel
            if float(on_pixel_ratio) > 0.5:
                digits.append(1)
            
            if test:
                print(segment)
                print(on_pixel_ratio)
                print(digits)
                print('=============================================================')
                print('')

        # Else
        else:
            # Define positions(x, y)
            pos_segs = [(0, 0),         # top
                        (0, 0),         # top left
                        (w-s, s),       # top right
                        (0, int((h-s)/2)),   # center
                        (0, int((h+s)/2)),   # bottom left
                        (w-s, int((h+s)/2)), # bottom right
                        (0, h-s)        # bottom
                        ]
            # Height and Width of segment(width, height)
            size_seg = [(l, s),        # top
                        (s, ls),       # top left
                        (s, ls),       # top right
                        (l, s),        # center
                        (s, ls),       # bottom left
                        (s, ls),       # bottom right
                        (l, s),        # bottom
                        ]
            on = [0] * len(pos_segs)

            # Judgement of ON/OFF
            for i, ((x, y), (width, height)) in enumerate(zip(pos_segs, size_seg)):
                # On pixel counting
                segment = distortion_img[y:y+height, x:x+width]
                on_pixel = cv2.countNonZero(segment)
                total_pixel = width*height
                on_pixel_ratio = on_pixel/total_pixel
                
                if test:
                    print(segment)
                    print(x,y)
                    print('w, h', width, height)
                    print('On pixel', on_pixel)
                    print('On pixel', total_pixel)
                    print('ON pixel ratio: '+str(i)+' =>', on_pixel_ratio)
                
                if float(on_pixel_ratio) > float(threshold):
                    on[i]= 1

            # Recognition of segment
            try:
                digit = DIGITS_LOOKUP_RH[tuple(on)]
            except KeyError:
                digit = 'Failed'

            digits.append(digit)

            if test:
                print('ON '+str(num+1)+ ': ', on)
                print('=============================================================')
                print('')
                print(digits)

    # Remove Faild
    for i, val in enumerate(digits):
        if val == 'Failed':
            if i == 1:
                digits.pop(i)
            else:
                digits[i] = 0
                log = str(i+1) + " is outlier"

    return digits, log
