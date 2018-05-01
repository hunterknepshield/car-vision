'''
Simple vehicle detection implementation that determines if there is a vehicle
ahead, calculates the distances to which and determines if we're encroaching on
the vehicle ahead.
'''
import cv2
import numpy as np


def error(imgA, imgB):
    '''
    Calculate the mean squared error between the two images
    @params:
        imgA: Half of an image
        imgB: The other half of an image
    @returns:
        mse: the mean sqaured error calculated between the two images
    '''
    mse = np.sum((imgA.astype('float') - imgB.astype('float')) ** 2)
    mse /= float(imgA.shape[0] * imgA.shape[1])
    return mse


def diff_horizontal(img):
    '''
    Differentiate the image based on left and right halves of the image
    @params:
        img: image of a potential car
    @returns:
        Calculated mean squared error between the left and right image halves
    '''
    h,w,c = img.shape
    half = w//2

    #Split the image and resize
    left = img[0:h, 0:half]
    left = cv2.resize(left, (32,64))
    right = img[0:h, half:half+half-1]
    right = cv2.flip(right,1)
    right = cv2.resize(right, (32,64))

    return ( error(left,right) )


def diff_vertical(img):
    '''
    Differentiate the image based on top and bottom halves of the image
    @params:
        img: image of a potential car
    @returns:
        Calculated mean squared error between the top and bottom image halves
    '''
    h,w,c = img.shape
    half = h//2

    #Split the image and resize
    bttm = img[half:half+half, 0:w]
    bttm = cv2.resize(bttm, (32,64))
    top = img[0:half, 0:w]
    top = cv2.flip(top,1)
    top = cv2.resize(top, (32,64))

    return ( error(top,bttm) )


def new_roi(rx, ry, rw, rh, rectangles):
    '''
    Decipher if the region is a new region of interest (i.e. a newly detected car).
    *Mainly applicable for video
    '''
    for r in rectangles:
        if abs(r[0] - rx) < 20 and abs(r[1] - ry) < 20:
           return False
    return True


def decipher_car(road, cascade, scalar=2, downsize=True, debug=False):
    '''
    Cascade file approach to deciphering cars
    @params:
        road: image of the road ahead
        cascade: Pre-trained cascade classifier for haar-car-like features
    @return:
        roi: regions within the image that contain cars
    '''
    roi = []
    h,w,c = road.shape

    if downsize:
        #Scale down the image size (helps filter false positives)
        road = cv2.resize(road, (w//scalar, h//scalar))
    else:
        #Scale up the image size (helps decipher vehicles at distance)
        road = cv2.resize(road, (w*scalar, h*scalar))

    #Noise reduction
    img = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),0)

    #Haar-car-like detection
    cars = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=2)

    minY = road.shape[0]-(road.shape[0]*0.2)
    for (x,y,w,h) in cars:
        car = road[y:y+h, x:x+w]
        if y <  minY:
            diffX = round(diff_horizontal(car))
            diffY = round(diff_vertical(car))
            if debug:
                print('(dX->'+str(diffX)+', dY->'+str(diffY)+')')
            if diffX > 1000 and diffY > 2000:
                roi.append( [x*scalar,y*scalar,w*scalar,h*scalar])

    return roi


def object_ahead(road, lane, cascade):
    '''
    Determine whether or not there is an object or vehicle ahead on the projected path
    @params:
        lane: image of the projected path for the vehicle
        road: image of the entire road ahead
        cascade: Pre-trained cacade classifer for haar-car-like features
    @returns:
        block: true if there is object ahead or false otherwise
        lane: image of the painted projected path with vehicles detected
        rectangles: array of image regions where vehicles were detected
    '''
    rectangles = []
    roi = []

    #Detected regions with cars
    regions = decipher_car(road, cascade)

    for region in regions:
        if new_roi(region[0],region[1],region[2],region[3],rectangles):
            rectangles.append(region)

    #Draw rectangles around cars
    for r in rectangles:
        cv2.rectangle(lane,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,0,255),3)

    #Naive approach
    return (len(rectangles) > 0, lane, rectangles)


def detect_vehicles(road,lane,cascade,video=False):
    '''
    Detect vehicles
    @params:
        road: image of the entire road ahead
        lane: segmented lane image of projected path
        cascade: Pre-trained cascade classifer for haar-car-like features
    @returns:
        lane: image of the painted projected path with vehicle detected
        rectangles: array of image regions where vehicles were detected
    '''
    if video:
        detected,lane,rectangle = object_ahead(road,lane,cascade)
        return (lane,rectangle)
    else:
        detected,lane,rectangle = object_ahead(road,lane,cascade)
        if detected:
            #Detected vehicle pertinent code here.
            return (lane,rectangle)
