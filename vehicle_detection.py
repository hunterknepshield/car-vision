'''
Simple vehicle detection implementation that determines if there is a vehicle
ahead, calculates the distances to which and determines if we're encroaching on
the vehicle ahead.
'''
import cv2
import numpy as np
from lane_detection import show


def write_result(imgs,img,name):
    '''
    Write the images resulting from transformations and morphology
    @params:
        imgs: list of all images
        img: the resultant image post manipulation
        name: the name descriptor for the image
    @returns:
        None
    '''
    newfile = '.'.join(PATH.split('.')[:-1]) + '_'+name+'.' + PATH.split('.')[-1]

    if imgs != None:
        # Create side-by-side comparison and write
        result = np.hstack(imgs)
        cv2.imwrite(newfile, result)
    else:
        cv2.imwrite(newfile, img)


def error(imgA, imgB):
    '''
    Calculate the mean squared error
    '''
    mse = np.sum((imgA.astype('float') - imgB.astype('float')) ** 2)
    mse /= float(imgA.shape[0] * imgA.shape[1])
    return mse


def diff_horizontal(img):
    '''
    Differentiate the image based on left and right similarity
    @params:
        img: image of a potential car
    @returns:
        Mean squared error
    '''
    h,w,c = img.shape
    half = w/2

    left = img[0:h, 0:half]
    left = cv2.resize(left, (32,64))
    right = img[0:h, half:half+half-1]
    right = cv2.flip(right,1)
    right = cv2.resize(right, (32,64))

    return ( error(left,right) )


def diff_vertical(img):
    '''
    Differentiate the image based on top and bottom similarity
    @params:
        img: image of a potential car
    @returns:
        Mean squared error
    '''
    h,w,c = img.shape
    half = h/2

    bttm = img[half:half+half, 0:w]
    bttm = cv2.resize(bttm, (32,64))
    top = img[0:half, 0:w]
    top = cv2.flip(top,1)
    top = cv2.resize(top, (32,64))

    return ( error(top,bttm) )


def new_roi(rx,ry,rw,rh,rectangles):
    '''
    Decipher if the region is a new region of interest (i.e. a newly detected car).
    *Mainly applicable for video
    '''
    for r in rectangles:
        if abs(r[0] - rx) < 40 and abs(r[1] - ry) < 40:
           return False
    return True


def decipher_car(road, cascade):
    '''
    Cascade file approach to decipher cars
    @params:
        road: image of the road ahead
        cascade: Pre-trained cascade xml file of haar car features
    @return:
        roi: regions within the image that contain cars
    '''
    #TODO(rjswitzer3) -
    # 1. Need to look into other means of filtering false positives
    # 2. Fix Mean squared error shenanigans
    # 3. Work on deciphering cars at distance (relates to #1)
    scalar = 2
    roi = []
    h,w,c = road.shape

    #Scale down the image size (helps filter false positives)
    road = cv2.resize(road, (w/scalar, h/scalar))
    h,w,c = road.shape

    # haar detection
    cars = cascade.detectMultiScale(road, 1.1, 2)

    #minY = int(h*0.3)

    for (x,y,w,h) in cars:
        car = road[y:y+h, x:x+w]

        carWidth = car.shape[0]
        #if y > minY: #TODO(rjswitzer3) - Fix this garbage
        diffX = diff_horizontal(car)
        diffY = round(diff_vertical(car))

        print(diffX)
        print(diffY)
        #if diffX > 1600 and diffX < 300 and diffY > 12000: #TODO(rjswitzer3) - FIX
        roi.append( [x*scalar,y*scalar,w*scalar,h*scalar])

    return roi



def object_ahead(road):
    '''
    Determine whether or not there is an object or vehicle ahead on the projected path
    @params:
        lane: image of the projected path for the vehicle
        road: image of the entire road ahead
    @returns:
        block: true if there is object ahead or false otherwise
    '''
    rectangles = []
    roi = [0,0,0,0]
    h,w,c = road.shape
    cascade = cv2.CascadeClassifier('cars.xml')

    regions = decipher_car(road, cascade)
    for region in regions:
        if new_roi(region[0],region[1],region[2],region[3],rectangles):
            rectangles.append(region)

    #Draw rectangles around cars
    for r in rectangles:
        cv2.rectangle(road,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,0,255),3)

    show('ROI',road)

    if len(rectangles) > 0:
        return True
    else:
        return False



def calc_distance():
    '''
    Calculate the distance to the object ahead and cache it (or return? - TBD)
    @params:
        TBD
    @returns:
        None
    '''
    print('Calculating distance to object ahead...')
    #TODO Implement


def collate_velocity():
    '''
    Decipher whether or not we are encroaching on the vehicle ahead.
    @params:
        TBD
    @returns:
        None
    '''
    print('Analyzing velocity to determine encroachment...')
    #TODO Research, gameplan, implement


def detect_vehicles(road,lane):
    '''
    Detect vehicles
    @params:
        road: image of the entire road ahead
        lane: segmented lane image of projected path
    @returns:
        TBD
    '''
    print('Initiating vehicle detection...')
    if object_ahead(road):
        calc_distance()
