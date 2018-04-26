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
        if abs(r[0] - rx) < 20 and abs(r[1] - ry) < 20:
           return False
    return True


def decipher_car(road, cascade, scalar=2, decimate=True):
    '''
    Cascade file approach to decipher cars
    @params:
        road: image of the road ahead
        cascade: Pre-trained cascade xml file of haar car features
        scalar: inteegr value to scale the image based on
        decimate: boolean indicating whether or not to scale down the image
    @return:
        roi: regions within the image that contain cars
    '''
    #TODO(rjswitzer3) -
    # 1. Fix Mean squared error shenanigans
    # 2. Need to look into other means of filtering false positives
    roi = []
    h,w,c = road.shape

    if decimate:
        #Scale down the image size (helps filter false positives)
        road = cv2.resize(road, (w/scalar, h/scalar))
    else:
        #Scale up the image size (helps decipher vehicles at distance)
        road = cv2.resize(road, (w*scalar, h*scalar))

    #Noise reduction
    img = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),0)

    # haar detection
    cars = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=2)

    for (x,y,w,h) in cars:
        car = road[y:y+h, x:x+w]
        #show('CAR',car) #TODO TESTING

        diffX = round(diff_horizontal(car))
        diffY = round(diff_vertical(car))

        print('(dX->'+str(diffX)+', dY->'+str(diffY)+')') #TODO TESTING
        #[1600,3000,12000], [1600,16000,6000]
        #if diffX > 1600 and diffX < 3000 and diffY > 12000: #TODO(rjswitzer3) [1&2]
        roi.append( [x*scalar,y*scalar,w*scalar,h*scalar])

    return roi


def object_ahead(road,lane):
    '''
    Determine whether or not there is an object or vehicle ahead on the projected path
    @params:
        lane: image of the projected path for the vehicle
        road: image of the entire road ahead
    @returns:
        block: true if there is object ahead or false otherwise
    '''
    rectangles = []
    roi = []
    cascade = cv2.CascadeClassifier('cars.xml')

    for i in range(2,4): #(1-4)
        roi.append( decipher_car(road, cascade, i) )
        #roi.append( decipher_car(road, cascade, i, False) )
    regions = [r for region in roi for r in region]
    #regions = decipher_car(road, cascade, 2) #Use this if performances sucks,

    for region in regions:
        if new_roi(region[0],region[1],region[2],region[3],rectangles):
            rectangles.append(region)

    #Draw rectangles around cars
    for r in rectangles:
        cv2.rectangle(lane,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,0,255),3)

    if len(rectangles) > 0:
        return [True,lane]
    else:
        return [False,lane]


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


def detect_vehicles(road,lane,video=False):
    '''
    Detect vehicles
    @params:
        road: image of the entire road ahead
        lane: segmented lane image of projected path
    @returns:
        TBD
    '''
    if video:
        detected,lane = object_ahead(road,lane)
        return lane
    else:
        detected,lane = object_ahead(road,lane)
        if detected:
            calc_distance()
