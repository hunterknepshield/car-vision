'''
Basic driver assistance implementation that provides lane depature detection,
estimates in-lane vehicle distances, and core cruise control support.
'''
import cv2
import sys
import os
from lane_detection import detect_lines
from vehicle_detection import detect_vehicles


#Set of supported image formats
EXTENSIONS = set(['jpg','jpeg','jif','jfif','jp2','j2k','j2c','fpx','tif', \
		  'tiff','pcd','png','ppm','webp','bmp','bpg','dib','wav', \
		  'cgm','svg'])


def perceive_road(file):
    '''
    Handles lane and vehicle detection then determines course of action
    @params:
        file: input image(s) of road
    @returns:
        TBD
    '''
    lane,path,full = detect_lines(cv2.imread(file))
    detect_vehicles(path,full)


if __name__ == '__main__':
    '''
    Manage input, output, and program's operational flow
    '''
    if (len(sys.argv) < 2):
        print('Error: No target provided')
        sys.exit()

    for file in sys.argv[1:]:
        if not os.path.isdir(file) and file.split('.')[1].lower() in EXTENSIONS:
            perceive_road(file)
        else:
            print('Error: unsupported input - ' + str(file))
