'''
Basic driver assistance implementation that provides lane depature detection,
estimates in-lane vehicle distances, and core cruise control support.
'''
import cv2
import sys
import os
from lane_detection import detect_lines
#from vehicle_detection import detect_vehicles


#Set of supported image formats
EXTENSIONS = set(['jpg','jpeg','jif','jfif','jp2','j2k','j2c','fpx','tif', \
		  'tiff','pcd','png','ppm','webp','bmp','bpg','dib','wav', \
		  'cgm','svg'])

# Set of supported video formats
VIDEO_EXTENSIONS = set(['mp4'])


def perceive_road(file):
    '''
    Handles lane and vehicle detection then determines course of action
    @params:
        file: input image of road
    @returns:
        TBD
    '''
    detect_lines(cv2.imread(file), True)
    #detect_vehicles()


def perceive_road_video(file):
	'''
	A quick wrapper around lane detection for a video file. See above.
	'''
	cap = cv2.VideoCapture(file)
	while cap.isOpened():
		(ret, frame) = cap.read()
		painted = detect_lines(frame, False) # Don't stop stuff with show calls
		cv2.imshow('Frame', painted)
		# 25 ms is suggested for smooth video playback
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
	cv2.waitKey(1)


if __name__ == '__main__':
    '''
    Manage input, output, and program's operational flow
    '''
    if (len(sys.argv) < 2):
        print('Error: No target provided')
        sys.exit()

    for file in sys.argv[1:]:
		ext = file.split('.')[1].lower()
		if os.path.isdir(file):
			print('Error: cannot supply directory - ' + str(file))
		elif ext in EXTENSIONS:
            perceive_road(file)
		elif ext in VIDEO_EXTENSIONS:
			perceive_road_video(file)
        else:
            print('Error: unsupported input - ' + str(file))
