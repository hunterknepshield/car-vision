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

# Set of supported video formats
VIDEO_EXTENSIONS = set(['mp4','avi'])

#Set path
PATH = '/'


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
    newfile = '.'.join(PATH.split('.')[:-1]) + '_'+name+'.jpg' #+ PATH.split('.')[-1]

    if imgs != None:
        # Create side-by-side comparison and write
        result = np.hstack(imgs)
        cv2.imwrite(newfile, result)
    else:
        cv2.imwrite(newfile, img)


def perceive_road(file, debug=False):
	'''
	Handles lane and vehicle detection then determines course of action
	@params:
		file: input image of road
	@returns:
		TBD
	'''
	road = cv2.imread(file)
	painted = detect_lines(road, debug)
	painted = detect_vehicles(road, painted)
	cv2.imshow('Painted', painted)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.waitKey(1)


def perceive_road_video(file):
	'''
	A quick wrapper around lane detection for a video file. See above.
	'''
	cap = cv2.VideoCapture(file)

	while cap.isOpened():
		(ret, frame) = cap.read()
		if frame is None:
			# End of video
			break
		painted = detect_lines(frame, False) # Don't stop stuff with show calls
		painted = detect_vehicles(frame, painted, True)
		cv2.imshow('Frame', painted)
		# 25 ms is suggested for smooth video playback
		if cv2.waitKey(1) & 0xFF == ord('q'):
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
		global PATH
		PATH = file
		if os.path.isdir(file):
			print('Error: cannot supply directory - ' + str(file))
		elif ext in EXTENSIONS:
			# Debug only if there's a single image
			perceive_road(file, False)#len(sys.argv) == 2)
		elif ext in VIDEO_EXTENSIONS:
			print(ext)
			perceive_road_video(file)
		else:
			print('Error: unsupported input - ' + str(file))
