'''
Basic driver assistance implementation that provides lane depature detection,
estimates in-lane vehicle distances, and core cruise control support.
'''
import cv2
import sys
import os
import numpy as np
from lane_detection import detect_lines
from vehicle_detection import detect_vehicles


#Set of supported image formats
EXTENSIONS = set(['jpg','jpeg','jif','jfif','jp2','j2k','j2c','fpx','tif', \
				  'tiff','pcd','png','ppm','webp','bmp','bpg','dib','wav', \
				  'cgm','svg'])

# Set of supported video formats
VIDEO_EXTENSIONS = set(['mp4','avi'])


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
	painted = detect_lines(road, debug=debug)
	trapp = painted[1]
	painted = detect_vehicles(road, painted[0])
	paint = painted[0]
	rectangle = painted[1]
	paint = getdistance(trapp,rectangle, paint)
	cv2.imshow('Painted', paint)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.waitKey(1)


def perceive_road_video(file):
	'''
	A quick wrapper around lane detection for a video file. See above. Returns
	false when the user enters 'Q' (capital), indicating that all further files
	should be skipped.
	'''
	cont = True
	cap = cv2.VideoCapture(file)
	is_our_dashcam = file.startswith('videos/dash-cam')
	debug = False
	paint_extra = False
	fast_forward = 0
	manual = False
	while cap.isOpened():
		if fast_forward > 0:
			fast_forward -= 1
			ret = cap.grab() # Doesn't bother decoding a frame
			if not ret:
				break
			continue
		(ret, frame) = cap.read()
		if not ret or frame is None:
			# End of video
			break
		# Set debug to false os we don't show anything with pyplot and block
		painted = detect_lines(frame, is_our_dashcam=is_our_dashcam, debug=debug, paint_extra=paint_extra)
		trapp = painted[1]
		painted = detect_vehicles(frame, painted[0], True)
		paint = painted[0]
		rectangle = painted[1]
		paints = getdistance(trapp, rectangle, paint)
		cv2.imshow(file, paints)
		# 25 ms is suggested for smooth video playback, 1 seems to work too
		# Debug makes it annoying to escape the loop, so wait forever if it's on
		input = cv2.waitKey(1 if not debug and not manual else 0) & 0xFF
		if input == ord('q'):
			# Quit (breaks out of this current video only)
			break
		elif input == ord('Q'):
			# Quit everything (terminates the program)
			cont = False
			break
		elif input == ord('p'):
			# Pause
			cv2.waitKey(0)
		elif input == ord('d'):
			# Debug toggle
			debug = not debug
		elif input == ord('e'):
			# Extra painting toggle
			paint_extra = not paint_extra
		elif input == ord('f'):
			# Fast-forward
			fast_forward = 15
		elif input == ord('F'):
			# Fast-forward even more
			fast_forward = 60
		elif input == ord('m'):
			# Manual toggle
			manual = not manual
	cap.release()
	cv2.destroyAllWindows()
	cv2.waitKey(1)
	return cont

# flaws: the distance is calculated by the ratio of the object in real life to object within the image
# given the focal lenght of the camera and image we can find image ratio, but car height vary so the distance
# isn't the most accurate.
def getdistance(trapp, rectangle, image, focal=.10, carheight = 1.45):
	'''
	:param trapp: the trapezoid points
	:param rectangle: all of the detected objects
	:param focal: focal length of camera
	:param carheight: height of the object
	:return: painted image
	'''
	imheight = image.shape[1]
	if len(trapp) > 0:
		leftboundary = trapp[len(trapp)-1][1]
		rightboundarx = trapp[len(trapp)-2][0]
		leftboundarx = trapp[len(trapp)-1][0]
		middlex = int(leftboundarx+((rightboundarx-leftboundarx)/2))
		cameraheight = carheight/2
		for rect in rectangle:
			if len(rect) !=4:
				continue
			objectyl = rect[1]
			objectyr = rect[1] + rect[3]
			objectheight = objectyr-objectyl
			if rect[0] >= trapp[0][0] and (rect[0]+rect[2]) <= trapp[1][0]: # checks if object is within the bounds of the lane
				distance = (focal *carheight* imheight)/(objectheight*cameraheight) #this is the distance formula
				print("distance to the detected car is: " + str(distance) +" meters away")
				cv2.line(image,(middlex,leftboundary),(rect[0]+rect[2], rect[1]+rect[3]), color=(255,0,0), thickness=1)
				if distance < 25:
					print("slow down")
	return image

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
			perceive_road(file, debug=len(sys.argv) == 2)
		elif ext in VIDEO_EXTENSIONS:
			if not perceive_road_video(file): # User signaled to stop looping
				break
		else:
			print('Error: unsupported input - ' + str(file))
