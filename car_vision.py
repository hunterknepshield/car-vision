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
VIDEO_EXTENSIONS = set(['mp4', 'avi'])


def perceive_road(file, debug=False):
	'''
	Handles lane and vehicle detection then determines course of action
	@params:
		file: input image of road
	@returns:
		TBD
	'''
	painted = detect_lines(cv2.imread(file), debug=debug)
	cv2.imshow(file, painted)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.waitKey(1)
	#detect_vehicles()


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
		cv2.imshow(file, painted)
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
			# Debug only if there's a single image
			perceive_road(file, debug=len(sys.argv) == 2)
		elif ext in VIDEO_EXTENSIONS:
			if not perceive_road_video(file): # User signaled to stop looping
				break
		else:
			print('Error: unsupported input - ' + str(file))
