import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

def show(name, image):
	'''
	A wrapper for cv2.imshow that takes care of waitKey and window cleanup.
	'''
	cv2.imshow(name, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.waitKey(1)

def show_with_axes(name, image):
	'''
	A wrapper for pyplot.imshow, which includes axes. Naturally, cv2 uses
	BGR instead of RGB, so we need to correct for that.
	'''
	print(name)
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.show()

def select_trapezoid(image):
	'''
	Based on the image dimensions, we select from the bottom corners to
	slightly above the center of the image.
	'''
	(rows, cols) = (image.shape[0], image.shape[1])
	bl = (int(cols*.1), int(rows))
	br = (int(cols*.9), int(rows))
	tl = (int(cols*.45), int(rows*.5))
	tr = (int(cols*.55), int(rows*.5))

	# For debugging purposes...
	#'''	
	cv2.line(image, bl, br, (0, 0, 255), 10) # Red
	cv2.line(image, br, tr, (0, 255, 0), 10) # Green
	cv2.line(image, tr, tl, (255, 0, 0), 10) # Blue
	cv2.line(image, tl, bl, (255, 255, 255), 10) # White
	show_with_axes('Lines', image)
	#'''

def detect_lines(image):
	'''
	Works with static images. Ideally we can scale to videos easily.
	'''
	#show('Image', image)
	select_trapezoid(image)

if __name__ == '__main__':
	for file in sys.argv[1:]:
		detect_lines(cv2.imread(file))
