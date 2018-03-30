import cv2
import numpy as np
import sys

def show(name, image):
	'''
	A wrapper for cv2.imshow that takes care of waitKey and window cleanup.
	'''
	cv2.imshow(name, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.waitKey(1)

def detect_lines(image):
	'''
	Works with static images. Ideally we can scale to videos easily.
	'''
	show('Image', image)

if __name__ == '__main__':
	for file in sys.argv[1:]:
		detect_lines(cv2.imread(file))
