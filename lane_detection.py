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
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.title(name)
	plt.show()

def select_trapezoid(image):
	'''
	Based on the image dimensions, we select from the bottom corners to
	slightly above the center of the image.
	'''
	(rows, cols) = (image.shape[0], image.shape[1])
	bottom_left = (int(cols*.1), int(rows))
	bottom_right = (int(cols*.9), int(rows))
	top_left = (int(cols*.45), int(rows*.6))
	top_right = (int(cols*.55), int(rows*.6))

	# For debugging purposes...
	#'''
	lines = image.copy()
	cv2.line(lines, bottom_left, bottom_right, (0, 0, 255), 10) # Red
	cv2.line(lines, bottom_right, top_right, (0, 255, 0), 10) # Green
	cv2.line(lines, top_right, top_left, (255, 0, 0), 10) # Blue
	cv2.line(lines, top_left, bottom_left, (255, 255, 255), 10) # White
	show_with_axes('Lines', lines)
	#'''

	# cv2 really wants these points in clockwise order from top left
	points = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)
	print(points)
	width = bottom_right[0] - bottom_left[0]
	height = bottom_left[1] - top_left[1]
	new_perspective = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
	print(new_perspective)

	perspective_matrix = cv2.getPerspectiveTransform(points, new_perspective)
	print(perspective_matrix)
	warped = cv2.warpPerspective(image, perspective_matrix, (width, height))
	show_with_axes('Warped', warped)

def detect_lines(image):
	'''
	Works with static images. Ideally we can scale to videos easily.
	'''
	#show('Image', image)
	select_trapezoid(image)


def cannyedge(image, lowerbound, upperbound):
	'''
	applies canny edge detector to get lane markings
	'''
	return cv2.Canny(image,lowerbound,upperbound)

def yellowlane(image):
	'''
	convert image to hsv to get yellow channel, and gray scale to get white channel
	'''
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	lowery = np.array([20, 100, 100], dtype = "uint8")
	uppery = np.array([30, 255, 255], dtype="uint8")
	grays = cv2.inRange(gray, 200, 255)
	yellow = cv2.inRange(hsv, lowery, uppery)

if __name__ == '__main__':
	for file in sys.argv[1:]:
		detect_lines(cv2.imread(file))
