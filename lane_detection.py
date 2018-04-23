import cv2
import numpy as np
from matplotlib import pyplot as plt


def show(name, image):
	'''
	A wrapper for cv2.imshow that takes care of waitKey and window cleanup.
	'''
	cv2.imshow(name, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.waitKey(1)


def show_with_axes(name, image, conversion=cv2.COLOR_BGR2RGB):
	'''
	A wrapper for pyplot.imshow, which includes axes. Naturally, cv2 uses
	BGR instead of RGB, so we need to correct for that. Can be overridden.
	'''
	plt.imshow(cv2.cvtColor(image, conversion))
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
	show_with_axes('Original', image)

	# For debugging purposes...
	'''
	lines = image.copy()
	cv2.line(lines, bottom_left, bottom_right, (0, 0, 255), 10) # Red
	cv2.line(lines, bottom_right, top_right, (0, 255, 0), 10) # Green
	cv2.line(lines, top_right, top_left, (255, 0, 0), 10) # Blue
	cv2.line(lines, top_left, bottom_left, (255, 255, 255), 10) # White
	show_with_axes('Lines', lines)
	#'''

	# The ordering of `points` and `new_perspective` must match.
	points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
	width = bottom_right[0] - bottom_left[0]
	height = bottom_left[1] - top_left[1]
	new_perspective = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32)

	perspective_matrix = cv2.getPerspectiveTransform(points, new_perspective)
	warped = cv2.warpPerspective(image, perspective_matrix, (width, height))
	show_with_axes('Warped', warped)

	# Warp back
	undo_matrix = cv2.getPerspectiveTransform(new_perspective, points)
	undone = cv2.warpPerspective(warped, undo_matrix, (cols, rows))
	show_with_axes('Undone', undone)

	# Change white to green
	mask = np.logical_and(undone[:,:,0] >= 200, undone[:,:,1] >= 200, undone[:,:,2] >= 200)
	undone[mask] = (0, 255, 0)
	show_with_axes('Changed', undone)

	# Superimpose the changed version on the original, cropping in slightly to
	# prevent tearing.
	replacement_anchors = points.copy()
	replacement_anchors[0:2, 1] += 1 # Move top
	replacement_anchors[1:3, 0] -= 1 # Move right
	replacement_anchors[2:4, 1] += 1 # Move bottom
	replacement_anchors[3, 0] += 1 # Move left
	replacement_anchors[0, 0] += 1 # Left again, slicing syntax can't do -1:1
	replacement_mask = np.zeros(undone.shape, dtype=np.uint8)
	# Fill the region with white with the same depth as the original image.
	cv2.fillConvexPoly(replacement_mask, replacement_anchors.astype(np.int32), (255,)*image.shape[2])
	show_with_axes('Replacement mask', replacement_mask)
	np.putmask(image, replacement_mask, undone)
	show_with_axes('Superimposed', image)


def detect_lines(image):
	'''
	Works with static images. Ideally we can scale to videos easily.
	'''
	#show('Image', image)
	select_trapezoid(image)


def cannyedge(image, lowerbound, upperbound):
	'''
	Applies canny edge detector to get lane markings
	'''
	return cv2.Canny(image,lowerbound,upperbound)


def yellowlane(image):
	'''
	Convert image to hsv to get yellow channel, and gray scale to get white channel
	'''
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	lowery = np.array([20, 100, 100], dtype="uint8")
	uppery = np.array([30, 255, 255], dtype="uint8")
	grays = cv2.inRange(gray, 200, 255)
	yellow = cv2.inRange(hsv, lowery, uppery)
