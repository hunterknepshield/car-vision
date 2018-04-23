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
	BGR instead of RGB, so we need to correct for that. Can be overridden. Also
	properly handles grayscale images automatically.
	'''
	if len(image.shape) == 2 or image.shape[2] == 1:
		# Just show grayscale
		plt.imshow(image, cmap='gray' if conversion is None or isinstance(conversion, int) else conversion)
	else:
		plt.imshow(cv2.cvtColor(image, conversion) if conversion is not None else image)
	plt.title(name)
	plt.show()


def get_birds_eye_view(image):
	'''
	Based on the image dimensions, we select from the bottom corners to
	slightly above the center of the image. Returns this region warped to be a
	bird's eye view, as well as the points and the transform matrix calculated
	to achieve it.

	To undo the transformation, use
	cv2.warpPerspective(warped, warp_matrix, \
					    (original.shape[1], original.shape[0]), \
						flags=cv2.WARP_INVERSE_MAP)
	or call
	get_original_view(warped, warp_matrix, original.shape)
	'''
	# Generate the original anchor points.
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
	original_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
	width = bottom_right[0] - bottom_left[0]
	height = bottom_left[1] - top_left[1]
	warped_points = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32)

	# Warp to bird's eye view
	perspective_matrix = cv2.getPerspectiveTransform(original_points, warped_points)
	warped = cv2.warpPerspective(image, perspective_matrix, (width, height))
	show_with_axes('Warped', warped)
	return (warped, original_points, perspective_matrix)


def get_original_view(warped, warp_matrix, original_shape):
	'''
	Utility function to undo a warp. Parameters to cv2.warpPerspective are a
	little odd, so this wraps them nicely.
	'''
	return cv2.warpPerspective(warped, warp_matrix, (original_shape[1], original_shape[0]), flags=cv2.WARP_INVERSE_MAP)


def superimpose(original, modified, anchors):
	'''
	Superimposes the modified image within the bounds of anchors on top of the
	original image.
	'''
	assert(original.shape == modified.shape)
	superimposed = original.copy()
	# Crop in slightly to prevent tearing, since it appears that cv2 and numpy
	# calculate diagonal edges slightly differently.
	cropped_anchors = anchors.copy()
	cropped_anchors[0:2, 1] += 1 # Move top
	cropped_anchors[1:3, 0] -= 1 # Move right
	cropped_anchors[2:4, 1] += 1 # Move bottom
	cropped_anchors[3, 0] += 1 # Move left
	cropped_anchors[0, 0] += 1 # Left again, slicing syntax can't do -1:1

	replacement_mask = np.zeros(modified.shape, dtype=np.uint8)
	# Fill the region with white with the same depth as the original image. The
	# anchors must be integers.
	cv2.fillConvexPoly(replacement_mask, cropped_anchors.astype(np.int32), (255,)*original.shape[2])
	# For debugging...
	#show_with_axes('Replacement mask', replacement_mask)
	np.putmask(superimposed, replacement_mask, modified)
	return superimposed


def detect_lines(image):
	'''
	Works with static images. Ideally we can scale to videos easily.
	'''
	#show('Image', image)
	(warped, trapezoid_points, warp_matrix) = get_birds_eye_view(image)
	show_with_axes('Warped', warped)

	# Change white to green
	mask = np.logical_and(warped[:,:,0] >= 200, warped[:,:,1] >= 200, warped[:,:,2] >= 200)
	warped[mask] = (0, 255, 0)

	undone = get_original_view(warped, warp_matrix, image.shape)
	show_with_axes('Undone', undone)

	# Generally, looks like any thresholds higher than these do it as well.
	#show_with_axes('Edges', cv2.Canny(undone, 100, 150))
	#	for upper in range(0, 250, 25):
	#		for lower in range(0, min(110, upper), 10):
	#			show_with_axes('Edges, lower = {}, upper = {}'.format(lower, upper), cv2.Canny(undone, lower, upper))

	superimposed = superimpose(image, undone, trapezoid_points)
	show_with_axes('Superimposed', superimposed)


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
