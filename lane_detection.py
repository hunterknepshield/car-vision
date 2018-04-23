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

	To undo the transformation, use either
	get_original_view(warped, warp_matrix, original.shape), or
	cv2.warpPerspective(warped, warp_matrix, (original.shape[1], original.shape[0]), flags=cv2.WARP_INVERSE_MAP).
	'''
	# Generate the original anchor points.
	(rows, cols) = (image.shape[0], image.shape[1])
	bottom_left = (int(cols*.1), int(rows))
	bottom_right = (int(cols*.9), int(rows))
	top_left = (int(cols*.45), int(rows*.6))
	top_right = (int(cols*.55), int(rows*.6))

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


def binary_histogram(binary_image, mask=None):
	'''
	Scans horizontally across a binary image and returns a histogram of pixel
	count against the column index. Gives an idea of horizontal location of lane
	markings.
	'''
	masked = np.bitwise_and(binary_image, mask) if mask is not None else binary_image
	masked[masked > 0] = 1
	return np.sum(masked, axis=0) # Sum along columns


def find_histogram_maxima(histogram):
	'''
	Returns a pair (of which one or both may be None) of column numbers that
	have local maxima. Exactly divides the histogram in half and searches on
	either side of the dividing line. This fits the assumption that lanes will
	never cross over to the other side.
	'''
	half = int(histogram.shape[0]/2)
	left = np.argmax(histogram[:half])
	if histogram[left] == 0:
		left = None
	right = np.argmax(histogram[half:]) + half
	if histogram[right] == 0:
		right = None
	return (left, right)


def detect_lines(image):
	'''
	Works with static images. Ideally we can scale to videos easily.
	'''
	show_with_axes('Original', image)
	(warped, trapezoid_points, warp_matrix) = get_birds_eye_view(image)
	show_with_axes('Warped', warped)

	'''
	# Change white to green
	mask = np.logical_and(warped[:,:,0] >= 200, warped[:,:,1] >= 200, warped[:,:,2] >= 200)
	warped[mask] = (0, 255, 0)
	'''

	gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	# TODO(hknepshield) blur first?
	(thresh_num, thresholded) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	show_with_axes('Thresholded', thresholded)
	# Look at 50 pixel tall strips of the image and find the two lines from the
	# peaks in the strips' histograms.
	left_points = []
	right_points = []
	strip_size = 50
	for strip_end in range(thresholded.shape[0], 0, -strip_size):
		strip_begin = max(strip_end - strip_size, 0)
		mask = np.zeros(thresholded.shape[:2], dtype=np.uint8)
		mask[strip_begin:strip_end, :] = 255
		hist = binary_histogram(thresholded, mask)
		(lmax, rmax) = find_histogram_maxima(hist)
		'''
		plt.plot(hist)
		plt.title('Histogram for strip ({}, {})'.format(strip_begin, strip_end))
		plt.show()
		'''
		if lmax is not None:
			point = (lmax, int((strip_begin + strip_end)/2))
			cv2.circle(warped, point, 1, (255, 0, 0), thickness=8)
			left_points.append(point)
		if rmax is not None:
			point = (rmax, int((strip_begin + strip_end)/2))
			cv2.circle(warped, point, 1, (0, 0, 255), thickness=8)
			right_points.append(point)
	# Interpolate to the bottom of the image, since un-warping as-is causes a
	# pretty large gap from the bottom to the first point.
	bottom_row = thresholded.shape[0] - 1
	left_points = np.array(left_points, dtype=np.int32)
	left_line = np.polyfit(left_points[:, 1], left_points[:, 0], 2) # Quadratic
	left_interp = np.polyval(left_line, bottom_row)
	left_points = np.insert(left_points, 0, np.array([left_interp, bottom_row], dtype=left_points.dtype), axis=0)
	right_points = np.array(right_points, dtype=np.int32)
	right_line = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
	right_interp = np.polyval(right_line, bottom_row)
	right_points = np.insert(right_points, 0, np.array([right_interp, bottom_row], dtype=right_points.dtype), axis=0)
	cv2.polylines(warped, [left_points], False, (0, 255, 0), thickness=5)
	cv2.polylines(warped, [right_points], False, (0, 255, 0), thickness=5)
	show_with_axes('Lines', warped)

	undone = get_original_view(warped, warp_matrix, image.shape)
	show_with_axes('Undone', undone)

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
