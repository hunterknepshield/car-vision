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


def strip_threshold_and_histogram(gray, strip_begin, strip_end, debug=False):
	'''
	Scans horizontally across a binary image and returns a histogram of pixel
	count against the column index after thresholding it. Gives an idea of
	horizontal location of lane markings.
	'''
	masked = gray[strip_begin:strip_end, :]
	# TODO(hknepshield) blur first?
	(thresh_num, thresholded) = cv2.threshold(masked, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	hist = np.sum(thresholded, axis=0) # Sum along columns
	if debug:
		# Draws the dividing line between the halves of the strip as well.
		midpoint = masked.shape[1]//2
		plt.subplot(311)
		plt.imshow(masked, cmap='gray')
		plt.axvline(x=midpoint, color='r')
		plt.title('Strip {} to {}'.format(strip_begin, strip_end))
		plt.subplot(312)
		plt.imshow(thresholded, cmap='gray')
		plt.axvline(x=midpoint, color='r')
		plt.title('Thresholded')
		plt.subplot(313)
		plt.plot(hist)
		plt.axvline(x=midpoint, color='r')
		plt.xlim([0, thresholded.shape[1]]) # Kill margins so it lines up nicely
		plt.title('Histogram')
		plt.show()
	return hist


DEFAULT_HORIZONTAL_THRESHOLD=50

def find_histogram_maximum(histogram, left_threshold, right_threshold, vertical_threshold=5, debug=False):
	'''
	Finds the index of the maximum value of the supplied histogram. There are
	several conditions to make sure the histogram is roughly unimodal (e.g. the
	histogram should be one half of the lane, not the full lane). For a visual
	representation of what this means, pass debug=True. Returns None if there
	isn't a high-confidence unimodal maximum value.
	'''
	max_index = np.argmax(histogram)
	max_value = histogram[max_index]
	'''
	if debug:
		# In order to have a non-None return value, the plot must:
		# - Have the horizontal red line above the magenta line,
		# - Have both yellow lines visible,
		# - Have no points above the magenta line as well as to the right of the
		#   	cyan line (if it exists), and
		# - Have no points above the magenta line as well as to the left of the
		#   	green line (if it exists).
		plt.plot(histogram)
		plt.axhline(y=max_value, color='r')
		plt.axvline(x=max_index, color='r')
		plt.axhline(y=vertical_threshold, color='k')
		if max_value - vertical_threshold >= 0:
			plt.axhline(y=max_value - vertical_threshold, color='m')
		if max_index - left_threshold//2 > 0:
			plt.axvline(x=max_index - left_threshold//2, color='y')
			if max_index - left_threshold >= 0:
				plt.axvline(x=max_index - left_threshold, color='c')
		if max_index + right_threshold//2 < len(histogram) - 1:
			plt.axvline(x=max_index + right_threshold//2, color='y')
			if max_index + right_threshold < len(histogram):
				plt.axvline(x=max_index + right_threshold, color='g')
		plt.show()
	#'''
	if max_value < vertical_threshold:
		# Max is too small
		return None
	points_near_max = histogram > max_value - vertical_threshold
	# Ensure that there are no points larger than (max - vertical_threshold)
	# further away than horizontal_threshold from max_index
	if max_index - left_threshold//2 > 0:
		if np.any(points_near_max[:max(max_index - left_threshold, 0)]):
			# There are points above (max_value - vertical_threshold) too far
			# away from the max
			return None
	else:
		# The max is too close to the left end of the range
		return None
	if max_index + right_threshold//2 < len(histogram) - 1:
		if np.any(points_near_max[min(max_index + right_threshold + 1, len(histogram)):]):
			# There are points above (max_value - vertical_threshold) too far
			# away from the max
			return None
	else:
		# The max is too close to the right end of the range
		return None
	return max_index

def find_left_histogram_maximum(histogram, scale=1.15, horizontal_threshold=DEFAULT_HORIZONTAL_THRESHOLD, **kwargs):
	'''
	The left side of a lane will have convergence towards the right, so slightly
	stretch the right threshold more than normal.
	'''
	return find_histogram_maximum(histogram, horizontal_threshold, int(horizontal_threshold*scale), **kwargs)

def find_right_histogram_maximum(histogram, scale=1.15, horizontal_threshold=DEFAULT_HORIZONTAL_THRESHOLD, **kwargs):
	'''
	The right side of a lane will have convergence towards the left, so slightly
	stretch the left threshold more than normal.
	'''
	return find_histogram_maximum(histogram, int(horizontal_threshold*scale), horizontal_threshold, **kwargs)


def points_on_lines(warped, strip_size=50, hood_size=10, debug=False):
	'''
	Performs binary thresholding on the specified image, then scans horizontal
	strips to find where the lane markings are. Returns a pair of arrays of
	points, one for each line. They may not be the same length if one line
	couldn't be found in a given strip.
	'''
	gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	left_points = []
	right_points = []
	bottom_row = gray.shape[0] - hood_size
	# We care more about finer accuracy at the bottom of the image, so we put
	# the potential partial strip down there.
	for strip_begin in range(0, bottom_row, strip_size):
		strip_end = min(strip_begin + strip_size, bottom_row)
		hist = strip_threshold_and_histogram(gray, strip_begin, strip_end, debug)
		half = len(hist)//2
		lmax = find_left_histogram_maximum(hist[:half], debug=debug)
		rmax = find_right_histogram_maximum(hist[half:], debug=debug) # Offset!
		if lmax is not None:
			point = (lmax, (strip_begin + strip_end)//2)
			left_points.append(point)
		if rmax is not None:
			rmax += half # No cute way to inline this correction
			point = (rmax, (strip_begin + strip_end)//2)
			right_points.append(point)
	return (np.array(left_points, dtype=np.int32), np.array(right_points, dtype=np.int32))


def interpolate_bottom_of_lines(lines, bottom_row, polynomial_degree=1):
	'''
	Since points on the lines are marked on the warped image, there will be a
	large gap at the bottom when warping back without interpolating all the way
	to the bottom. Takes a tuple of (left_points, right_points) for simple
	wrapping around points_on_lines(). Only interpolates if both lines have
	enough points close to the bottom of the image.
	'''
	(left_points, right_points) = lines
	# It's more or less safe to assume that points closer to the car are more
	# likely to fit a straighter line.
	closeness_cutoff = bottom_row//3
	(left_interp, right_interp) = (None, None) # Only interpolate both at once
	# Argmax blows up on empty lists
	if len(left_points) > 0:
		left_cutoff = np.argmax(left_points[:, 1] > closeness_cutoff)
		# Need at least 2 points to interpolate
		if len(left_points) - left_cutoff > 1:
			left_line = np.polyfit(left_points[left_cutoff:, 1], left_points[left_cutoff:, 0], polynomial_degree)
			left_interp = np.array([[np.polyval(left_line, bottom_row), bottom_row]], dtype=left_points.dtype)
	if len(right_points) > 0:
		right_cutoff = np.argmax(right_points[:, 1] > closeness_cutoff)
		# Need at least 2 points to interpolate
		if len(right_points) - right_cutoff > 1:
			right_line = np.polyfit(right_points[right_cutoff:, 1], right_points[right_cutoff:, 0], polynomial_degree)
			right_interp = np.array([[np.polyval(right_line, bottom_row), bottom_row]], dtype=right_points.dtype)
	if left_interp is not None and right_interp is not None:
		# Only interpolate when both lines can be interpolated, otherwise we get
		# awkward trapezoid-like shapes that look worse than the original.
		left_points = np.append(left_points, left_interp, axis=0)
		right_points = np.append(right_points, right_interp, axis=0)
	return (left_points, right_points)


def paint_lane(warped, left_points, right_points, alpha=0.25):
	'''
	Paints circles on the points supplied for both lines, the interpolated
	lines, and the interpolated lane area. All painting is done with the
	specified alpha value. The source image is not modified.
	'''
	painted_lane = warped.copy()
	if len(left_points) == 0 and len(right_points) == 0:
		# Nothing to draw
		return painted_lane
	if len(left_points) > 1:
		# Need 2 or more points to draw a line
		cv2.polylines(painted_lane, [left_points], False, (255, 0, 255), thickness=5)
	if len(right_points) > 1:
		# Need 2 or more points to draw a line
		cv2.polylines(painted_lane, [right_points], False, (255, 0, 255), thickness=5)
	for left_point in left_points:
		cv2.circle(painted_lane, tuple(left_point), 1, (255, 0, 0), thickness=10)
	for right_point in right_points:
		cv2.circle(painted_lane, tuple(right_point), 1, (0, 0, 255), thickness=10)
	if len(left_points) != 0 and len(right_points) != 0:
		# We actually have a polygon to fill. Method assumes that points are in
		# clockwise order.
		cv2.fillConvexPoly(painted_lane, np.concatenate((left_points, right_points[::-1])), (0, 255, 0))
	return cv2.addWeighted(painted_lane, alpha, warped, 1 - alpha, 0)


def detect_lines(image, debug=False):
	'''
	Works with static images. Ideally we can scale to videos easily.
	'''
	if debug:
		show_with_axes('Original', image)
	(warped, trapezoid_points, warp_matrix) = get_birds_eye_view(image)
	if debug:
		show_with_axes('Warped', warped)

	#yellowlane(warped)

	(left_points, right_points) = interpolate_bottom_of_lines(points_on_lines(warped, debug=debug), warped.shape[0] - 1)
	# Alternatively, without interpolation to the very bottom:
	#(left_points, right_points) = points_on_lines(warped)
	painted = paint_lane(warped, left_points, right_points)
	if debug:
		show_with_axes('Painted', painted)

	undone = get_original_view(painted, warp_matrix, image.shape)
	if debug:
		show_with_axes('Undone', undone)

	superimposed = superimpose(image, undone, trapezoid_points)
	if debug:
		show_with_axes('Superimposed', superimposed)
	return superimposed


def yellowlane(image):
	'''
	Convert image to hsv to get yellow channel, and gray scale to get white channel
	'''
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	lowery = np.array([20, 100, 100], dtype="uint8")
	uppery = np.array([30, 255, 255], dtype="uint8")
	grays = cv2.inRange(gray, 200, 255)
	yellow = cv2.inRange(hsv, lowery, uppery)
	show_with_axes('hsv', hsv)
	show_with_axes('lab', lab)
	show_with_axes('gray', gray)
	show_with_axes('grays', grays)
	show_with_axes('yellows', yellow)
