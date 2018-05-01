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


def paint_trapezoid(image, points, color=(255, 255, 255), middle_color=(0, 0, 255), thickness=8, alpha=0.25):
    '''
    Assumes points are in a clockwise order. Paints directly on the input image.
    '''
    assert(len(points) == 4)
    painted = image.copy()
    cv2.line(painted, tuple(points[0]), tuple(points[1]), color, thickness)
    cv2.line(painted, tuple(points[1]), tuple(points[2]), color, thickness)
    cv2.line(painted, tuple(points[2]), tuple(points[3]), color, thickness)
    cv2.line(painted, tuple(points[3]), tuple(points[0]), color, thickness)
    # Find the middle to paint a vertical line
    sorted_by_y = points[points[:, 1].argsort()]
    top_middle = (sorted_by_y[0] + sorted_by_y[1])//2
    bottom_middle = (sorted_by_y[2] + sorted_by_y[3])//2
    cv2.line(painted, tuple(top_middle), tuple(bottom_middle), middle_color, thickness)
    return cv2.addWeighted(painted, alpha, image, 1 - alpha, 0, image) # dst=image


def get_birds_eye_view(image, is_our_dashcam, debug=False):
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
    if is_our_dashcam:
        # For our dashcam videos... (camera is slightly far to the right)
        bottom_left = (-500, int(rows)) # Yes, it's fine to have negative coords
        bottom_right = (int(cols) + 100, int(rows))
        top_left = (int(cols*.25), int(rows*.35))
        top_right = (int(cols*.55), int(rows*.35))
    else:
        # Canonical for images...
        bottom_left = (int(cols*.1), int(rows))
        bottom_right = (int(cols*.9), int(rows))
        top_left = (int(cols*.45), int(rows*.6))
        top_right = (int(cols*.55), int(rows*.6))

    # The ordering of `points` and `new_perspective` must match.
    original_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    width = bottom_right[0] - bottom_left[0]
    height = bottom_left[1] - top_left[1]
    warped_points = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32)

    if debug:
        lines = paint_trapezoid(image.copy(), original_points, color=(255, 0, 0))
        show_with_axes('Trapezoid selection', lines)

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
    original image. Does __not__ draw on the original image.
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


def mask_yellow(image):
    '''
    Convert image to hsv to get a mask covering yellow parts of the image.
    Output is a binary image in the range [0, 1].
    '''
    # Option 1: HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array([20, 100, 100], dtype=hsv.dtype)
    hsv_upper = np.array([30, 255, 255], dtype=hsv.dtype)
    hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    #show_with_axes('HSV mask', hsv_mask)

    '''
    # Option 2: Lab, roughly equivalent
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_lower = np.array([0, 130, 150], dtype=lab.dtype)
    lab_upper = np.array([255, 255, 255], dtype=lab.dtype)
    lab_mask = cv2.inRange(lab, lab_lower, lab_upper)
    #show_with_axes('Lab mask', lab_mask)
    '''

    return hsv_mask.astype('bool') # Normalizes to [0, 1]
    #return lab_mask.astype('bool') # Normalizes to [0, 1]


def mask_gray(gray):
    '''
    Wrapper around cv2.threshold() call so we can easily swap out thresholding
    methods for grayscale images if necessary. Returns a binary image in the
    range [0, 1].
    '''
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    thresholded = cv2.threshold(scaled_sobel, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresholded


def histogram(binary):
    '''
    Scans vertically across a binary image and returns a histogram of pixel
    count against the column index after thresholding it. Gives an idea of
    horizontal location of lane markings based on location of the maximum.
    '''
    return np.sum(binary, axis=0)


DEFAULT_HORIZONTAL_THRESHOLD=50

def find_histogram_maximum(histogram, left_threshold, right_threshold, vertical_threshold=5, debug=False, subplot_num=None):
    '''
    Finds the index of the maximum value of the supplied histogram. There are
    several conditions to make sure the histogram is roughly unimodal (e.g. the
    histogram should be one half of the lane, not the full lane). For a visual
    representation of what this means, pass debug=True. Returns None if there
    isn't a high-confidence unimodal maximum value.
    '''
    max_index = np.argmax(histogram)
    max_value = histogram[max_index]
    if debug:
        if subplot_num is not None:
            plt.subplot(subplot_num)
            plt.title('Histogram')
        # In order to have a non-None return value, the plot must:
        # - Have the horizontal red line above the black line,
        # - Have both yellow lines visible,
        # - Have no points above the magenta line as well as to the left of the
        #       cyan line (if it exists), and
        # - Have no points above the magenta line as well as to the right of the
        #       green line (if it exists).
        plt.plot(histogram)
        plt.xlim([0, len(histogram)])
        plt.ylim([0, max(max_value, vertical_threshold) + 3])
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
        if subplot_num is None:
            plt.show()
        # else we assume that plt.show() will get called elsewhere
    if max_value < vertical_threshold:
        # Max is too small
        return None
    points_near_max = histogram >= max_value - vertical_threshold
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


# Somewhat-arbitrary constant defining that a region of an image has "enough"
# yellow to be confident that the lane marking is in fact yellow and not white.
YELLOW_THRESHOLD = 200 # count of pixels in a binary image

def points_on_lines(warped, strip_size=50, hood_size=10, debug=False):
    '''
    Performs binary thresholding on the specified image, then scans horizontal
    strips to find where the lane markings are. Returns a pair of arrays of
    points, one for each line. They may not be the same length if one line
    couldn't be found in a given strip.
    '''
    # By default, use grayscale and binary thresholding
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # But if one or both of the lanes is yellow, prefer the HSV mask instead
    vhalf = warped.shape[1]//2
    lyellow = mask_yellow(warped[:, :vhalf])
    ryellow = mask_yellow(warped[:, vhalf:])
    (use_lyellow, use_ryellow) = (lyellow.sum() >= YELLOW_THRESHOLD, ryellow.sum() >= YELLOW_THRESHOLD)
    '''
    # This is obvious enough from the titles of the binary strips plotted below.
    if debug:
        print('Using yellow for left? {} ({}). Right? {} ({}).'.format(use_lyellow, lyellow.sum(), use_ryellow, ryellow.sum()))
    '''
    left_points = []
    right_points = []
    bottom_row = warped.shape[0] - hood_size
    # We care more about finer accuracy at the bottom of the image, so we put
    # the potential partial strip down there.
    for strip_begin in range(0, bottom_row, strip_size):
        strip_end = min(strip_begin + strip_size, bottom_row)

        if use_lyellow:
            lstrip = lyellow[strip_begin:strip_end, :]
        else:
            lstrip = mask_gray(gray[strip_begin:strip_end, :vhalf])
        lhist = histogram(lstrip)
        lmax = find_left_histogram_maximum(lhist, debug=debug, subplot_num=325)

        if use_ryellow:
            rstrip = ryellow[strip_begin:strip_end, :]
        else:
            rstrip = mask_gray(gray[strip_begin:strip_end, vhalf:])
        rhist = histogram(rstrip)
        rmax = find_right_histogram_maximum(rhist, debug=debug, subplot_num=326) # Offset!

        if lmax is not None:
            point = (lmax, (strip_begin + strip_end)//2)
            left_points.append(point)
        if rmax is not None:
            rmax += vhalf # No cute way to inline this correction
            point = (rmax, (strip_begin + strip_end)//2)
            right_points.append(point)

        if debug:
            # Left stuff in one column
            plt.subplot(321)
            plt.imshow(cv2.cvtColor(warped[strip_begin:strip_end, :vhalf], cv2.COLOR_BGR2RGB))
            plt.title('Left strip')
            plt.subplot(323)
            plt.imshow(lstrip, cmap='gray')
            plt.title('Binary ({})'.format('gray' if not use_lyellow else 'yellow'))
            # 325 gets populated from find_left_histogram_maximum call
            # Right stuff in the other
            plt.subplot(322)
            plt.imshow(cv2.cvtColor(warped[strip_begin:strip_end, vhalf:], cv2.COLOR_BGR2RGB))
            plt.title('Right strip')
            plt.subplot(324)
            plt.imshow(rstrip, cmap='gray')
            plt.title('Binary ({})'.format('gray' if not use_ryellow else 'yellow'))
            # 326 gets populated from find_right_histogram_maximum call
            # General housekeeping and prettification
            plt.suptitle('Strip {} to {}; left = {}, right = {}'.format(strip_begin, strip_end, lmax, rmax))
            plt.tight_layout()
            plt.show()
    return (np.array(left_points, dtype=np.int32), np.array(right_points, dtype=np.int32))


def interpolate_bottom_of_lines(lines, bottom_row, polynomial_degree=1):
    '''
    Since points on the lines are marked on the warped image, there will be a
    large gap at the bottom when warping back without interpolating all the way
    to the bottom. Takes a tuple of (left_points, right_points) for simple
    wrapping around points_on_lines(). Only interpolates if both lines have more
    than one point close to the bottom of the image.
    '''
    (left_points, right_points) = lines
    # It's more or less safe to assume that points closer to the car are more
    # likely to fit a straighter line.
    closeness_cutoff = bottom_row//3
    (left_interp, right_interp) = (None, None) # Only interpolate both at once
    # Argmax blows up on empty lists, so we need nested checks
    if len(left_points) > 0:
        left_cutoff = np.argmax(left_points[:, 1] > closeness_cutoff)
        # Need at least (degree + 1) points to interpolate
        if len(left_points) - left_cutoff > polynomial_degree:
            left_line = np.polyfit(left_points[left_cutoff:, 1], left_points[left_cutoff:, 0], polynomial_degree)
            left_interp = np.array([[np.polyval(left_line, bottom_row), bottom_row]], dtype=left_points.dtype)
    if len(right_points) > 0:
        right_cutoff = np.argmax(right_points[:, 1] > closeness_cutoff)
        # Need at least (degree + 1) points to interpolate
        if len(right_points) - right_cutoff > polynomial_degree:
            right_line = np.polyfit(right_points[right_cutoff:, 1], right_points[right_cutoff:, 0], polynomial_degree)
            right_interp = np.array([[np.polyval(right_line, bottom_row), bottom_row]], dtype=right_points.dtype)
    if left_interp is None or right_interp is None:
        # Only return a result when both lines can be interpolated, otherwise we
        # get awkward trapezoid-like shapes that look worse than the original.
        (left_interp, right_interp) = (None, None)
    return (left_interp, right_interp)


def interpolate_top_of_lines(lines, top_middle, polynomial_degree=2):
    '''
    If the top of the lines don't have the same Y coordinate, we want to
    interpolate so they don't have an ugly trapezoid-ish shape at the top. Only
    interpolates if the two lines have uneven points at the top of the image and
    the line that needs to be extended has more than one point. Default
    polynomial_degree allows quadratics since the lines may curve more towards
    the top of the image.
    '''
    (left_points, right_points) = lines
    if len(left_points) == 0 or len(right_points) == 0:
        # Can't interpolate with a missing line
        return (None, None)
    left_ymin = np.min(left_points[:, 1])
    right_ymin = np.min(right_points[:, 1])
    (left_interp, right_interp) = (None, None)
    if left_ymin < right_ymin:
        # Need to interpolate right line.
        # We ideally want a quadratic, but if we can't get that, a line should
        # be relatively decent still.
        while len(right_points) <= polynomial_degree:
            polynomial_degree -= 1
        if polynomial_degree > 0:
            # Interpolate the right line
            right_line = np.polyfit(right_points[:, 1], right_points[:, 0], polynomial_degree)
            right_interp = np.array([[np.polyval(right_line, left_ymin), left_ymin]], dtype=right_points.dtype)
            # Make sure we didn't cross the middle of the trapezoid
            if right_interp[0, 0] < top_middle or right_interp[0, 0] > top_middle*2:
                right_interp = None
    elif right_ymin < left_ymin and len(left_points) > polynomial_degree:
        # Need to interpolate left line.
        # We ideally want a quadratic, but if we can't get that, a line should
        # be relatively decent still.
        while len(left_points) <= polynomial_degree:
            polynomial_degree -= 1
        if polynomial_degree > 0:
            # Interpolate the left line
            left_line = np.polyfit(left_points[:, 1], left_points[:, 0], polynomial_degree)
            left_interp = np.array([[np.polyval(left_line, right_ymin), right_ymin]], dtype=left_points.dtype)
            # Make sure we didn't cross the middle of the trapezoid
            if left_interp[0, 0] > top_middle or left_interp[0, 0] < 0:
                left_interp = None
    # If both lines already have the same highest point, no interpolation needed
    return (left_interp, right_interp)


def paint_lane(warped, left_points, right_points, alpha=0.25):
    '''
    Paints circles on the points supplied for both lines, the interpolated
    lines, and the interpolated lane area. Paints directly on the input image.
    '''
    if len(left_points) == 0 and len(right_points) == 0:
        # Nothing to draw
        return warped
    # These nasty one-liners sort by Y value
    if len(left_points) != 0:
        left_points = left_points[left_points[:, 1].argsort()]
    if len(right_points) != 0:
        right_points = right_points[right_points[:, 1].argsort()]
    painted = warped.copy()
    if len(left_points) != 0 and len(right_points) != 0:
        # We actually have a polygon to fill. Method assumes that points are in
        # clockwise order.
        cv2.fillConvexPoly(painted, np.concatenate((left_points, right_points[::-1])), (0, 255, 0))
    if len(left_points) > 1:
        # Need 2 or more points to draw a line
        cv2.polylines(painted, [left_points], False, (255, 0, 255), thickness=5)
    if len(right_points) > 1:
        # Need 2 or more points to draw a line
        cv2.polylines(painted, [right_points], False, (255, 0, 255), thickness=5)
    for left_point in left_points:
        cv2.circle(painted, tuple(left_point), 1, (255, 0, 0), thickness=10)
    for right_point in right_points:
        cv2.circle(painted, tuple(right_point), 1, (0, 0, 255), thickness=10)
    return cv2.addWeighted(painted, alpha, warped, 1 - alpha, 0, warped)


def detect_lines(image, is_our_dashcam=False, debug=False, paint_extra=False):
    '''
    Works with static images. Ideally we can scale to videos easily. Does
    __not__ paint directly on the input image.
    '''
    if debug:
        show_with_axes('Original', image)
    (warped, trapezoid_points, warp_matrix) = get_birds_eye_view(image, is_our_dashcam, debug)
    if debug:
        show_with_axes('Warped', warped)
    unblurred_warped = warped.copy()
    warped = cv2.GaussianBlur(warped, (5, 5), 0)

    lines = points_on_lines(warped, debug=debug)
    if debug:
        # Show results without interpolation
        painted_no_interp = paint_lane(unblurred_warped.copy(), lines[0], lines[1])
    # We don't want the interpolations interfering with each other
    top_interp = interpolate_top_of_lines(lines, warped.shape[1]//2)
    bottom_interp = interpolate_bottom_of_lines(lines, warped.shape[0] - 1)
    (left_points, right_points) = lines
    interpolated = False
    if top_interp[0] is not None:
        interpolated = True
        left_points = np.append(left_points, top_interp[0], axis=0)
    if top_interp[1] is not None:
        interpolated = True
        right_points = np.append(right_points, top_interp[1], axis=0)
    if bottom_interp[0] is not None:
        interpolated = True
        left_points = np.append(left_points, bottom_interp[0], axis=0)
    if bottom_interp[1] is not None:
        interpolated = True
        right_points = np.append(right_points, bottom_interp[1], axis=0)

    # Don't paint on the slightly uglier blurred version
    painted = paint_lane(unblurred_warped, left_points, right_points)
    if debug:
        if interpolated:
            show_with_axes('Painted (no interpolation)', painted_no_interp)
            show_with_axes('Painted (with interpolation)', painted)
        else:
            show_with_axes('Painted', painted)

    undone = get_original_view(painted, warp_matrix, image.shape)
    if debug:
        show_with_axes('Undone', undone)

    superimposed = superimpose(image, undone, trapezoid_points)
    if paint_extra:
        paint_trapezoid(superimposed, trapezoid_points)
    if debug:
        show_with_axes('Superimposed', superimposed)
    return (superimposed,trapezoid_points)
