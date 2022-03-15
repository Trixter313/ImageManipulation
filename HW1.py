import math
import glob
import cv2 as cv
import numpy as np

# Specify file path here for all functions to use
filePath = "example-images/nola.jpg"
filePathSplit = filePath.split("/")
fileName = filePathSplit[len(filePathSplit) - 1]

# File path for 2nd image for histogram
filePath2 = "example-images/chicken.jpg"

# Set to false to not have tests run
runTests = True;

# Set to false to not save files from tests
saveFiles = True;

def displayImage(img, prefix = "Image"):
	"""Uses OpenCV to display a window of an image in memory.
		Note: Make sure to close the image by pressing any key, rather than ui=sing the system close button.
		Closing improperly will result in the program getting stuck.

		Parameters:\n
		img - the image you want to display.\n
		prefix (optional) - text that you want To show up in the window title bar. Displays as "[prefix] filePath".
	"""
	cv.imshow("[" + prefix + "]", img)
	cv.waitKey(0)
	cv.destroyAllWindows()

def BGR_to_HSV(bgrimg):
	"""Returns an HSV image from an RGB image.

		Parameters:\n
		bgrimg - an RGB image you want to turn into HSV.
	"""
	hsvimg = np.empty(bgrimg.shape, np.uint16)
	for n in range(bgrimg.shape[0]):
		for m in range(bgrimg.shape[1]):
			b = bgrimg[n, m, 0] / 255
			g = bgrimg[n, m, 1] / 255
			r = bgrimg[n, m, 2] / 255
			cmax = max(b, g, r)
			cmin = min(b, g, r)

			v = round(cmax * 255)
			s = 0 if (cmax == 0) else round(((cmax - cmin) / cmax) * 255)

			# Hue calculation
			if s == 0:
				h = 361
			else:
				delta = cmax - cmin

				if cmax == b:
					h = 4 + (r - g) / delta
				elif cmax == g:
					h = 2 + (b - r) / delta
				elif cmax == r:
					h = (g - b) / delta
				
				# We want the max to be 359, not 360
				h = round(h * (359/6))

				# To prevent negatives
				if h < 0:
					h += 359

			# Put it all together
			hsvimg[n, m] = np.array([h, s, v])
	return hsvimg

# Uncomment block to test BGR_to_HSV()
if runTests:
	imgBGR = cv.imread(filePath)
	displayImage(imgBGR, "BGR")
	imgHSV = BGR_to_HSV(imgBGR)
	if saveFiles:
		cv.imwrite("HSV_" + fileName, imgHSV)
	displayImage(imgHSV, "HSV")

def yellow_mask(hsvimg):
	"""Returns a mask of yellow hues between 46 and 66 from an HSV image.
		Note: Ignores pixels with a hue or saturation under 50.

		Parameters:\n
		hsvimg - the image you want to make a mask for.
	"""
	# Use np.zeros so we don't have to put 0's where the mask shouldn't be
	mask = np.zeros((hsvimg.shape[0], hsvimg.shape[1], 1), np.uint8)
	for n in range(hsvimg.shape[0]):
		for m in range(hsvimg.shape[1]):
			# Skip images with an S or V value under 50
			if hsvimg[n, m, 1] > 49 and hsvimg[n, m, 2] > 49:
				# Now skip non-yellow pixels from those
				if 46 <= hsvimg[n, m, 0] <= 66:
					mask[n, m, 0] = 255
			else:
				mask[n, m, 0] = 0
	return mask

# Uncomment block to test yellow_mask()
if runTests:
	imgYellowMask = yellow_mask(imgHSV)
	if saveFiles:
		cv.imwrite("YellowMask_" + fileName, imgYellowMask)
	displayImage(imgYellowMask, "Yellow Mask")

def green_mask(hsvimg):
	"""Returns a mask of green hues between 67 and 168 from an HSV image.
		Note: Ignores pixels with a hue or saturation under 50.

		Parameters:\n
		hsvimg - the image you want to make a mask for.
	"""
	# Use np.zeros so we don't have to put 0's where the mask shouldn't be
	mask = np.zeros((hsvimg.shape[0], hsvimg.shape[1], 1), np.uint8)
	for n in range(hsvimg.shape[0]):
		for m in range(hsvimg.shape[1]):
			# Skip images with an S or V value under 50
			if hsvimg[n, m, 1] > 49 and hsvimg[n, m, 2] > 49:
				# Now skip non-green pixels from those
				if 67 <= hsvimg[n, m, 0] <= 168:
					mask[n, m, 0] = 255
			else:
				mask[n, m, 0] = 0
	return mask

# Uncomment block to test green_mask()
if runTests:
	imgGreenMask = green_mask(imgHSV)
	displayImage(imgGreenMask, "Green Mask")

	imgGreenSelected = cv.bitwise_and(imgBGR, imgBGR, mask = imgGreenMask)
	displayImage(imgGreenSelected, "Green Selected")
	if saveFiles:
		cv.imwrite("GreenMask_" + fileName, imgGreenMask)
		cv.imwrite("GreenSelected_" + fileName, imgGreenSelected)

def hue_mask(hsvimg, minhue, maxhue, minsat = "50", minval = "50"):
	"""Returns a mask for the specified hue range and minimum saturation and value.
		Note: To specify a range that goes through 0, put the larger number as the minhue.

		Parameters:\n
		hsvimg - the image you want to make a mask for.\n
		minhue - the beginning of the hue range you want to mask.\n
		maxhue - the end of the hue range you want to mask.\n
		minsat - the minimum saturation value you want to mask.\n
		minval - the minimum value value you want to mask.
	"""
	# Use np.zeros so we don't have to put 0's where the mask shouldn't be
	mask = np.zeros((hsvimg.shape[0], hsvimg.shape[1], 1), np.uint8)
	for n in range(hsvimg.shape[0]):
		for m in range(hsvimg.shape[1]):
			# Skip images with an S or V value under minsat and minval
			if hsvimg[n, m, 1] >= int(minsat) and hsvimg[n, m, 2] >= int(minval):
				# Now skip unwanted from those
				# Need to handle when range goes through 0 Ex. if someone wants reds from ~330-30
				if minhue > maxhue:
					if minhue <= hsvimg[n, m, 0] <= 0 or 0 <= hsvimg[n, m, 0] <= maxhue:
						mask[n, m, 0] = 255
				elif minhue <= hsvimg[n, m, 0] <= maxhue:
					mask[n, m, 0] = 255
	return mask

# Uncomment block to test hue_mask()
if runTests:
	# Test of same range as in yellow to confirm correctness:
	imgHueMaskYellow = hue_mask(imgHSV, 46, 66)
	displayImage(imgHueMaskYellow, "Hue Mask Yellow")
	imgYellowSelected = cv.bitwise_and(imgBGR, imgBGR, mask = imgHueMaskYellow)
	displayImage(imgYellowSelected, "Yellow Selected")

	# Test of reversing order to go through 0:
	imgHueMaskReverse = hue_mask(imgHSV, 345, 15)
	displayImage(imgHueMaskReverse, "Hue Mask Reverse (Red)")
	imgRedSelected = cv.bitwise_and(imgBGR, imgBGR, mask = imgHueMaskReverse)
	displayImage(imgRedSelected, "Red Selected")

	if saveFiles:
		cv.imwrite("HueMaskYellow_" + fileName, imgHueMaskYellow)
		cv.imwrite("YellowSelected_" + fileName, imgYellowSelected)
		cv.imwrite("HueMaskReverse_Red_" + fileName, imgHueMaskReverse)
		cv.imwrite("RedSelected_" + fileName, imgRedSelected)

def compute_HS_histogram(hsvimg):
	"""Computes the histogram for an HSV image using nine hue ranges and five saturation ranges:
			H 0-39					S 50-90\n
			H 40-79					S 91-131\n
			H 80-119				S 132-172\n
			H 120-159				S 173-213\n
			H 160-199				S 214-255\n
			H 200-239\n
			H 240-279\n
			H 280-319\n
			H 320-359

		Parameters:\n
		hsvimg - the image you want to compute the histogram of.
	"""
	histogram = np.zeros((9, 5), float)
	for n in range(hsvimg.shape[0]):
		for m in range(hsvimg.shape[1]):
			# Skip images with an S or V value under 50
			if hsvimg[n, m, 1] > 49 and hsvimg[n, m, 2] > 49:
				# Check hue
				if hsvimg[n, m, 0] <= 39:
					x = 0
				elif hsvimg[n, m, 0] <= 79:
					x = 1
				elif hsvimg[n, m, 0] <= 119:
					x = 2
				elif hsvimg[n, m, 0] <= 159:
					x = 3
				elif hsvimg[n, m, 0] <= 199:
					x = 4
				elif hsvimg[n, m, 0] <= 239:
					x = 5
				elif hsvimg[n, m, 0] <= 279:
					x = 6
				elif hsvimg[n, m, 0] <= 319:
					x = 7
				else:
					x = 8
				
				# Check saturation
				if hsvimg[n, m, 1] <= 90:
					y = 0
				elif hsvimg[n, m, 1] <= 131:
					y = 1
				elif hsvimg[n, m, 1] <= 172:
					y = 2
				elif hsvimg[n, m, 1] <= 213:
					y = 3
				else:
					y = 4
				
				# Increment appropriate value
				histogram[x, y] += 1
	histogram = histogram.flatten()
	histogramSum = sum(histogram)
	for i, val in enumerate(histogram):
		histogram[i] = histogram[i] / histogramSum
	return histogram

if runTests:
	histogram1 = compute_HS_histogram(imgHSV)
	print("Histogram:", histogram1)

def L1(hist1, hist2):
	"""Calculates the L1 distance between two normalized 1D histograms. Returns the distance.

		Parameters:\n
		hist1 - one of the histograms.\n
		hist2 - the other histogram.
	"""
	distance = 0
	for i, val in enumerate(hist1):
		distance += abs(hist1[i] - hist2[i])
	return distance

if runTests:
	imgBGR2 = cv.imread(filePath2)
	imgHSV2 = BGR_to_HSV(imgBGR2)
	histogram2 = compute_HS_histogram(imgHSV2)
	print("L1 distance:", L1(histogram1, histogram2))

def L2(hist1, hist2):
	"""Calculates the L2 distance between two normalized 1D histograms. Returns the distance.

		Parameters:\n
		hist1 - one of the histograms.\n
		hist2 - the other histogram.
	"""
	distance = 0
	for i, val in enumerate(hist1):
		distance += (hist1[i] - hist2[i])**2
	distance = math.sqrt(distance)
	return distance

if runTests:
	print("L2 distance:", L2(histogram1, histogram2))

# For each image name in the list, read in the image using cv.imread, convert to
# HSV using your BGR_to_HSV function, and compute the normalized 45 bin HS histogram
# using your compute_HS_histogram function. Compute either the L1 or L2 distance
# between the key histogram and the histogram of every other image and keep track of
# the image name that is the least distance. Return a tuple of the key image name and
# the closest match image name.

def nn1(listoffnames, dist='L1'):
	"""Calculates the 1-nearest neighbor from a list of images.
		Returns a tuple of the image name and the nearest neighbor name.
		Note: The first image will act as the key image to compare the other images to.

		Parameters:\n
		names - An array of image paths.\n
		distanceType (optional) - A string selecting L1 or L2 distance to be used.
		If not specified, the default is L1.
	"""
	nearestNeighbor = [None, None]
	for i, value in enumerate(listoffnames):
		# Set key to 1st element and get its histogram for distance
		if i == 0:
			key = cv.imread(value)
			key = BGR_to_HSV(key)
			keyHistogram = compute_HS_histogram(key)
		# Import images 1 by 1, compute histogram, and compare distance to current lowest
		else:
			image = cv.imread(value)
			image = BGR_to_HSV(image)
			histogram = compute_HS_histogram(image)
			if dist == "L1":
				distance = L1(keyHistogram, histogram)
			elif dist == "L2":
				distance = L2(keyHistogram, histogram)
			else:
				raise Exception("Incorrect string value being passed into nn1(). Should be either L1 or L2.")
			if i == 1:
				nearestNeighbor[0] = value
				nearestNeighbor[1] = distance
			elif distance < nearestNeighbor[1]:
				nearestNeighbor[0] = value
				nearestNeighbor[1] = distance

	return (listoffnames[0], nearestNeighbor[0])

if runTests:
	print(nn1(["example-images/chicken.jpg", "example-images/cow.jpg", "example-images/kitten.jpg", "example-images/pig.jpg", "example-images/puppy1.jpg", "example-images/puppy2.jpg", "example-images/puppy3.jpg"]))

filelist = glob.glob('example-images/*.jpg')
# change to 'example-images\\*.jpg' on Windows

listoflists = []

for i in range(len(filelist)):
	#print(filelist[0])
	listoflists.append(filelist.copy())	# make a copy of the list of names and append it 
	save = filelist[0] 
	filelist.remove(filelist[0])
	filelist.append(save)	# these 3 lines move the first file name to the end
												# which makes the list have a different first element 

# listoflists now contains n versions of the filelist with
# a different file name as the first element (to act as the key)

for i in range(len(listoflists)):
	# debug code to make sure the lists really had a different first element
	# print("list", i)
	# for j in range(len(listoflists[i])):
	# 	print(listoflists[i][j])
	keyimgname, matchimgname = nn1(listoflists[i], 'L1')
	keyimg = cv.imread(keyimgname)
	matchimg = cv.imread(matchimgname)
	cv.imshow('key image', keyimg)
	cv.imshow('match image', matchimg)
	cv.waitKey(0)
	cv.destroyAllWindows()