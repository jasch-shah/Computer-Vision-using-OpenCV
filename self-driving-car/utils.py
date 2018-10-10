import cv2
import numpy as np
import matplotlib.image as mpimg

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def load_image(data_dir, image_file):
	# load rgb images from file

	return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def crop(image):
	# crop image
	return image[60:-25, :, :] # remove the sky and car front

def resize(image):
	# resize image to input shape
	return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
	# convert image ffrom rgb to yuv
	return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
	# combine all preprocess functions into one

	image = crop(image)
	image = resize(image)
	image = rgb2yuv(image)
	return image


def choose_image(data_dir, center, left, right, steering_angle):
	# randomly choose an image from center, left or right and adjust the steering angle
	choice = np.random.choice(3)
	if choice == 0:
		return load_image(data_dir, left), steering_angle + 0.2
	elif choice == 1:
		return load_image(data_dir, right), steering_angle - 0.2
	return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
	# random flip the image left <-> right, and adjust steering angle

    if np.random.rand() < 0.5:
	    image = cv2.flip(image, 1)
	    steering_angle -= steering_angle
    return image, steering_angle

def random_translate(image, steering_angle, range_x, range_y):
	# random shift the image virtually and horizontally
	trans_x = range_x * (np.random.rand() - 0.5)
	trans_y = range_y * (np.random.rand() - 0.5)
	steering_angle += trans_x * 0.002
	trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
	height, width = image.shape[:2]
	image = cv2.warpAffline(image, trans_m, (width, height))
	return image, steering_angle


def random_shadow(image):
	# generate and adds random shadows

	x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
	x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
	xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

	mask = np.zeros_like(image[:, :, 1])
	mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

	# choose which side should have shadow and adjust

	cond = mask == np.random.randint(2)
	s_ratio = np.random.uniform(low=0.2, high=0.5)

	# adjust saturation in HLS(Hue, Light, Saturation)
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
	return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
	# HSV is also called as HSB
	hsv = cv2.cvtColor(image, cv2.RGB2HSV)
	ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
	hsv[:,:,2] = hsv[:,:,2] * ratio


def augment(data_dir, center, left, right, steering_angle, range_x=100, range_y=100):
    # Generate augmented image and adjust steering angle

    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
    	i = 0
    	for index in np.random.permutation(image_paths.shape[0]):
    	    center, left, right = image_paths[index]
    	    steering_angle = steering_angles[index]
    	    # augmentation
    	    if is_training and np.random.rand() < 0.6:
    	        image, steering_angle = augment(data_dir, center, left, right, steering_angle)
    	    else:
    	        image = load_image(data_dir, center)

    	    # add image and steering angle to batch
    	    images[i] = preprocess(image)
    	    steers[i] = steering_angle
    	    i += 1
    	    if i == batch_size:
    	        break
    	yield images, steers                    	




