import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn import preprocessing
from sklearn import linear_model
from random import shuffle
from random import random
from pprint import pprint
import pickle

def show(image):
	cv2.imshow('Image', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def threshold(image, thresh):
	return cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]

def neg_threshold(image, thresh):
	return cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY_INV)[1]

def area(image):
	(height, width) = image.shape
	ar = height*width
	for i in range(height):
		for j in range(width):
			if (image[i][j] != 0):
				ar -= 1
	return ar

def perimeter(image):
	new_image = cv2.Canny(image, 100, 200)
	peri = 0
	(height, width) = new_image.shape
	for i in range(height):
		for j in range(width):
			if (new_image[i][j] != 0):
				peri += 1
	return peri

def circularity(path):
	list_imgs = os.listdir(path)
	num = len(list_imgs)
	print 'Finding circularity constants for', num, 'images...'
	circu = [0 for i in range(num)]
	i = 0
	for img in list_imgs:
		image = threshold(cv2.imread(path+'/'+img, 0), 230)
		ar = area(image)
		peri = perimeter(image)
		circu[i] = (peri*peri)/(4.0*np.pi*ar)
		i += 1
	return circu

def low_pass_filter(signal):
	y = np.fft.fft(signal)
	x = [y[i] for i in range(10)] + [0]*(len(y)-10)
	y = np.fft.ifft(x)
	return y

def frequency(signal):
	return abs(np.fft.fft(signal))

def hu_moments(image, thresh):
	new_image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY_INV)[1]
	return cv2.HuMoments(cv2.moments(new_image)).flatten()

def hog(image):
	gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
	(mag, ang) = cv2.cartToPolar(gx, gy)
	bins = np.int32(16*ang/(2*np.pi))
	bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
	mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
	hists = [np.bincount(b.ravel(), m.ravel(), 16) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)
	return hist

def gabor(image, stddev):
	thetas = [0,30,60,90,120,150,180]
	lambdas = [1,2,4,8,16]
	feature_vector = []
	for i in lambdas:
		for j in thetas:
			gabmat = cv2.getGaborKernel((5,5),stddev,j,i,1)
			new_image = cv2.filter2D(image,-1,gabmat)
			feature_vector += [np.mean(new_image), np.std(new_image)]
	return feature_vector

def dct(image, size):
	new_image = np.float32(image)/255.0 
	new_image = cv2.dct(cv2.resize(new_image,(64,64)))
	vect = []
	j = 0
	while (j<=size):
		k = 0
		while (k<=j):
			vect += [new_image[k][j-k]]
			k += 1
		j += 1
	return vect

#h1 = hu_moments('/home/sahil/Documents/plankton/train/chaetognath_non_sagitta', 5)
#h2 = hu_moments('/home/sahil/Documents/plankton/train/chaetognath_sagitta', 5)
#c1 = circularity('/home/sahil/Documents/plankton/train/chaetognath_non_sagitta')
#c2 = circularity('/home/sahil/Documents/plankton/train/chaetognath_sagitta')
#plt.plot(low_pass_filter(h1+h2))
#plt.show()

#image = cv2.imread('/home/sahil/Desktop/plankton_kaggle/plankton.jpg', 0)
#print gabor(image, 1)
#show(image);
#print hog(image)

#classifier = svm.SVC(cache_size = 1000, kernel = 'rbf', probability=True)
#classifier.fit(x, y)
#classifier.predict(x)

def process_images_hog(path):
	list_classes = os.listdir(path)
	num_classes = len(list_classes)
	print 'Beginning HOG processing of images across', num_classes, 'classes...'
	i = 1
	image_data = []
	for class_name in list_classes:
		class_path = path+'/'+class_name
		list_images = os.listdir(class_path)
		num_images = len(list_images)
		print 'Beginning processing of', num_images, 'images in class', class_name, i
		for image_name in list_images:
			image = cv2.imread(class_path+'/'+image_name, 0)
			rows, cols = image.shape
			rotmat = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
			new_image = cv2.warpAffine(image,rotmat,(cols,rows))
			try:
				hog_hist = hog(image)
				image_data.append((hog_hist,i))
				hog_hist = hog(new_image)
				image_data.append((hog_hist,i))
			except:
				print 'Error in evaluating HOG descriptors for an image in class', class_name, i
		i += 1
	print 'Processing finished. Shuffling image data worth', len(image_data), 'tuples...'
	shuffle(image_data)
	x = []
	y = []
	for term in image_data:
		x.append(term[0])
		y.append(term[1])
	print 'Writing data to file...'
	data = open('img_hog_data.pickle','w')
	pickle.dump((np.array(x), np.array(y)), data)
	data.close()
	print 'Image Processing Done'

def process_images_gabor(path, stddev):
	list_classes = os.listdir(path)
	num_classes = len(list_classes)
	print 'Beginning Gabor processing of images across', num_classes, 'classes...'
	i = 1
	image_data = []
	for class_name in list_classes:
		class_path = path+'/'+class_name
		list_images = os.listdir(class_path)
		num_images = len(list_images)
		print 'Beginning processing of', num_images, 'images in class', class_name, i
		for image_name in list_images:
			image = cv2.imread(class_path+'/'+image_name, 0)
			rows, cols = image.shape
			rotmat = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
			new_image = cv2.warpAffine(image,rotmat,(cols,rows))
			try:
				gabor_feature = gabor(image,stddev)
				image_data.append((gabor_feature,i))
				gabor_feature = gabor(new_image,stddev)
				image_data.append((gabor_feature,i))
			except:
				print 'Error in evaluating Gabor-filtered image for an image in class', class_name, i
		i += 1
	print 'Processing finished. Shuffling image data worth', len(image_data), 'tuples...'
	shuffle(image_data)
	x = []
	y = []
	for term in image_data:
		x.append(term[0])
		y.append(term[1])
	print 'Writing data to file...'
	data = open('img_gabor_data.pickle','w')
	pickle.dump((np.array(x), np.array(y)), data)
	data.close()
	print 'Image Processing Done'

def process_images_hu(path, thresh):
	list_classes = os.listdir(path)
	num_classes = len(list_classes)
	print 'Beginning Hu Moments processing of images across', num_classes, 'classes...'
	i = 1
	image_data = []
	for class_name in list_classes:
		class_path = path+'/'+class_name
		list_images = os.listdir(class_path)
		num_images = len(list_images)
		print 'Beginning processing of', num_images, 'images in class', class_name, i
		for image_name in list_images:
			image = cv2.imread(class_path+'/'+image_name, 0)
			rows, cols = image.shape
			rotmat = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
			new_image = cv2.warpAffine(image,rotmat,(cols,rows))
			try:
				hu = hu_moments(image, thresh)
				image_data.append((hu,i))
				hu = hu_moments(new_image, thresh)
				image_data.append((hu,i))
			except:
				print 'Error in evaluating Hu Moments for an image in class', class_name, i
		i += 1
	print 'Processing finished. Shuffling image data worth', len(image_data), 'tuples...'
	shuffle(image_data)
	x = []
	y = []
	for term in image_data:
		x.append(term[0])
		y.append(term[1])
	print 'Writing data to file...'
	data = open('img_hu_data.pickle','w')
	pickle.dump((np.array(x), np.array(y)), data)
	data.close()
	print 'Image Processing Done'

def process_images_dct(path, size):
	list_classes = os.listdir(path)
	num_classes = len(list_classes)
	print 'Beginning DCT processing of images across', num_classes, 'classes...'
	i = 1
	image_data = []
	for class_name in list_classes:
		class_path = path+'/'+class_name
		list_images = os.listdir(class_path)
		num_images = len(list_images)
		print 'Beginning processing of', num_images, 'images in class', class_name, i
		for image_name in list_images:
			image = cv2.imread(class_path+'/'+image_name, 0)
			rows, cols = image.shape
			rotmat = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
			new_image = cv2.warpAffine(image,rotmat,(cols,rows))
			try:
				dct_feature = dct(image,size)
				image_data.append((dct_feature,i))
				dct_feature = dct(new_image,size)
				image_data.append((dct_feature,i))
			except:
				print 'Error in evaluating DCT-filtered image for an image in class', class_name, i
		i += 1
	print 'Processing finished. Shuffling image data worth', len(image_data), 'tuples...'
	shuffle(image_data)
	x = []
	y = []
	for term in image_data:
		x.append(term[0])
		y.append(term[1])
	print 'Writing data to file...'
	data = open('img_dct_data.pickle','w')
	pickle.dump((np.array(x), np.array(y)), data)
	data.close()
	print 'Image Processing Done'

def svm_analysis(image_data_file, training_percent, C_value, gamma_value):
	print 'Loading data from file...'
	fd = open(image_data_file, 'r')
	data = pickle.load(fd)
	x_unproc, y = data
	x = preprocessing.scale(x_unproc)
	num_training = round(len(x)*training_percent)
	x_train = x[:num_training]
	x_test = x[num_training:]
	y_train = y[:num_training]
	y_test = y[num_training:]
	print 'Training', len(x_train), 'data samples on SVM...'
	classifier = svm.SVC(C=C_value, cache_size=1000, gamma=gamma_value, kernel='rbf', probability=False)
	classifier.fit(x_train, y_train)
#	print 'Testing', len(x_test), 'data samples on SVM...'
#	y_predicted = classifier.predict(x_test)
#	plt.plot(y_test, y_predicted, 'r*')
#	plt.show()
	print 'Calculating test score...'
	score = classifier.score(x_test, y_test)
	print 'Score on test data is', score
	print 'SVM Analysis Done'

def lin_svm_analysis(image_data_file, training_percent, C_value):
	print 'Loading data from file...'
	fd = open(image_data_file, 'r')
	data = pickle.load(fd)
	x_unproc, y = data
	x = preprocessing.scale(x_unproc)
	num_training = round(len(x)*training_percent)
	x_train = x[:num_training]
	x_test = x[num_training:]
	y_train = y[:num_training]
	y_test = y[num_training:]
	print 'Training', len(x_train), 'data samples on Linear SVM...'
	classifier = svm.LinearSVC(C=C_value)
	classifier.fit(x_train, y_train)
#	print 'Testing', len(x_test), 'data samples on Linear SVM...'
#	y_predicted = classifier.predict(x_test)
#	plt.plot(y_test, y_predicted, 'r*')
#	plt.show()
	print 'Calculating test score...'
	score = classifier.score(x_test, y_test)
	print 'Score on test data is', score
	print 'Linear SVM Analysis Done'

def log_reg_analysis(image_data_file, training_percent, C_value):
	print 'Loading data from file...'
	fd = open(image_data_file, 'r')
	data = pickle.load(fd)
	x_unproc, y = data
	x = preprocessing.scale(x_unproc)
	num_training = round(len(x)*training_percent)
	x_train = x[:num_training]
	x_test = x[num_training:]
	y_train = y[:num_training]
	y_test = y[num_training:]
	print 'Training', len(x_train), 'data samples on Logistic Regression Model...'
	classifier = linear_model.LogisticRegression(C=C_value)
	classifier.fit(x_train, y_train)
#	print 'Testing', len(x_test), 'data samples on SVM...'
#	y_predicted = classifier.predict(x_test)
#	plt.plot(y_test, y_predicted, 'r*')
#	plt.show()
	print 'Calculating test score...'
	score = classifier.score(x_test, y_test)
	print 'Score on test data is', score
	print 'Logistic Regression Analysis Done'

def num_images(path):
	list_classes = os.listdir(path)
	image_data = []
	for class_name in list_classes:
		class_path = path+'/'+class_name
		image_data += [len(os.listdir(class_path))]
	return image_data

#process_images_gabor('/home/sahil/Documents/plankton/train',8)
#svm_analysis('img_gabor_data.pickle',0.6,8,0.1)
#process_images_hog('/home/sahil/Documents/plankton/train')
#svm_analysis('img_hog_data.pickle',0.6,8,4)
#process_images_dct('/home/sahil/Documents/plankton/train',14)
#svm_analysis('img_dct_data.pickle',0.6,128,4)
#log_reg_analysis('img_dct_data.pickle',0.6,16)
#x = num_images('/home/sahil/Documents/plankton/train')
#pprint (len(x))
#plt.plot(np.sort(x))
#plt.show()
#pprint((dct(cv2.imread('plankton.jpg',0),14)))

