import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import multiclass
from random import shuffle
from random import random
from pprint import pprint
import pickle
import csv

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

def process_images_gabor_multilabel(path, stddev):
	class_labels = []
	class_names = []
	with open(path+'/classes.txt') as tsv:
		for line in csv.reader(tsv, delimiter='\t'):
			curr_class_label = [int(line[0])]
			class_names += [line[1]]
			for i in range(2,len(line)):
				curr_class_label += [int(line[i])]
			class_labels += [curr_class_label]
	print 'Beginning Gabor processing of images across', len(class_names), 'classes with multilabels...'
	i = 0
	image_data = []
	for class_name in class_names:
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
				image_data.append((gabor_feature,class_labels[i]))
				gabor_feature = gabor(new_image,stddev)
				image_data.append((gabor_feature,class_labels[i]))
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
	data = open('img_gabor_multilabel_data.pickle','w')
	pickle.dump((np.array(x), np.array(y)), data)
	data.close()
	print 'Image Processing Done'

def process_images_hex(path, num_labels):
	class_labels = []
	class_names = []
	with open(path+'/classes.txt') as tsv:
		for line in csv.reader(tsv, delimiter='\t'):
			curr_class_labels = [0 for i in range(num_labels)]
			curr_class_labels[int(line[0])-1] = 1
			class_names += [line[1]]
			for i in range(2,len(line)):
				curr_class_labels[int(line[i])-1] = 1
			class_labels.append(curr_class_labels)
	print 'Beginning processing of images across', len(class_names), 'classes with HEX (multilabels)...'
	i = 0
	image_data = []
	for class_name in class_names:
		class_path = path+'/'+class_name
		list_images = os.listdir(class_path)
		num_images = len(list_images)
		print 'Beginning processing of', num_images, 'images in class', class_name, i
		for image_name in list_images:
			image = cv2.imread(class_path+'/'+image_name, 0)
			shp = np.shape(image)
			max_l = max(shp)
			new_image = np.zeros((max_l,max_l), np.uint8) + 255
			new_image[0:shp[0],0:shp[1]] = image
			image = cv2.resize(new_image, (44,44), interpolation = cv2.INTER_AREA)
			image = np.reshape(image, (1936,))
			image_data.append((image,class_labels[i]))
		i += 1
	print 'Processing finished. Shuffling image data worth', len(image_data), 'tuples...'
	shuffle(image_data)
	x = []
	y = []
	for term in image_data:
		x.append(term[0])
		y.append(term[1])
	print 'Splitting 60-40 into training and validation data...'
	lx = int(round(0.6*len(x)))
	ly = int(round(0.6*len(y)))
	train_x = x[:lx]
	valid_x = x[lx+1:]
	train_y = y[:ly]
	valid_y = y[ly+1:]
	print 'Writing data to file...'
	data = open('train_hex.pickle','w')
	pickle.dump((np.array(train_x), np.array(train_y)), data)
	data.close()
	data = open('valid_hex.pickle','w')
	pickle.dump((np.array(valid_x), np.array(valid_y)), data)
	data.close()
	print 'Image Processing Done'

def process_images_gabor_hex(path, num_labels, stddev):
	class_labels = []
	class_names = []
	with open(path+'/classes.txt') as tsv:
		for line in csv.reader(tsv, delimiter='\t'):
			curr_class_labels = [0 for i in range(num_labels)]
			curr_class_labels[int(line[0])-1] = 1
			class_names += [line[1]]
			for i in range(2,len(line)):
				curr_class_labels[int(line[i])-1] = 1
			class_labels.append(curr_class_labels)
	print 'Beginning processing of images across', len(class_names), 'classes with HEX (multilabels)...'
	i = 0
	image_data = []
	for class_name in class_names:
		class_path = path+'/'+class_name
		list_images = os.listdir(class_path)
		num_images = len(list_images)
		print 'Beginning processing of', num_images, 'images in class', class_name, i
		for image_name in list_images:
			image = cv2.imread(class_path+'/'+image_name, 0)
			#rows, cols = image.shape
			#rotmat = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
			#new_image = cv2.warpAffine(image,rotmat,(cols,rows))
			try:
				gabor_feature = gabor(image,stddev)
				image_data.append((gabor_feature,class_labels[i]))
				#gabor_feature = gabor(new_image,stddev)
				#image_data.append((gabor_feature,class_labels[i]))
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
	print 'Splitting 60-40 into training and validation data...'
	#x = x[:5000] #smaller dataset
	#y = y[:5000]
	x = np.array(x)
	x = np.delete(x,np.where(np.std(x,0)==0),1) #removing attributes with 0 standard deviation
	x = np.divide((x - np.mean(x,0)),np.std(x,0))
	#print 'mean', np.mean(x,0), 'stddev', np.std(x,0)
	lx = int(round(0.6*len(x)))
	ly = int(round(0.6*len(y)))
	train_x = x[:lx]
	valid_x = x[lx+1:]
	train_y = y[:ly]
	valid_y = y[ly+1:]
	print 'Writing data to file...'
	data = open('train_gabor_hex.pickle','w')
	pickle.dump((np.array(train_x), np.array(train_y)), data)
	data.close()
	data = open('valid_gabor_hex.pickle','w')
	pickle.dump((np.array(valid_x), np.array(valid_y)), data)
	data.close()
	print 'Image Processing Done'

def multilabel_svm_analysis(image_data_file, training_percent, C_value, gamma_value):
	print 'Loading data from file...'
	fd = open(image_data_file, 'r')
	data = pickle.load(fd)
	x_unproc, y_unproc = data
	x = preprocessing.scale(x_unproc)
	binzr = preprocessing.MultiLabelBinarizer()
	y = binzr.fit_transform(y_unproc)
	num_training = round(len(x)*training_percent)
	x_train = x[:num_training]
	x_test = x[num_training:]
	y_train = y[:num_training]
	y_test = y[num_training:]
	print 'Training', len(x_train), 'data samples on SVM...'
	estimator = svm.SVC(C=C_value, cache_size=1000, gamma=gamma_value, kernel='rbf', probability=True)
	classifier = multiclass.OneVsRestClassifier(estimator, n_jobs = -1)
	classifier.fit(x_train, y_train)
	print 'Calculating test probabilities...'
	probs = classifier.predict_proba(x_test)
	any_score = 0
	all_score = 0
	all_labels = binzr.inverse_transform(y_test)
	print 'Calculating test scores...'
	for i in range(len(probs)):
		labels = sorted(all_labels[i])
		top_k_classes = sorted(sorted(range(len(probs[i])), key=lambda j: probs[i][j])[-len(labels):])
		top_k_classes = [k+1 for k in top_k_classes]
		if (len(set(labels).intersection(top_k_classes))>0):
			any_score += 1
		if(cmp(labels,top_k_classes)==0):
			all_score += 1
		print (labels,top_k_classes)
	any_score = float(any_score)/len(y_test)
	all_score = float(all_score)/len(y_test)
	print 'AnyClassMatches score on test data is', any_score, 'and AllClassesMatch score on test data is', all_score
	print 'SVM Analysis Done'

def find_legal_ground_truths(path):
	legal = []
	with open(path+'/classes.txt') as tsv:
		for line in csv.reader(tsv, delimiter='\t'):
			curr_truth = []
			for i in range(2,len(line)):
				curr_truth += [int(line[i])-1]	#'-1' because counting begins at 0
			curr_truth += [int(line[0])-1]
			legal.append(curr_truth)
	data = open('img_lgt.pickle','w')
	pickle.dump(legal, data)
	data.close()

def hex_stochastic_gradient(f, y, legal_ground_truths): #LGTs must contain lists sorted in increasing order; remember that counting starts at 0
	y_vals = []
	for i in range(len(y)):
		if(y[i]==1):
			y_vals += [i]
	if y_vals in legal_ground_truths:
		partition_func = 0
		marg_probs = [0 for i in range(len(f))]
		cond_probs = [[0 for i in range(len(f))] for i in range(len(f))]
		for x in legal_ground_truths:
			new_score = 0
			for i in x:
				new_score += f[i]
			new_score = np.exp(new_score)
			for i in x:
				marg_probs[i] += new_score
				for j in x:
					cond_probs[i][j] += new_score
		marg_probs = np.array(marg_probs)
		marg_probs = marg_probs/float(sum(marg_probs))
		for i in range(len(f)):
			cond_probs[i] = np.array(cond_probs[i])
			cond_probs[i] = cond_probs[i]/float(sum(cond_probs[i]))
		#print marg_probs
		#print cond_probs
		gradients = [0 for i in range(len(f))]
		for i in range(len(gradients)):
			for y in y_vals:
				gradients[i] += cond_probs[y][i] - marg_probs[i]
		return gradients			
	else:
		print 'Illegal ground truth value', y_vals
		return False

def hex_predict_prob(f, legal_ground_truths): #LGTs must contain lists sorted in increasing order; remember that counting starts at 0
	probs = [0 for i in range(len(f))]
	for x in legal_ground_truths:
		new_score = 0
		for i in x:
			new_score += f[i]
		new_score = np.exp(new_score)
		for i in x:
			probs[i] += new_score
	probs = np.array(probs)
	probs = probs/float(sum(probs))
	return probs

def view_confmat(mat):
	plt.imshow(mat)
	plt.show()

#process_images_gabor('/home/sahil/Documents/minip_plankton/train',8)
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
#process_images_gabor_multilabel('/home/sahil/Documents/minip_plankton/train',8)
#multilabel_svm_analysis('img_gabor_multilabel_data.pickle',0.6,8,0.1)

#lgt = [[0],[0,1],[0,2],[0,1,2]]
#f = [1,1,1]
#y = [1,1,1]
#print hex_stochastic_gradient(f, y, lgt)
#print hex_predict_prob(f, lgt)




#################################              ANN STUFF HERE            ##########################################

def layerOpANN( layer, x ):
    sz = len(layer)
    o = []
    for i in range(0,sz):
        e = np.exp( -1*np.dot( layer[i], x ) )
        o1 = 1.0/(1.0 + e )
        o.append(o1)
    return np.array(o)

#if numHiddenLayers are N then sizeOfLayers should have N+2 elements
# and layers will output N+1 matrices.
def initANN( numHiddenLayer, sizeOfLayer ):
    layers = []
    steps = []
    for i in range(0,numHiddenLayer+1):
        a = np.random.normal(0,1,(sizeOfLayer[i+1], (sizeOfLayer[i])))
	b = np.zeros((sizeOfLayer[i+1],sizeOfLayer[i]))
        layers.append(a)
	steps.append(b)
    return (layers, steps)

def backPropANN( y, o, layerIndex, dell, prevLayer=None ):
    global lgt
    if layerIndex == 0:
        dell[ layerIndex ] = np.array(hex_stochastic_gradient(o, y, lgt))
        #dell[ layerIndex ] = o*(1-o)*(y-o)
    elif prevLayer.any():
        dl1 = dell[ layerIndex-1 ]
        if layerIndex == 1:
            shp1 = np.shape(dl1)
            shp2 = np.shape(prevLayer)
            prevL = prevLayer[0:shp2[0],0:(shp2[1])]
            layerMemory = np.dot( dl1, prevL )
            dell[ layerIndex ] = o*(1-o)*layerMemory
        else:
            shp1 = np.shape(dl1)	
            shp2 = np.shape(prevLayer)
            prevL = prevLayer[0:shp2[0],0:(shp2[1])]
            layerMemory = np.dot( dl1, prevL )
            dell[ layerIndex ] = o*(1-o)*layerMemory
    return dell

def predictANN( layers, x ):
    global lgt
    inpt = x
    n = len(layers)
    outpt = 0.0
    for i in range(0,n):
        outpt = layerOpANN( layers[i], inpt )
        inpt = outpt
    #outpt = np.array(outpt/ ( np.linalg.norm(outpt) ) )
    outpt = hex_predict_prob(outpt, lgt)
    return np.array(outpt)

def updateLayersANN( layers, prevsteps, dell, o, x, eta, mu ):
    numLayers = len( layers )
    for i in range(0,numLayers):
        layer = layers[i]
        delLayer = dell[ (numLayers-1) - i ]
        if i == 0:
            deltaW = np.multiply( np.array([delLayer]).T, np.array([x]) )
            step = eta * deltaW + mu * prevsteps[i]
            layer = np.add(layer,step)
            layers[i] = layer
	    prevsteps[i] = step
        else:
            deltaW = np.multiply( np.array([delLayer]).T, np.array([o[i-1]]) )
            step = eta * deltaW + mu * prevsteps[i]
            #layerOld = layer
            layer = np.add(layer,step)
            #print np.linalg.norm(layer - layerOld)
            layers[i] = layer
	    prevsteps[i] = step
    return (layers, prevsteps)

def trainANN( data, eta, mu, epsilon, iterations, m, n ):
    f = open( data, 'r' )
    w = pickle.load(f)
    f.close()
    xX,yY = w
    nEx = np.shape(xX)[0]
    (layers, prevs) = initANN( n, m )
    old_err = np.Inf
    err = 0
    errs = []
    itr = 1
    dup_ack_2 = 0
    dup_ack_5 = 0
    while np.abs(err-old_err) >= epsilon and itr<=iterations:
        dO = 0;
        for i in range(0,nEx):
            #if (i%1000==0):
            #    print 'iteration', itr, 'example', i
            ( x , y ) = (xX[i],yY[i])
            #x = np.resize(x,(16,))
            #x = (map(float,x)) / (np.linalg.norm(x))
            #y1 = [ 0 for i in range(0,121)]
            #y1[y] = 1
            #y = y1
            dell = [ [] for k in range(0,(n+1)) ]
            o = [ [] for k in range(0,(n+1)) ]
            for j in range(0,(n+1)):
                if j==0:
                    o[j] = layerOpANN( layers[j], x )
                else:
                    o[j] = layerOpANN( layers[j], o[j-1] )
            for j in range(0,(n+1)):
                if j==0:
                    dell = backPropANN( y, o[n-j], j, dell )
                else:
                    dell = backPropANN( y, o[n-j], j, dell, layers[n+1-j] )
            (layers, prevs) = updateLayersANN( layers, prevs, dell, o, x, eta/(np.sqrt(itr)), mu/(np.sqrt(itr)) )
            #dO = np.linalg.norm( np.subtract( hex_predict_prob(y, lgt), hex_predict_prob(o[n], lgt) ) ) + dO
            dO = np.linalg.norm( np.subtract( y, o[n] ) ) + dO
	old_err = err
        err = dO/nEx
	print 'iteration', itr, 'error', err
	errs += [err]
	if (old_err<err): #TCP-inspired variation of eta
		dup_ack_5 = 0
		dup_ack_2 = dup_ack_2 + 1
		if (dup_ack_2==2):
			dup_ack_2 = 0
			eta = 0.5*eta
			print 'double ack, -, eta', eta
	else:
		dup_ack_2 = 0
		dup_ack_5 = dup_ack_5 + 1
		if (abs(old_err-err)>=10*epsilon and dup_ack_5==5):
			dup_ack_5 = 0
			eta = 1.25*eta
			print 'pentaple ack, +, eta', eta
        itr = itr + 1
    a = ( layers, dell, errs )
    return a

def getScoreANN( layers, data ):
    """
    :type layers: type np.array of np.array
    :param layers: layers learnt by the ANN
    """
    global lgt
    f = open( data, 'r' )
    w = pickle.load(f)
    f.close()
    xX,yY = w
    nEx = (np.shape(xX))[0]
    i = 0
    acc = 0
    score = 0
    confMat = np.zeros([121,121])
    for i in range(0,nEx):
        #if (i%1000==0):
        #    print 'example', i
        ( x , y ) = (xX[i],yY[i])
        #x = np.resize(x,(16,))
        #y1 = [ 0 for i in range(0,121)]
        #y1[y] = 1
        #y = y1
        p = predictANN(layers,x)
        #p = np.array( p / ( np.linalg.norm(p) ) )
        p = np.array(map(np.log,p))
        y = np.array(y)
	y = hex_predict_prob(y, lgt)
	p = p[19:] #we don't want to just match superclasses!
	y = y[19:]
        confMat[p.argmax(),y.argmax()] = confMat[p.argmax(),y.argmax()] + 1
        if (p.argmax() == y.argmax()):
            acc = acc + 1.0
        score = score + np.dot(p,y)
    score = score/nEx
    acc = acc/nEx
    return (score, acc, confMat)

#################################	HARSH, EXECUTING STUFF HERE!	####################################
find_legal_ground_truths('/home/sahil/Documents/minip_plankton/train') #pathname "<cwd>", ensuring "classes.txt" is inside this directory as well.
process_images_gabor_hex('/home/sahil/Documents/minip_plankton/train', 140, 8) #pathname to "train" folder
f = open( '/home/sahil/Documents/minip_plankton/img_lgt.pickle', 'r' ) #pathname "<cwd>/img_lgt.pickle"
lgt = pickle.load(f)
f.close()

def try_ann():
	etas = [0.1]	# learning rate
	mus = [0.1]	# momentum rate
	hidn = [32]	# number of hidden layers to test for
	data = []
	for i in etas:
		for j in hidn:
			for l in mus:
				for k in range(1):
					print 'try', k, 'eta', i, 'mu', l, 'hidden', j
					(layers, delta, errs) = trainANN( '/home/sahil/Documents/minip_plankton/train_gabor_hex.pickle', i, l, 0.00000001, 200, [66,j,140], 1 ) #pathname "<cwd>/train_gabor_hex.pickle"
					(score, acc, confMat) = getScoreANN(layers, '/home/sahil/Documents/minip_plankton/valid_gabor_hex.pickle') #pathname "<cwd>/valid_gabor_hex.pickle"
					print 'score', score, 'accuracy', acc
					data.append((i, j, layers, delta, errs, score, acc, confMat))
	f = open('/home/sahil/Documents/minip_plankton/ann_gabor_hex_momentum_88.pickle','w') #pathname "<cwd>/<any_name>.pickle"
	pickle.dump(data,f)
	f.close()

try_ann()
