#Artificial Neural Network

import os
import numpy as np
import cv2
import pickle
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from pylab import *
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from random import shuffle
from pprint import pprint
import pickle

def processImages():
    classNames = open('classNames.txt','r')
    data = open('data.pickle','w')
    itr = 0
    mem = []
    for folderName in classNames:
        folderName = (folderName.split('\n'))[0]
        y = itr
        print folderName
        imgNames = [ f for f in os.listdir(folderName) ]
        for img in imgNames:
            if 'o' not in img:
                i = cv2.imread(folderName+'/'+img+'')
                shp = np.shape(i)
                max_l = max(shp)
                img = np.zeros((max_l,max_l,3), np.uint8) + 255
                img[ 0:shp[0], 0:shp[1], 0:shp[2] ] = i
                i = cv2.resize( img,( 44, 44 ), interpolation = cv2.INTER_AREA)
                i = ( ( i[:,:,0]/3.0) + (i[:,:,1]/3.0) + (i[:,:,2] /3.0) )
                i = np.reshape( i, (44*44,) )
                mem.append( (i,y) )
        itr = itr + 1
    random.shuffle(mem)
    memx = []
    memy = []
    for e in mem:
        memx.append( e[0] )
        memy.append( e[1] )
    pickle.dump( (np.array(memx),np.array(memy)) , data )
    data.close()
    print 'done'

def makeSmallerDS( name, size, data ):
    f = open(data,'r')
    dataset = pickle.load( f )
    x, y = dataset
    x = np.array(x)
    xSmall = x[0:size,:]
    ySmall = y[0:size]
    fS = open(name+'.pickle','w')
    pickle.dump([xSmall,ySmall],fS)
    fS.close()

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
    for i in range(0,numHiddenLayer+1):
        a = np.random.normal(0,1,(sizeOfLayer[i+1], (sizeOfLayer[i])))
        layers.append(a)
    return layers

def backPropANN( y, o, layerIndex, dell, prevLayer=None ):
    if layerIndex == 0:
        dell[ layerIndex ] = o*(1-o)*(y-o)
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
    inpt = x
    n = len(layers)
    outpt = 0.0
    for i in range(0,n):
        outpt = layerOpANN( layers[i], inpt )
        inpt = outpt
    return np.array(outpt)

#parse the input file and convert it into correct format
def parse( l,sz ):
    lSplit = l.split(',')
    y1 = int(lSplit.pop())
    y = [ 0 for i in range(0,sz) ]
    y[y1] = 1
    y = np.array(y)
    x = map(float,lSplit)
    x = np.array(x)
    return ( x , y )

def updateLayersANN( layers, dell, o, x, eta ):
    numLayers = len( layers )
    for i in range(0,numLayers):
        layer = layers[i]
        delLayer = dell[ (numLayers-1) - i ]
        if i == 0:
            deltaW = np.multiply( np.array([delLayer]).T, np.array([x]) )
            step = eta * deltaW
            layer = np.add(layer,step)
            layers[i] = layer
        else:
            deltaW = np.multiply( np.array([delLayer]).T, np.array([o[i-1]]) )
            step = eta * deltaW
            layerOld = layer
            layer = np.add(layer,step)
            layers[i] = layer
    return layers

def trainANN( data, eta, epsilon, iterations, m, n ):
    f = open( data, 'r' )
    w = pickle.load(f)
    xX,yY = w
    nEx = np.shape(xX)[0]
    layers = initANN( n, m )
    err = np.Inf
    itr = 1
    while err >= epsilon and itr<=iterations:
        dO = 0;
        for i in range(0,nEx):
            ( x , y ) = (xX[i],yY[i])
            x = np.resize(x,(16,))
            #x = (map(float,x)) / (np.linalg.norm(x))
            y1 = [ 0 for i in range(0,121)]
            y1[y] = 1
            y = y1
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
            layers = updateLayersANN( layers, dell, o, x, eta/(np.sqrt(itr)) )
            dO = np.linalg.norm( np.subtract( y, o[n] ) ) + dO
        err = dO/(n+1)
        itr = itr + 1
    a = ( layers, dell )
    return a

def svmANN( data, layers ):
    f = open( data, 'r' )
    w = pickle.load(f)
    xX,yY = w
    nEx = (np.shape(xX))[0]
    i = 0
    vecs = []
    for i in range(0,nEx):
        x  = xX[i]
        x = np.resize(x,(16,))
        p = predictANN( layers, x )
        vecs.append(p)
    clf=svm.SVC(kernel='rbf', C=128)
    clff=clf.fit(vecs,yY)
    return clff

def testSvmANN( data, layers, clff ):
    f = open( data, 'r' )
    w = pickle.load(f)
    xX,yY = w
    nEx = (np.shape(xX))[0]
    i = 0
    acc = 0
    for i in range(0,nEx):
        ( x , y ) = (xX[i],yY[i])
        x = np.resize(x,(16,))
        x1 = predictANN( layers, x )
        p = clff.predict(x1)[0]
        if p == y:
            acc = acc + 1
    acc1 =  (acc*1.0)/nEx
    return acc, acc1


    

# The function gets the score of the ANN model trained. It returns two values, number of correct predictions and log loss score.
def getScoreANN( layers, data, outputFeatures ):
    """
    :type layers: type np.array of np.array
    :param layers: layers learnt by the ANN
    """
    f = open( data, 'r' )
    w = pickle.load(f)
    xX,yY = w
    nEx = (np.shape(xX))[0]
    i = 0
    acc = 0
    score = 0
    confMat = np.zeros([121,121])
    for i in range(0,nEx):
        ( x , y ) = (xX[i],yY[i])
        x = np.resize(x,(16,))
        y1 = [ 0 for i in range(0,121)]
        y1[y] = 1
        y = y1
        p = predictANN(layers,x)
        p = np.array( p / ( np.linalg.norm(p) ) )
        p = np.array(map(np.log,p))
        y = np.array(y)
        confMat[p.argmax(),y.argmax()] = confMat[p.argmax(),y.argmax()] + 1
        if (p.argmax() == y.argmax()):
            acc = acc + 1.0
        score = score + np.dot(p,y)
    score = score / nEx
    acc = acc
    return score, acc, confMat
                
                    
#------------------------------------------
#Convolutional Neural Network

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

def evaluate(learning_rate=0.1, n_epochs=1,
                    dataset='data.pickle',
                    nkerns=[20, 50, 60], batch_size=500):
    """ 
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)
    data = open(dataset,'r')
    datasets = pickle.load(data)
    print "dataset loaded..."

    train_set_x, train_set_y = shared_dataset(datasets)
    valid_set_x, valid_set_y = shared_dataset(datasets)
    test_set_x, test_set_y = shared_dataset(datasets)

    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (44, 44)  # this is the size of MNIST images

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 44, 44))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (44-5+1,44-5+1)=(40,40)
    # maxpooling reduces this further to (40/2,40/2) = (20,20)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],20,20)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 44, 44),
            filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (20-5+1,20-5+1)=(16,16)
    # maxpooling reduces this further to (16/2,16/2) = (8,8)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],8,8)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 20, 20),
            filter_shape=(nkerns[1], nkerns[0], 3, 3), poolsize=(2, 2))

    layer2 = LeNetConvPoolLayer(rng, input=layer1.output,
            image_shape=(batch_size, nkerns[1], 9, 9),
            filter_shape=(nkerns[2], nkerns[1], 2, 2), poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer3_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(rng, input=layer3_input, n_in=nkerns[2] * 4 * 4,
                         n_out=1024, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=1024, n_out=121)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)
    
    print index
    print batch_size

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], [ layer4.errors(y), layer4.negative_log_likelihood(y) ], givens={ x: test_set_x[index * batch_size: (index + 1) * batch_size], y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], [ layer4.errors(y), layer4.negative_log_likelihood(y) ], givens={ x: valid_set_x[index * batch_size: (index + 1) * batch_size], y: valid_set_y[index * batch_size: (index + 1) * batch_size]})
    
    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by SGD. Since this model has many parameters, it would be tedious to manually create an update rule for each model parameter. We thus create the updates list by automatically looping over all (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2) # go through this many minibatches before checking the network on the validation set; in this case we check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                validation_losses_acc = [validation_losses[i][0] for i
                                     in xrange(n_valid_batches)]
                validation_log_losses = [validation_losses[i][1] for i
                                     in xrange(n_valid_batches)]
                print validation_losses
                this_validation_loss = numpy.mean(validation_losses_acc)
                this_validation_log_loss = numpy.mean(validation_log_losses)
                print('epoch %i, minibatch %i/%i, validation cost %f, validation mis-accuracy %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_log_loss, this_validation_loss*100. ))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_losses = [test_model(i) for i
                                     in xrange(n_test_batches)]
                    test_losses_acc = [test_losses[i][0] for i
                                         in xrange(n_test_batches)]
                    test_log_losses = [test_losses[i][1] for i
                                         in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses_acc)
                    print(('epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    return (layer0,layer1,layer2,layer3,layer4)



#-----------------------------------------------------------------------------------------------------------

def siftFV(imgName, folder='..'):
    img = cv2.imread(folder+'/'+imgName)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp = sift.detect(gray,None)
    #img=cv2.drawKeypoints(gray,kp)
    #cv2.imwrite(folder+'/'+'sift_'+imgName,img)
    return kp

def processImgsToSift():
    classNames = open('classNames.txt','r')
    data = open('dataSIFTed.pickle','w')
    itr = 0
    mem = []
    for folderName in classNames:
        folderName = (folderName.split('\n'))[0]
        y = itr
        print folderName
        imgNames = [ f for f in os.listdir(folderName) ]
        for img in imgNames:
            if 'o' not in img:
                kp = siftFV(img,folderName)
                imgKP = []
                for v in kp:
                    w = [ v.pt, v.size ]
                    imgKP.append(w)
                mem.append( (imgKP,y) )
        itr = itr + 1
    random.shuffle(mem)
    memx = []
    memy = []
    for e in mem:
        memx.append( e[0] )
        memy.append( e[1] )
    pickle.dump( (memx,memy) , data )
    data.close()
    print 'done'

def distance(x,cpj):
    d = np.sqrt( np.square( x[0] - cpj[0] ) + np.square( x[1] - cpj[1] ) )
    return d

def centroid( clusteri ):
    sz = len(clusteri)
    x = 0
    y = 0
    sm = 0
    for i in range(0,sz):
        x = x + (clusteri[i][0][0]*clusteri[i][1])
        y = y + (clusteri[i][0][1]*clusteri[i][1])
        sm = sm + clusteri[i][1]
    if sz:
        x = float(x)/sm
        y = float(y)/sm
    else:
        return []
    return np.array([x,y])

def argmaxim( k, cp, clusters ):
    for i in range(0,k):
        a = centroid(clusters[i])
        if len(a):
            cp[i] = a
    return np.array(cp)

def kMeans( k, cp, xX, itrMax ):
    for iter in range(0,itrMax):
        clusters = [[] for i in range(0,k)]
        nEx = (np.shape(xX))[0]
        for i in range(0,nEx):
            sz = len(xX[i])
            for p in range(0,sz):
                x = xX[i][p]
                dA = np.array( [ distance(x[0],cp[j,:]) for j in range(0,k) ] )
                kMin = dA.argmin()
                clusters[kMin].append(x)
        cp = argmaxim( k, cp, clusters )
    return cp

def getBOWdimension( data, k ):
    print 'Calculating bag of Words centroid representative points...'
    f = open(data,'r')
    w = pickle.load(f)
    xX, yY = w
    nEx = (len(xX))
    i = 0
    mxX1 = 0
    mnX1 = np.Inf
    mxX2 = 0
    mnX2 = np.Inf
    for i in range(0,nEx):
        sz = len(xX[i])
        for j in range(0,sz):
            point = xX[i][j][0]
            mxX1 = max(mxX1,point[0])
            mnX1 = min(mnX1,point[0])
            mxX2 = max(mxX2,point[1])
            mnX2 = min(mnX2,point[1])
    print mxX1,mnX1,mxX2,mnX2
    centroidsX1 = np.random.uniform(mnX1*2,mxX1/2,k)
    centroidsX2 = np.random.uniform(mnX2*2,mxX2/2,k)
    cp = np.array( [centroidsX1,centroidsX2] )
    cp = cp.T
    cp = kMeans( k, cp, xX, 5 )
    print 'Done!!!'
    return cp

def getIMGvec( bowVec, img ):
    sz = len(img)
    lz = len(bowVec)
    imgVec = [ 0.0 for i in range(0,lz) ]
    for p in range(0,sz):
        x = img[p]
        dA = np.array( [ distance(x[0],bowVec[j,:]) for j in range(0,lz) ] )
        k = dA.argmin()
        imgVec[k] = imgVec[k] + 1.0
    imgVec = np.array(imgVec)
    return  imgVec

def processImgPostSift( bow, dataInput, dataOutput ):
    print 'Converting Raw Images to Bag of Words Representation ...'
    f = open(dataInput,'r')
    w = pickle.load(f)
    xX, yY = w
    print 'Data from '+str(dataInput)+' fetched'
    nEx = (len(xX))
    print 'Number of data points is '+str(nEx)
    mem = []
    for i in range(0,nEx):
        iv = getIMGvec( bow, xX[i] )
        mem.append( [iv,yY[i]] )
    random.shuffle(mem)
    memx = []
    memy = []
    for e in mem:
        memx.append( e[0] )
        memy.append( e[1] )
    print 'Saving the data in '+str(dataOutput)+' file ...'
    f1 = open(dataOutput,'w')
    pickle.dump([memx,memy],f1)
    return "...Done!"

#--------------------------------------------------------------------

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

def process_images_gabor( stddev ):
	classNames = open('classNames.txt','r')
	list_classes = classNames
	#print 'Beginning Gabor processing of images across', num_classes, 'classes...'
	i = 0
	image_data = []
	for class_name in list_classes:
		folderName = (class_name.split('\n'))[0]
		class_path = folderName
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

#--------------------------------------------------------


def randomForest(data, training_percent, numOfTrees ):
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
	print 'Training', len(x_train), 'data samples on RandomForest...'
	clf = RandomForestClassifier(n_estimators=numOfTrees)
	clff = clf.fit(x_train, y_train)
#	print 'Testing', len(x_test), 'data samples on SVM...'
#	y_predicted = classifier.predict(x_test)
#	plt.plot(y_test, y_predicted, 'r*')
#	plt.show()
	print 'Calculating validation accuracy...'
	score = clff.score(x_test, y_test)
	print 'Accuracy on validation data is', score
	print 'RandomForest Analysis Done'
	return clff


def svm(data, training_percent, C_value, gamma_value):
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
	classifier = svm.SVC(C=C_value, cache_size=1000, gamma=gamma_value, kernel='rbf', probability=True)
	classifier.fit(x_train, y_train)
#	print 'Testing', len(x_test), 'data samples on SVM...'
#	y_predicted = classifier.predict(x_test)
#	plt.plot(y_test, y_predicted, 'r*')
#	plt.show()
	print 'Calculating test score...'
	score = classifier.score(x_test, y_test)
	print 'Score on test data is', score
	print 'SVM Analysis Done'
	return classifier
	
def logReg(data, training_percent, C_value):
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
	return classifier

def num_images(path):
	list_classes = os.listdir(path)
	image_data = []
	for class_name in list_classes:
		class_path = path+'/'+class_name
		image_data += [len(os.listdir(class_path))]
	return image_data

