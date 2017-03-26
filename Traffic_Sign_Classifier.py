
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[1]:

import csv
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import cv2
from sklearn.utils import shuffle
from skimage import exposure
import scipy as sp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file = 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
print('data loaded')


# In[8]:

#all helper functions

#helper for time logs
current_milli_time = lambda: int(round(time.time() * 1000))

#displays data summary
def display_summary():
    n_train = len(X_train)
    n_test = len(X_test)
    n_valid = len(X_valid)
    image_shape = np.shape(X_train[0])
    n_classes = len(set(train['labels']))
    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_valid)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)
    print("Shape of labels =", y_train.shape)
    
#loading signnames into a dictionary for visualization step
def load_sign_names():
    sign_dict = {}
    with open('signnames.csv', mode='r') as infile:
        reader = csv.reader(infile)
        sign_dict = dict((rows[0],rows[1]) for rows in reader)
    print('done loading sign names to a dictionary')
    return sign_dict

#shows random image from input
def show_random_image(X, y, sign_dict, count=10):
    for i in range(count):
        index = random.randint(0, len(X))
        show_image(X[index], y[index], sign_dict)

#displays 1x1 image
def show_image(image, y, sign_dict):
    mean_val = np.mean(image)
    squeezed_img = image.squeeze()
    plt.figure(figsize=(1,1))
    print('{num}:{name}, mean_pixel value: {mv}'.format(num=y, name=sign_dict[str(y)], mv=mean_val))
    plt.imshow(squeezed_img, cmap="gray")   
    
#find under represented signs
def find_under_represented():
    y_train_bins = np.bincount(y_train)
    y_val = np.nonzero(y_train_bins)[0]
    freq_list = list(zip(y_val,y_train_bins[y_val]))
    under_rep_classes = [x[0] for x in freq_list if x[1] < 400]
    return under_rep_classes

#old method (unsued)
def apply_affine_transform(img,rotation=None,translation=None):
    if rotation is not None:
        angle = np.random.uniform(rotation)
        rows,cols,ch = img.shape    
        rotated = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),angle,1)
        img = cv2.warpAffine(img,rotated,(cols,rows))

    if translation is not None:
        x_trans = translation*np.random.uniform()
        y_trans = translation*np.random.uniform()
        translated = np.float32([[1,0,x_trans],[0,1,y_trans]])
        img = cv2.warpAffine(img,translated,(cols,rows))
    
    return img

#new method 
#inspired by blog:https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc#.utpqjpn6l
#similar to what I had but works far better than mine with the minor adjustments added
def transform_image(img,angle=None,shear=None,trans=None):
    
    # Rotation
    if angle is not None:
        ang_rot = np.random.uniform(angle)-angle/2
        rows,cols,ch = img.shape    
        Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
        img = cv2.warpAffine(img,Rot_M,(cols,rows))
        
    # Translation
    if trans is not None:
        tr_x = trans*np.random.uniform()-trans/2
        tr_y = trans*np.random.uniform()-trans/2
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
        img = cv2.warpAffine(img,Trans_M,(cols,rows))

    # Shear
    if shear is not None:
        pts1 = np.float32([[5,5],[20,5],[5,20]])
        pt1 = 5+shear*np.random.uniform()-shear/2
        pt2 = 20+shear*np.random.uniform()-shear/2
        pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
        shear_M = cv2.getAffineTransform(pts1,pts2)
        img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    return img

#brigthens a given image
def brighten_image(image, low_limit=1.5, high_limit=2.0):
    image = np.array(image, dtype = np.uint8)
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = np.random.uniform(low=low_limit, high=high_limit)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

#create more images using transformations for under-represented sets
def generate_images_for_under_reps(X, y):
    for i in range(len(y)):
        if y[i] in under_represented:
            transformed_img = transform_image(X[i],20,10,5) #full transforms
            X = np.append(X, [transformed_img], axis=0)
            y = np.append(y, y[i])
            rotated_img = transform_image(X[i],20,None,None) #rotations only
            X = np.append(X, [rotated_img], axis=0)
            y = np.append(y, y[i])
    return X, y

def generate_images_random(X, y):
    for i in range(3000):
        transformed_img = transform_image(X[i],30,10,5)
        X = np.append(X, [transformed_img], axis=0)
        y = np.append(y, y[i])
    return X, y

def brighten_dark_images(X):
    for i in range(len(X)):
        mean_pixel_value = np.mean(X[i])
        if mean_pixel_value <= 20.:
            X[i] = brighten_image(X[i], low_limit=2.2, high_limit=2.5)
        elif mean_pixel_value > 20. and mean_pixel_value < 35.:
            X[i] = brighten_image(X[i])
        else:
            continue
            
def check_for_darks(X):
    count = 0
    for image in X:
        if np.mean(image) < 30:
            count += 1
    return count

def brighten_inputs(X):
    d = check_for_darks(X)
    count = 0
    while( d > 10):
        print("dark images Before iteration: {s}".format(s=d))
        brighten_dark_images(X)
        count += 1
        print('done brightening images for iteration: {i}'.format(i=count))
        d = check_for_darks(X)
    d_after = check_for_darks(X)
    print("dark images: {s}".format(s=d_after))
    
def apply_random_brightness(X):
    for j in range(10000):
        index = random.randint(0, len(X_train)-1)
        if np.mean(X[index]) > 100.:
            X[index] = brighten_image(X[index], low_limit=.5, high_limit=1)

def pre_process(X, is_train=False):
    if is_train:
        apply_random_brightness(X)
        print('done apply random brightness')
    gray_scale_X = gray_and_hist_equalize(X)
    print("gray scale done")
    gray_scale_normalized_X = normalize(gray_scale_X)
    print("normalized gray scale image")
    gray_scale_normalized_X = gray_scale_normalized_X[...,np.newaxis]
    return gray_scale_normalized_X

# Normalize the image data
def normalize(data):
    normalized_images = []
    for image in data:
        norm_img = (image - 128.) / 128
        normalized_images.append(norm_img)
    return np.array(normalized_images)

#convert images to grayscale
def gray_and_hist_equalize(data):
    gray_images = []
    clahe = cv2.createCLAHE() #adaptive histogram equalization
    for image in data:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_ah = clahe.apply(gray)
        gray_images.append(gray_ah)        
    return np.array(gray_images)

def process_cropped_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian_gray = cv2.GaussianBlur(gray_img, (5, 5), 0)
    resized_img = sp.misc.imresize(img, (32,32,3), interp='cubic')
    return resized_img

np.seterr(divide='ignore', invalid='ignore')


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.

# In[4]:

#summary and exploration
display_summary()
sign_dict = load_sign_names()
show_random_image(X_train, y_train, sign_dict, 4)


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[5]:

#pre-process step 1 - brighten inputs (for really dark images)
brighten_inputs(X_train)
brighten_inputs(X_valid)
brighten_inputs(X_test)


# In[6]:

#pre-process step 2 - find under represented classes and add additional images using rotations and transforms
under_represented = find_under_represented()
print("Before adding X: {s}".format(s=len(X_train)))
X_train, y_train = generate_images_for_under_reps(X_train, y_train)
print("After adding X: {s}".format(s=len(X_train)))
print("After generating data, number of under-represented classes: {ur}".format(ur=len(find_under_represented())))


# In[ ]:

#pre-process step 3 - generate more data for random input images using rotations and transforms - skipping
X_train, y_train = generate_images_random(X_train, y_train)
print("random generation done: {s}".format(s=len(X_train)))


# In[9]:

#pre-process step 4 - training: apply random brightness, convert to grayscale, normalize pixel values between -1 & 1
X_train = pre_process(X_train, is_train=True)
print('done pre processing training set')


# In[10]:

#pre-process step 5 - validation data: convert to grayscale, normalize pixel values between -1 & 1
X_valid = pre_process(X_valid)
print('done pre processing validation set')


# In[11]:

#pre-process step 6 - test data: convert to grayscale, normalize pixel values between -1 & 1
X_test = pre_process(X_test)
print('done pre processing test set')


# In[12]:

#pre-process step 7 - shuffle data
X_train, y_train = shuffle(X_train, y_train)
print("shuffle done")


# In[13]:

#sanity check: display random images from training data after pre-processing
show_random_image(X_train, y_train, sign_dict, 5)


# In[14]:

#hyper parameters
EPOCHS = 50
BATCH_SIZE = 128
mu = 0 # mean for tf.truncated_normal
sigma = 0.1 #SD for tf.truncated_normal
dropout = 0.7 #dropout probability
rate = 0.001 #learning rate


# ### Model Architecture

# In[15]:

#Default LeNet architecture with some changes
def LeNet(x):    

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 16), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(16))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x16. Output = 14x14x16.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x32.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 32), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(32))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x32.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x32. Output = 800.
    fc0   = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 800. Output = 240.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(800, 240), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(240))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation and Dropout
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Layer 4: Fully Connected. Input = 240. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(240, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation and Dropout
    fc2    = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


# In[16]:

#Tensorflow placeholders for input and ouput
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)


# In[17]:

### Train your model here.
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer() #learning_rate = rate
training_operation = optimizer.minimize(loss_operation)


# In[18]:

### Calculate and report the accuracy on the training and validation set.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[19]:

### Once a final model architecture is selected
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.


# In[20]:

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[52]:

#load new images
image1 = mpimg.imread('speed_limit_50.jpeg')
image2 = mpimg.imread('keep_right.jpg')
image3 = mpimg.imread('stop_sign.jpg')
image4 = mpimg.imread('turn_right.jpg')
image5 = mpimg.imread('yield_sign.png')


# In[53]:

#crop images to get only the traffic sign
crop_img1 = image1[30:115, 35:115]
crop_img2 = image2[20:120, 40:140]
crop_img3 = image3[5:480, 100:575]
crop_img4 = image4[20:110, 45:135]
crop_img5 = image5 #not needed for image 5


# In[54]:

#process images
test_image1 = process_cropped_image(crop_img1)
test_image2 = process_cropped_image(crop_img2)
test_image3 = process_cropped_image(crop_img3)
test_image4 = process_cropped_image(crop_img4)
test_image5 = process_cropped_image(crop_img5)


# In[75]:

#create test input
test_arr = [test_image1, test_image2, test_image3, test_image4, test_image5]
preprocessed_test_set = pre_process(test_arr)


# In[76]:

#create test expected output
new_y_set = np.array([2, 38, 14, 33, 13]) #from signnames.csv


# ### Predict the Sign Type for Each Image

# In[83]:

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    predicted_logits = sess.run(logits, feed_dict={x:preprocessed_test_set.astype(np.float32)})
    predicted_labels = np.argmax(predicted_logits, axis=1)


# In[86]:

#pinting predictions
print(predicted_labels)


# ### Analyze Performance

# In[88]:

### Calculate the accuracy for these 5 new images. 
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(preprocessed_test_set, new_y_set)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tk.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[91]:

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    top_3 = sess.run(tf.nn.top_k(tf.constant(predicted_logits), k=3))


# In[94]:

#printing top 3 from the predtictions for new test images
print(top_3)


# ---
# 
# ## Step 4: Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[ ]:

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")


# ### Question 9
# 
# Discuss how you used the visual output of your trained network's feature maps to show that it had learned to look for interesting characteristics in traffic sign images
# 

# **Answer:**

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 
