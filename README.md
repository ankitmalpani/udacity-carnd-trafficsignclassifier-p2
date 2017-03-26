# udacity-carnd-trafficsignclassifier-p2
Submission of Udacity CarND's 2nd project: Traffic Sign Classifier
This repository includes code to classify German Traffic signs using Convolutional networks.

[//]: # (Image References)

[image1]: ./sample_images/speed_limit_50.jpeg "speed_limit_50"
[image2]: ./sample_images/stop_sign.jpg "stop_sign"
[image3]: ./sample_images/keep_right.jpg "keep_right"
[image4]: ./sample_images/turn_right.jpg "turn_right"
[image5]: ./sample_images/yield_sign.png "yield_sign"
[image6]: ./processed_images/70.png "pro_70"
[image7]: ./processed_images/caution.png "caution"
[image8]: ./original_images/speed_30.png "speed_limit_30"
[image9]: ./original_images/no_entry.png "no_entry"
[image10]: ./original_images/keep_left.png "keep_left"

### Code structure
I have basically structured the code a little different than the original template notebook. I did this so that it is really easy to following what a given cell does (and it is not polluted with a lot of helper code & comments around it)
This will serve as a good index for the reader.
1) All necessary imports
2) Data Load
3) All helper functions/methods
4) Data exploration and Summary
5) Pre-processing data (multiple cells)
6) Model architecture and definition
7) Evaluation
8) Load, evaluation with new data

### Data Load, Summary & Exploration
Data load was easy once I downloaded the zip file from the given link in the instructions.
It was also good to have the data divided into training, validation and test sets.
* I used basic python to open files and then load data as numpy arrays. This is all specified in the second cell.
* To explore the data I printed the stats using basic python built-in tools such as `len` and `set` and this code is in thed `display_summary` method in helper functions.
* To plot/display the images in the data, I used matplotlib's `mpimg` package. The core code can be seen in `show_image` method in helper functions. I also added some helper methods around it to help me display random images. See: `show_random_image`
* Another helper I wrote was to load the signnames.csv into a dictionary so that I could use it to lookup sign names through out the notebook: `load_sign_names`
* With all these methods in hand, Some notes from when I explored the data, plotted images and printed some stats:
1. The one thing that struck me immediately was the size of the validation set. it was much smaller than the usual recommended 20%. At this point I had two options, either to augment test + validation sets and do my own splits or continue with what I had been given, and I decided to start with the latter. I wanted to see if I could come up with a model that'll prove to be accurate no matter how the validation set is. I decided to iteratively change this approach if needed. ( I didn't! )
2. Second noticeable was the quality of images - especially how dark some images were. They were dark enough to not be recognizable by a naked eye
3. The third thing that struck me was from pre-mentioned notes in the assignment instructions: Number of examples per label (some have more than others)


### General approach & intuition
This is a good place to talk about the general approach I took to solve this problem (given what I knew about the data in hand and a pre-constructed LeNet architecture that works to about 88% accuracy). My general intuition was to think of "what would I (or a human) need to classify these signs more accurately" and If I could get that input to LeNet, how much would it improve on accuracy? That thought is what helped me decide how I wanted to pre-process the data. (This is all before I had gone through any blogs, papers on how to increase accuracy and tune parameters)

### Pre-process
Keeping the general approach and intuition in mind

##### Pre-process step 1: Light'em up
1. I decided to take the first dig at fixing the darkness issue. To do this, I decided to calculate the mean pixel value of a given image using `np.mean()` which I though would be a good average representation of an image's general brightness. I plotted a few random images with their mean pixel value and found that everything that had a pixel value below `30` was really unrecognizable and I decided to set this as my threshold. Every image that had a value lower than 30 would need to be brightened.
2. To brighten the image I wrote a core helper method that would take the image and a factor by which the image should be brightened. To clarify: The factor is randomly chosen from a specified limit instead instead of hard-coding facto values. This can be seen in: `brighten_image(image, low_limit=1.5, high_limit=2.0)` method in helper functions.
3. After I did this, I noticed that the images with mean pixel value lower than 20 would hardly be affected by a factor of 1.5 and were still pretty dark (with values close to 30 after brightening), and hence I decided to brighten them with a higher factor as seen in helper method: `brighten_dark_images()`
4. At this point I was quite happy with the resulting images. They were far more recognizable and with just this change I ran the LeNet model unchanged and that did give me some improvement but it was still just minimally above `90%`
5. This made me realize that - even after applying the brightness, some images remained dark and still had a mean pixel value lower than `30` - most possibly due to the random factor chosen. Hence I decided to iterate and re-brighten till negligible number (`10`) of images had a mean pixel value below the threshold of `30`. This can be seen in helper method: `brighten_inputs()`. This last step hardly made a difference to my accuracy.
*Note*: Do read the note in Pre-process step 3 related to brightness as well.

###### Sample of images after brightening
![alt text][image8]
![alt text][image9]

##### Pre-process step 2: No discrimination
With limited improvements in accuracy, my next thought was to increase the data I am using in training. There were multiple ways of doing it.
* Actually download more data and process it to be similar to the current set (I pretty much eliminated this thought as the last resort)
* Generating random fake data by somehow changing the images a little bit (darkness, contrast etc.). I really didn't how to do this other than go back and overuse my brighten_image methods. This was also when I realized and asked if it could help me resolve the problem where some classes have very little data in the training set. I pretty much thought 'yes', but there could be a better approach to solving this under-representation issue. and hence the next few steps:
1. I decided to then only focus and fix the problem of under-representation. I needed to find out which output classes had the least number of entries in the training data. I did this by writing a helper method: `find_under_represented()`. I must note that this was after a couple of iterations and trials that I landed at the threshold of `400`. I printed out the frequency counts using `np.bincount` and then decided to start with `300` first. Only after a few runs I made that number `400`.
2. Once I had the list of under-represented classes, it meant generating more data for these classes - and this point I had no idea how to do it (and I didn't want to just change how bright images were), and hence I decided to turn to forums for some good suggestions. I couldn't find much but, saw that some of my fellow learners were trying to rotate images to get more data and this was the que I needed.
3. I decided to google up and read about affine transforms and finally write a core helper method that did exactly this: `apply_affine_transform`. What I should note here is that after reading more I found a much better version of the similar method in Vivek Yadav's (Udacity student/mentor) medium post: https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc#.utpqjpn6l which I decided to take inspiration from and change my method to something quite similar: `transform_image(img,angle=None,shear=None,trans=None)`
4. With this method, I decided to write a helper/wrapper that would help me generate more data for under-represented classes by calling af full transform once and a rotation once (so 2 additional entries for each under-represented class). This would mean that they would've 3-folded in the amount of training data they have. This can be seen in: `generate_images_for_under_reps`
5. I also decided to write a quick helper to generate fake data `generate_images_random` for about 3000 random inputs (where this thought had started) - but only to find out that this hardly made a difference to my accuracy. so I have skipped this step in pre-processing. You'll see this in a cell as a step - but I have skipped this in my final run

##### Pre-process step 3: Grays, Zero means & Adaptive Histograms
I was earlier of the intuition that colors are important in this task - even after reading 'gray scale conversion' in instructions - as human's tend to rely plenty on colors when identifying signs. However, when I was reading the forums/slack - a good point was made that, for traffic signs you have to rarely rely on colors and the general shape gives plenty input into what the sign does. The next few steps will describe these new pre-process methods that I applied.
1. I added a simple `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)` conversion
2. I also added a simple zero-mean normalization on pixel values as mentioned in one of the chapters in helper method `normalize`
3. This helped my model improve the accuracy but still couldn't hit `.93` or higher. This is when I started reading about the `histogram equalization` on some of the forum posts. I read up on this and decided to use `cv2.createCLAHE()` to do this.
4. All these steps can be seen in these methods: `normalize()`, `gray_and_hist_equalize()`, `pre_process()`

*Note:* What you'll also see here is a step that was an after thought after reading some blogs/papers. Especially Vivek Yadav who posted `98%` accuracy mentioned how he actually changed brightness randomly on images so that his network doesn't rely on brightness. It was quite interesting to read this and this was quite different to how I was approaching my problem (how humans would do it). I really thought this could be worth a try and hence added the step of introducing random brightness into about 10000 inputs where mean pixel value was above `100` (seen in method: `apply_random_brightness()`). This made a slight improvement to accuracy but not all that much - hence decided to leave it as is.

##### Sample of images after pre-processing (random brightness, gray scale and adaptive histograms)

![alt text][image6]
![alt text][image7]

##### Pre-process step 4: Shuffle
This is a regular shuffle as done in previous labs.


### Convolutional network Architecture

##### Hyperparams
Changing hyper params was an iterative process through out this exercise. I pretty much was running my model after every pre-process step I was making and at several points decided to change things like Epochs (from `10` to `30` to `50`) and the keep probability in the `dropout variable` from (`.8` to `.5` to `.65` to `.7`)

##### Convnet definition
With all the pre-processing I had done I was just about touching accuracy of `.93x` or `.92x`. It was enough to warrant a try on test evaluation. This is when I decided to change the the convnet definition and introduce more filters to see if that made a difference ( and it did!). Here are the steps I took: as seen in cell `In[15]` and I should also note that comments have a few typos and might not be updated with the correct numbers.
1. I first introduced a dropout to the LeNet architecture. I started with a low keep probability of `.5` to really see what difference can it make (is it under-fitting etc.). After a couple of iterations, I saw that `.65` or `.7` have similar affect on accuracy and are reasonably giving me good results (My validations had a decent accuracy)
2. I did think of changing the learning rate (or looking at decaying rate), but after reading up about `AdamOptimizer` realized that it handles the rate changes by itself.
3. After reading up a bit from the mentioned additional resources and skimming through forum posts and majorly going through `Andrej Karpathy's lectures` - I thought of adding more filters. I changed this value (`6` -> `16`) in the first layer and (`16` -> `32`) in the second. This made all the difference in how the network learned. This pushed me to a validation accuracy of `.96x` and `.97` on some epochs. I played a little ( not to my heart's content) with the number of filters but couldn't really get consistent results and I wanted some method to this process and hence stopped till I had a better grasp on the right numbers and decided to stick with this new architecture. It is shown in the template table below.
4. I should mention that I kept changing Epochs if I hadn't seen a dip in my accuracy for 2 epochs. There were iterations where 50 felt un-needed but my current architecture maxed around 48th and 50th epochs in most of the runs.
5. Once I was satisfied with the accuracy - I ran the model on test set to get a `94%` result!

| Layer         		|     Description	        					          |
|:---------------------:|:-------------------------------------------------------:|
| Input         		| 32x32x3 RGB image   							          |
| Convolution 5x5     	| 1x1 stride, valid padding, 16 filters,outputs 28x28x16  |
| RELU					| Acitvation											  |       
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				          |
| Convolution 5x5	    | 1x1 stride, valid padding, 32 filters,outputs 10x10x32  |
| RELU					| Acitvation											  |
| Max pooling	      	| 2x2 stride,  outputs 5x5x32				              |
| Flatten				| outputs 800											  |
| Fully connected		| Input 800, Output 240       						      |
| RELU					| Acitvation											  |
| Dropout				| Keep Prob: .7											  |
| Fully connected		| Input 240, Output 84       						      |
| RELU					| Acitvation											  |
| Dropout				| Keep Prob: .7											  |
| Fully connected		| Output layer, Input 84, Output 43      				  |
| Softmax				| final output        									          |

*Note*: The comments in the python notebook might not be updated on the layers.

### Testing a Model on New Images
I downloaded some random images from the internet. Surprisingly not the easiest to find. :)
They are shown here in their original sizes

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

Here is the process I followed to test these:

1. Loaded images using `matplotlib's imread`
2. Plotted these images using `matplotlib's imshow`
3. Cropped these images to get only the traffic signs using basic slicing
4. Converted these images to grayscale
5. Applied some gaussian smoothing with a kernel `5` (Remembered this from our first project)
6. Resized them to be similar to our input size of 32x32x3 using `scipy's imresize`
7. At this point the images were quite blurry (Removing Gaussian smoothing didn't help much), but I decided to see if the network classifies this anyway.
8. I created a basic numpy arrays with new input images and new outputs from signnames.csv
9. Ran the network to get predicted logits and labels
10. Ran the accuracy test with the new test data (similar to how I ran the earlier given test data)
11. The results were pretty good. I ran about 10 iterations on this data, adding/removing images and changing their order. However, I was seeing consistent accuracies of `.8` adn `1`

### Softmax probabilities
Here I basically used `tf.nn.top_k` function specified in the example to print the top 3 probabilities for my logits as seen in cells `In [91]` and `In [94]`.

*Important Note:* I have submitted top 3 here - I hope thats OK. I later realized that the question asked for top 5 - but I had already terminated my AWS GPU. Hence decided to keep it at 3, which gives the same idea. If top 5 is needed, let me know and I'll re-create an instance and run this exercise.

### Visualize
I have intentionally left this since this wasn't required right now and in the favor of submitting the assignment in time. I intend to play with this after the submission. along with various other options and improvements I need to try.

### Further improvements and trials
I would like to try out a few new things in the future from what I've read and researched and see if I could hit a `98%` accuracy

1. Optional Visualization step in the assignment
2. Apply L2 norms to my data - if that would make a difference
3. Adding an extra conv layer to see how it changes accuracy. Would it make a difference if it was a 1x1 filter
4. Trying out different acitivations like Leaky RELU
5. Also, taking inspiration from some papers, introducing randomness instead of cleaning up(brightening) the data.
