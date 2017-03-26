# udacity-carnd-trafficsignclassifier-p2
Submission of Udacity CarND's 2nd project: Traffic Sign Classifier
This repository includes code to classify German Traffic signs using Convolutional networks.

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


##### Pre-process step 2: No discrimination
With limited improvements in accuracy, my next thought was to increase the data I am using in training. There were multiple ways of doing it.
* Actually download more data and process it to be similar to the current set (I pretty much eliminated this thought as the last resort)
* Generating random fake data by somehow changing the images a little bit (darkness, contrast etc.). I really didn't how to do this other than go back and overuse my brighten_image methods. This was also when I realized and asked if it could help me resolve the problem where some classes have very little data in the training set. I pretty much thought 'yes', but there could be a better approach. and hence the next few steps:
1. I decided to then only focus and fix the problem of under-representation. I needed to find out which output classes had the least number of entries in the training data. I did this but writing a helper method: `find_under_represented()`. I must note that this was after a couple of iterations and trials that I landed at the threshold of `400`. I printed out the frequency counts using `np.bincount` and then decided to start with `300` first. Only after a few runs I made that number `400`.
2. Once I had the list of under-represented classes, it meant generating more data for these classes - and this point I had no idea how to do it (and I didn't want to just change how bright images were), and hence I decided to turn to forums for some good suggestions. I couldn't find much but, saw that some of my fellow learners were trying to rotate images to get more data and this was the que I needed.
3. I decided to google up and read about affine transforms and finally write a core helper method that did exactly this: `apply_affine_transform`. What I should note here is that after reading more I found a much better version of the similar method in Vivek Yadav's (Udacity student/mentor) medium post: https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc#.utpqjpn6l which I decided to take inspiration from and change my method to something quite similar: `transform_image(img,angle=None,shear=None,trans=None)`
4. With this method, I decided to write a helper/wrapper that would help me generate more data for under-represented classes by calling af full transform once and a rotation once (so 2 additional entries for each under-represented class). This would mean that they would've 3-folded in the amount of training data they have. This can be seen in: `generate_images_for_under_reps`
5. I also decided to write a quick helper to generate fake data `generate_images_random` for about 3000 random inputs (where this thought had started) - but only to find out that this hardly made a difference to my accuracy. so I have skipped this step in pre-processing. You'll see this in a cell as a step - but I have skipped this in my final run

##### Pre-process step 3: Grays, Zero means & Adaptive Histograms
I was earlier of the intuition that colors are important in this task - even after reading 'gray scale conversion' in instructions - as human's tend to rely plenty on colors when identifying signs. However, when I was reading the forums/slack - a good point was made that, for traffic signs you have to rarely rely on colors and the general shape gives plenty input into what the sign does. The next few steps will describe these new pre-process methods that I applied.
