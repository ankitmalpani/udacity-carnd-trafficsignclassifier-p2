# udacity-carnd-trafficsignclassifier-p2
Submission of Udacity CarND's 2nd project: Traffic Sign Classifier
This repository includes code to classify German Traffic signs using Convolutional networks.

###Code structure
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

###Data Load, Summary & Exploration
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

###General approach & intuition
This is a good place to talk about the general approach I took to solve this problem (given what I knew about the data in hand and a pre-constructed LeNet architecture that works to about 88% accuracy). My general intuition was to think of "what would I (or a human) need to classify these signs more accurately" and If I could get that input to LeNet, how much would it improve on accuracy? That thought is what helped me decide how I wanted to pre-process the data. (This is all before I had gone through any blogs, papers on how to increase accuracy and tune parameters)

###Pre-process
Keeping the general approach and intuition in mind
1. I decided to take the first dig at fixing the darkness issue. To do this, I decided to calculate the mean pixel value of a given image using `np.mean()` which I though would be a good average representation of an image's general brightness. I plotted a few random images with their mean pixel value and found that everything that had a pixel value below `30` was really unrecognizable and I decided to set this as my threshold.
