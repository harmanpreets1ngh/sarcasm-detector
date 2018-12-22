# Sarcasm Detector

This was a final group project for my course: Natual Language Processing. 3 of us worked on it during a period of 3-4 months. We were able to achieve good results over the time we worked on it, but there is a scope of improvement to this project using different algorithms. Feel free to add to it. 


## Members
- Alice Nin
- Harman Preet Singh
- Ryan Kalla


## Setup

1. Make sure you are using Python 3 and have the following installed:
  * numpy
  * nltk
  * scipy
  * sklearn


2. Run `naive_bayes/NaiveBayes.py` to run the Naive Bayes classifier

3. Run `svm/utils.py` to run the SVM and the MaxEnt classifiers


## Why Sarcasm Detection?

Merriam-Webster defines sarcasm as the use of words that mean the opposite of what you really want to say especially in order to insult someone, to show irritation, or to be funny. Detecting sarcasm is not a trivial task and can't even be done by Humans accurately. It is particularly a difficult task for machines as they lack all the context. Humans, while making sarcastic remarks also use audio and visual cues which can't be provided to the machine, specially when using text based processing methods. 

A few years ago, even the US Secret Service was looking for a working sarcasm detector to improve their intelligence coming from Twitter, which shows the importance of such a tool.


## Dataset

The dataset we are using was collected by Mathieu Cliche. The data in the set was preprocessed by Cliche in the following way. All tweets that have http links in them, and all tweets that start with ‘@’ were removed (this is so that the entire context of the sarcasm is contained within the tweet). Only tweets made from New York or San Francisco were kept (to maximize the probability that the tweets are in english). Tweets with non ASCII characters were removed. Hashtags, friend tags, and mentions were removed from the tweet. If, after this pruning stage, the tweet was at least 3 words long, it was added to the set.

The dataset is split into two collections: tweets that had #sarcasm prior to preprocessing (which count as positive data points), and tweets without #sarcasm (which are negative data points). Cliche saved each collection in a .npy file which we were able to read into arrays of tweets with the numpy module. From there we separated the first 20,000 sarcastic tweets and the first 100,000 non sarcastic tweets to use for training (leaving 5,273 sarcastic and 17,825 non sarcastic tweets for testing, which is about 16% of the dataset).


## Methodologies Used

Sarcasm detection is a supervised learning problem, and the models being used are binary classification models. We planned to build off of the work done by Mathieu Cliche, tested a Naive Bayes algorithm, moved to SVM and MaxEnt from there. Cliche focused on n-grams, sentiments, and topics. We thought an area of improvement could be refining those features. Sentimental analysis of figurative data and sentiment polarity classification of the text can also be done by dividing into subcategories like subjectivity, polarity and sarcasm like content which is possible by Supervised Machine learning.

Features we used were:
Unigrams, Bigrams, Repeated Characters, Capitalization, Sentiments

Unigrams and Bigrams acted as a foundation for the scores we achieved. We tried SVM and MaxEnt models without unigrams and bigrams and we got scores around ~0.30, but with their addition, we achieved results up to ~0.70. Every other feature just improved the model on top of each other.

	
## Evaluation Method

In order to determine the accuracy of our algorithms, we would calculate the F-score. The F-score is the harmonic mean of precision and recall. It is a standard that has been used in other research on sarcasm detection.

## Results

We used our Naive Bayes as a baseline for our sarcasm detection. Our Naive Bayes achieved an F-Score of 0.5525 with all our features.

When we ran the SVM model on a balanced dataset and achieved improved results. Our SVM achieved an F-Score of 0.7025 with all our features.

For MaxEnt model we used a similar dataset as SVM, and achieved slightly better results. Our MaxEnt achieved an F-Score of 0.7038 with all our features.

