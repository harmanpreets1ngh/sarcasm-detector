import sys
import re
import nltk
import math
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet
from nltk.corpus.reader.wordnet import WordNetError
import svm
import max_ent
from sklearn.preprocessing import StandardScaler
import os.path
import pickle

from datetime import datetime

STOPWORDS = set(stopwords.words('english'))
PUNCTUATION = { ',', '.', '?', '!', ';', ':' }

# Synsets processing
MIN_SENSES = 3
MAX_SENSES = 12
REMOVE_COMMON_NGRAMS = True
REMOVE_STOPWORDS = False
REMOVE_PUNCTUATION = False
SS_PUNCTUATION = PUNCTUATION - { '?', '!' }

################################## Load Data ###################################

COUNT_SARCASTIC_TRAINING_TWEETS = 20000
COUNT_NON_SARCASTIC_TRAINING_TWEETS = 100000

SARCASTIC = 1
NON_SARCASTIC = 0

NUM_MOST_COMMON_NGRAMS = 10000

# removes blank lines, replaces \n with space, removes duplicate spaces
def process_whitespace(token_str):
    no_new_line = re.sub(r'\n', " ", token_str)
    no_dup_spaces = re.sub(r'  +', " ", no_new_line)
    return no_dup_spaces

def load_data():
    posproc = np.load('../data/posproc.npy')
    negproc = np.load('../data/negproc.npy')

    sarcastic_tweets = []
    non_sarcastic_tweets = []

    for tweet in posproc:
        sarcastic_tweets.append((tweet.decode('utf-8'), SARCASTIC))

    for tweet in negproc:
        non_sarcastic_tweets.append((tweet.decode('utf-8'), NON_SARCASTIC))

    print('num sarcastic tweets: ' + str(len(sarcastic_tweets)))
    print('num non sarcastic tweets: ' + str(len(non_sarcastic_tweets)))

    return sarcastic_tweets, non_sarcastic_tweets

# Separate training and testing data
def get_data(sarcastic_tweets, non_sarcastic_tweets):
    labeled_training_sarcastic_tweets = sarcastic_tweets[0:COUNT_SARCASTIC_TRAINING_TWEETS]
    labeled_testing_sarcastic_tweets = sarcastic_tweets[COUNT_SARCASTIC_TRAINING_TWEETS:]

    labeled_training_non_sarcastic_tweets = non_sarcastic_tweets[0:COUNT_NON_SARCASTIC_TRAINING_TWEETS]
    labeled_testing_non_sarcastic_tweets = non_sarcastic_tweets[COUNT_NON_SARCASTIC_TRAINING_TWEETS:]

    training_sarcastic_tweets, training_sarcastic_tweets_labels = zip(*labeled_training_sarcastic_tweets)
    training_non_sarcastic_tweets, training_non_sarcastic_tweets_labels = zip(*labeled_training_non_sarcastic_tweets)
    testing_sarcastic_tweets, testing_sarcastic_tweets_labels = zip(*labeled_testing_sarcastic_tweets)
    testing_non_sarcastic_tweets, testing_non_sarcastic_tweets_labels = zip(*labeled_testing_non_sarcastic_tweets)

    train_tweets = training_sarcastic_tweets + training_non_sarcastic_tweets
    train_labels = training_sarcastic_tweets_labels + training_non_sarcastic_tweets_labels
    test_tweets = testing_sarcastic_tweets + testing_non_sarcastic_tweets
    test_labels = testing_sarcastic_tweets_labels + testing_non_sarcastic_tweets_labels

    sarc_freq_set, non_sarc_freq_set = get_sets(training_sarcastic_tweets, training_non_sarcastic_tweets)

    return train_tweets, train_labels, test_tweets, test_labels, sarc_freq_set, non_sarc_freq_set

def get_sets(training_sarcastic_tweets, training_non_sarcastic_tweets):
    print('creating', NUM_MOST_COMMON_NGRAMS, 'most common sarcastic and non-sarcastic ngram sets...')
    sarc_unigrams, sarc_bigrams = \
        get_unigrams_and_bigrams(training_sarcastic_tweets)
    non_sarc_unigrams, non_sarc_bigrams = \
        get_unigrams_and_bigrams(training_non_sarcastic_tweets)
    sarc_freq_unigrams, non_sarc_freq_unigrams = \
        get_freq_ngrams_sets(sarc_unigrams, non_sarc_unigrams)
    sarc_freq_bigrams, non_sarc_freq_bigrams = \
        get_freq_ngrams_sets(sarc_bigrams, non_sarc_bigrams)
    sarc_freq_set = sarc_freq_unigrams.union(sarc_freq_bigrams)
    non_sarc_freq_set = non_sarc_freq_unigrams.union(non_sarc_freq_bigrams)
    return sarc_freq_set, non_sarc_freq_set

def get_unigrams_and_bigrams(tweets):
    words_in_tweets = get_tweet_words(tweets)
    bigrams_in_tweets = find_ngrams_in_tweets(2, words_in_tweets)
    return words_in_tweets, bigrams_in_tweets

def get_freq_ngrams_sets(sarc_ngrams, non_sarc_ngrams):
    fd_only_sarc, fd_only_non_sarc = \
        remove_common_ngrams(sarc_ngrams, non_sarc_ngrams)

    sarc_freq_set = get_freq_ngrams_set(fd_only_sarc)
    non_sarc_freq_set = get_freq_ngrams_set(fd_only_non_sarc)

    return sarc_freq_set, non_sarc_freq_set

def get_freq_ngrams_set(fd_ngrams):
    most_common_ngrams = fd_ngrams.most_common(NUM_MOST_COMMON_NGRAMS)
    freq_ngrams, freq_counts = zip(*most_common_ngrams)
    return set(freq_ngrams)

def remove_common_ngrams(sarc_ngrams, non_sarc_ngrams):
    fd_sarc = get_ngram_freqs(sarc_ngrams)
    fd_non_sarc = get_ngram_freqs(non_sarc_ngrams)
    fd_only_sarc = fd_sarc - fd_non_sarc
    fd_only_non_sarc = fd_non_sarc - fd_sarc
    return fd_only_sarc, fd_only_non_sarc

def get_ngram_freqs(ngrams_in_tweets):
    ngrams = [ ngram for tweet in ngrams_in_tweets for ngram in tweet]
    fd_ngrams = nltk.FreqDist(ngrams)
    return fd_ngrams

################################### N-Grams ####################################

def get_tweet_words(tweets):
    tweet_words = [ nltk.word_tokenize(tweet) for tweet in tweets ]
    return tweet_words

def get_tweet_words_lowercase(tweets):
    tweet_words = [ word.lower()                    for word in
                        nltk.word_tokenize(tweet)   for tweet in tweets ]
    return tweet_words

def get_tweet_words_in_sents(tweets):
    tweet_sentences = [ [nltk.word_tokenize(sentence) for sentence in
                           nltk.sent_tokenize(tweet)] for tweet in tweets ]
    return tweet_sentences

def get_tweet_words_in_sents_lowercase(tweets):
    tweet_sentences = [ [ [word.lower()                    for word in
                             nltk.word_tokenize(sentence)] for sentence in
                             nltk.sent_tokenize(tweet)]    for tweet in tweets ]
    return tweet_sentences

def get_ngrams(n, tokens):
    return [tuple(tokens[i:i+n]) for i in range (len(tokens)-(n-1))]

def find_ngrams_in_tweets(n, tokenized_tweets):
    ngrams = []
    for tokens in tokenized_tweets:
        tweet_ngrams = get_ngrams(n, tokens)
        ngrams.append(tweet_ngrams)
    return ngrams


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tweet Vectors ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# find all words with frequency less than threshold
def find_freq(tokens, THRESHOLD):
    fdist = nltk.FreqDist(tokens)
    freq = [word for word in fdist if fdist[word] > THRESHOLD]
    print("frequent token count:", len(freq))
    return freq

# create a dictionary of all the words in the vocabulary to an index
def create_vocab_dict(freq_tokens):
    tokens = ['UNK'] + freq_tokens
    vocab_dict = {}
    for i, token in enumerate(tokens):
        vocab_dict[token] = i
    return vocab_dict

# creates a list of one hot vectors from tweets
def one_hot_vector_tweets(grams_in_tweets, vocab_dict):
    one_hot_vectors = []
    for tweet in grams_in_tweets:
        one_hot = [0] * len(vocab_dict)
        for token in tweet:
            one_hot[vocab_dict.get(token, 0)] = 1
        one_hot_vectors.append(one_hot)
    return one_hot_vectors

# creates a list of index vectors from tweets
LEN_INDEX_VECTORS = 128
def index_vector_tweets(grams_in_tweets, vocab_dict):
    index_vectors = []
    for tweet in grams_in_tweets:
        vector = [0] * LEN_INDEX_VECTORS
        for i, token in enumerate(tweet):
            vector[i] = vocab_dict.get(token, 0)
        index_vectors.append(vector)
    return index_vectors

######################## Unigram and Bigram Features #########################

# converts tokens into a list of word index vectors using an existing dictionary
def ngrams_to_indices(ngrams_in_tweets, vocab_dict):
    index_vectors = index_vector_tweets(ngrams_in_tweets, vocab_dict)
    return index_vectors

# Compute training vocabulary
def get_training_vocabulary(words_in_tweets, bigrams_in_tweets):
    words = [ word for tweet in words_in_tweets for word in tweet ]
    word_dict = create_vocab_dict(words)
    bigrams = [ bigram for tweet in bigrams_in_tweets for bigram in tweet ]
    bigram_dict = create_vocab_dict(bigrams)
    return word_dict, bigram_dict

def common_ngrams_count(ngrams_in_tweets, freq_set):
    freq_counts = []
    for tokens in ngrams_in_tweets:
        freq_count = 0
        for token in tokens:
            if token in freq_set:
                freq_count += 1
        freq_count *= 1.0
        freq_counts.append(freq_count)
    return freq_counts


def get_freq_ngram_counts(words_in_tweets, bigrams_in_tweets, sarc_freq_set, non_sarc_freq_set):
    print('getting freq sarcastic and non-sarcatistic ngram counts...')
    sarc_unigrams_count = common_ngrams_count(words_in_tweets, sarc_freq_set)
    sarc_bigrams_count = common_ngrams_count(bigrams_in_tweets, sarc_freq_set)
    non_sarc_unigrams_count = common_ngrams_count(words_in_tweets, non_sarc_freq_set)
    non_sarc_bigrams_count = common_ngrams_count(bigrams_in_tweets, non_sarc_freq_set)

    return sarc_unigrams_count, sarc_bigrams_count, non_sarc_unigrams_count, non_sarc_bigrams_count

############# Repeated Characters and Capitalized Words Features ###############

def _get_repeated_character_count_tweet(tweet):
    repeated_character_count = 0
    characters = ['null_1', 'null_2', 'null_3']
    repeated_characters = False
    for character in tweet:
        characters.pop(0)
        characters.append(character)
        if characters[0] == characters[1] and characters[1] == characters[2]:
            repeated_characters = True
            break
    if repeated_characters:
        repeated_character_count += 1
    repeated_character_count *= 1.0
    return repeated_character_count

def get_repeated_character_count_tweets(tweets):
    repeated_character_counts = []
    for tweet in tweets:
        repeated_character_counts.append(_get_repeated_character_count_tweet(tweet))
    return repeated_character_counts

def get_percent_caps(tweet):
    num_caps = 0
    for letter in tweet:
        if letter.isupper():
            num_caps += 1
    percent_caps = num_caps / len(tweet)
    adjusted_percent_caps = math.ceil(percent_caps * 100)
    adjusted_percent_caps *= 1.0
    return adjusted_percent_caps

def get_percent_caps_tweets(tweets):
    caps = {}
    percents = []
    for tweet in tweets:
        percent = get_percent_caps(tweet)
        count = caps.get(percent) or 0
        count += 1
        caps[percent] = count
        percents.append(percent)
    return percents

########################## Sentiment Score Features ###########################

def convert_tag(tag):
    if tag.startswith('NN'):
        return 'n'
    elif tag.startswith('VB'):
        return 'v'
    elif tag.startswith('JJ'):
        return 'a'
    elif tag.startswith('RB'):
        return 'r'
    else:
        return ''

SENTI_TAGS = { 'NN':'n', 'VB':'v', 'JJ':'a', 'RB':'r' }
SENTI_NETS = [ '.%02d' % i for i in range(1, MAX_SENSES+1) ]

def get_word_tag_senti_synset(word_tag_sense, DEBUG=False):
    try:
        synset = sentiwordnet.senti_synset(word_tag_sense)
    except WordNetError:
        synset = None
    except:
        print("Unexpected error getting synset:", sys.exc_info()[0])
    if DEBUG:
        print("--- %s : %s" % (word_tag_sense, synset))
    return synset

def get_word_tag_senti_score(word_tag, senti_scores_dict, VERBOSE=False):
    senti_score = None
    num_exceptions = 0
    if word_tag in senti_scores_dict:
        senti_score = senti_scores_dict[word_tag]
    else:
        synsets = [ get_word_tag_senti_synset(word_tag + n) for n in SENTI_NETS ]
        synsets_found = [ s for s in synsets if s != None ]
        num_exceptions += len(SENTI_NETS) - len(synsets_found)
        if len(synsets_found) >= MIN_SENSES:
            senti_score_pos = np.average([ s.pos_score() for s in synsets_found ])
            senti_score_neg = np.average([ s.neg_score() for s in synsets_found])
            senti_score = (senti_score_pos - senti_score_neg)
            senti_scores_dict[word_tag] = senti_score
            if VERBOSE:
                for i, n in enumerate(SENTI_NETS):
                    print("%s.%s: %s" % (word_tag, n, synsets[i]))
                print("Averages: POS %f NEG %f DIFF %f" % \
                    (senti_score_pos, senti_score_neg, senti_score))
    return senti_score, num_exceptions

def get_senti_score(tweet, senti_scores_dict, DEBUG=False, VERBOSE=False):
    tokens = nltk.word_tokenize(tweet)
    tagged = nltk.pos_tag(tokens)
    senti_tagged = [ w + '.' + SENTI_TAGS[t[:2]] for w, t in tagged if t[:2] in SENTI_TAGS ]

    avg_senti_score = 0
    num_senti_words = 0
    num_exceptions = 0

    for word_tag in senti_tagged:
        senti_score, word_tag_exceptions =  get_word_tag_senti_score(word_tag, senti_scores_dict)
        num_exceptions += word_tag_exceptions
        if senti_score != None:
            avg_senti_score += senti_score
            num_senti_words += 1

    if num_senti_words > 0:
        avg_senti_score /= num_senti_words

    if avg_senti_score >= 0:
        adjusted_score = math.ceil(avg_senti_score * 100)
    else:
        adjusted_score = math.floor(avg_senti_score * 100)

    adjusted_score *= 1.0

    return adjusted_score, num_senti_words, num_exceptions

def get_sentiments_tweets(tweets, senti_scores_dict):
    DEBUG = False
    sentiments = {}
    scores = []
    total_words_scored = 0
    total_exceptions = 0
    tweets_with_scored_words = 0
    tweets_with_exceptions = 0
    print("Scoring sentiment in %d tweets ..." % len(tweets))
    for i, tweet in enumerate(tweets):
        score, num_words_scored, num_exceptions = get_senti_score(tweet, senti_scores_dict, DEBUG)
        count = sentiments.get(score) or 0
        count += 1
        sentiments[score] = count
        scores.append(score)
        # monitoring ...
        total_words_scored += num_words_scored
        total_exceptions += num_exceptions
        tweets_with_scored_words += 1 if num_words_scored > 0 else 0
        tweets_with_exceptions += 1 if num_exceptions > 0 else 0
        if (i+1) % 100 == 0:
#           print("%d %d %s" % (i, score, tweet))
            print(".", end='', flush=True)
            DEBUG = True
        else:
            DEBUG = False
        if (i+1) % 5000 == 0:
            print()
    print((nltk.FreqDist(sentiments)).most_common(20))
    print("Tweets with scored words: %d; total words scored: %d" % \
        (tweets_with_scored_words, total_words_scored))
    print("Total word/tags with scores: %d" % len(senti_scores_dict))
    return scores

############################## Assemble Features ##############################

def assemble_features(tweets, words_in_tweets, bigrams_in_tweets, word_dict, bigram_dict, senti_scores_dict):
    index_vectors_unigrams = ngrams_to_indices(words_in_tweets, word_dict)
    index_vectors_bigrams = ngrams_to_indices(bigrams_in_tweets, bigram_dict)
    repeated_character_counts = get_repeated_character_count_tweets(tweets)
    percent_caps = get_percent_caps_tweets(tweets)
    sentiment_scores = get_sentiments_tweets(tweets, senti_scores_dict)

    features = []
    for i, uv in enumerate(index_vectors_unigrams):
        bv = index_vectors_bigrams[i]
        rc = [ repeated_character_counts[i] ]
        pc = [ percent_caps[i] ]
        ss = [ sentiment_scores[i] ]
        feature_vector = uv + bv + rc + pc + ss
        features.append(feature_vector)

    return np.array(features)

def assemble_scalar_features(tweets, sarc_freq_set, non_sarc_freq_set, senti_scores_dict):
    words_in_tweets, bigrams_in_tweets = get_unigrams_and_bigrams(tweets)
    sarc_unigrams_count, sarc_bigrams_count, \
    non_sarc_unigrams_count, non_sarc_bigrams_count = \
        get_freq_ngram_counts(words_in_tweets, bigrams_in_tweets, sarc_freq_set, non_sarc_freq_set)
    repeated_character_counts = get_repeated_character_count_tweets(tweets)
    percent_caps = get_percent_caps_tweets(tweets)
    sentiment_scores = get_sentiments_tweets(tweets, senti_scores_dict)

    features = list(zip(sarc_unigrams_count, sarc_bigrams_count, \
                        non_sarc_unigrams_count, non_sarc_bigrams_count, \
                        repeated_character_counts, percent_caps, sentiment_scores))
    return np.array(features)

# Assemble the features for training tweets
def get_train_features_tweets(tweets):
    words_in_tweets = get_tweet_words(tweets)
    bigrams_in_tweets = find_ngrams_in_tweets(2, words_in_tweets)
    word_dict, bigram_dict = get_training_vocabulary(words_in_tweets, bigrams_in_tweets)
    features = assemble_features(tweets, words_in_tweets, bigrams_in_tweets, word_dict, bigram_dict)
    return features, word_dict, bigram_dict

# Assemble the features for test tweets
def get_test_features_tweets(tweets, word_dict, bigram_dict):
    words_in_tweets = get_tweet_words(tweets)
    bigrams_in_tweets = find_ngrams_in_tweets(2, words_in_tweets)
    features = assemble_features(tweets, words_in_tweets, bigrams_in_tweets, word_dict, bigram_dict)
    return features

def scale_data(np_train_features, np_test_features):
    scaler = StandardScaler()
    scaler.fit(np_train_features)  # Don't cheat - fit only on training data
    scaled_train_features = scaler.transform(np_train_features)
    scaled_test_features = scaler.transform(np_test_features)  # apply same transformation to test data
    return scaled_train_features, scaled_test_features

# ------------------------------------------------------------------------
# Tests ---
# ------------------------------------------------------------------------

if __name__ == '__main__':

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    senti_scores_dict_name = "senti_scores_dict"

    if os.path.isfile(senti_scores_dict_name):
        with open(senti_scores_dict_name, 'rb') as f:
            senti_scores_dict = pickle.load(f)
        print('loading pickled dict')
    else:
        senti_scores_dict = {}
        print('creating new dict')

    sarcastic_tweets, non_sarcastic_tweets = load_data() #
    train_tweets, train_labels, test_tweets, test_labels, sarc_freq_set, non_sarc_freq_set = \
        get_data(sarcastic_tweets, non_sarcastic_tweets)

    USE_FULL_SET = True
    TRAIN_SIZE = 20000

    if USE_FULL_SET:
        _train_tweets = train_tweets
        _train_labels = train_labels
    else:
        # abbreviate the tweets for testing ...
        _train_tweets = train_tweets[:TRAIN_SIZE] + train_tweets[-TRAIN_SIZE:]
        _train_labels = train_labels[:TRAIN_SIZE] + train_labels[-TRAIN_SIZE:]

    np_train_features = \
        assemble_scalar_features(_train_tweets, sarc_freq_set, non_sarc_freq_set, senti_scores_dict)
    np_test_features = \
        assemble_scalar_features(test_tweets, sarc_freq_set, non_sarc_freq_set, senti_scores_dict)
    np_train_labels = np.array(_train_labels)
    np_test_labels = np.array(test_labels)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")

    scaled_train_features, scaled_test_features = \
        scale_data(np_train_features, np_test_features)

    with open(senti_scores_dict_name, 'wb') as f:
        pickle.dump(senti_scores_dict, f)

    print()
    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
    print(' MaxEnt '.center(80, "~"))
    print()

    C = 0.1

    # max_ent.cross_validate_lr(scaled_train_features, np_train_labels, C=C)

    max_ent.train_and_validate_lr(scaled_train_features, np_train_labels, scaled_test_features, np_test_labels, C=C)

    print()
    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
    print(' SVM '.center(80, "~"))
    print()

    C = 0.01

    # svm.cross_validate_svm(scaled_train_features, np_train_labels, C=C)

    svm.train_and_validate_svm(scaled_train_features, np_train_labels, scaled_test_features, np_test_labels, C=C)

    nowStr = datetime.now().strftime("%B %d, %Y %I:%M:%S %p")
    print("====" + nowStr + "====")
