import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import svm
from nltk.stem.porter import PorterStemmer
import csv
import re

## Machine Learning Online Class - Exercise 6: Support Vector Machines

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     processEmail
#     emailFeatures
#

# ==================== All function declaration ====================

def getVocabList():
    vocabList = []
    n = 1899
    with open('../../data/ex6/vocab.txt', 'r') as csvfile:
        vocabreader = csv.reader(csvfile, delimiter='\t')
        for row in vocabreader:
            vocabList.append(row[1])
    return vocabList

def processEmail(email_contents):
    vocabList = getVocabList()
    word_indices = []
    # Preprocss Email
    email_contents = email_contents.lower()
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    print('==== Processed Email ====')
    
    pattern = '[\s' + re.escape("@$/#.-:&*+=[]?!(){},'\">_<;%") + ']'
    all_words = re.split(pattern, email_contents)
    all_words = [x for x in all_words if x != '']

    stemmer = PorterStemmer()

    for w in all_words:
        w = re.sub('[^a-zA-Z0-9]', '', w)
        w = w.strip()
        w = stemmer.stem(w)
        # ============= YOUR CODE HERE =============
        # Instructions: Fill in this function to add the index of str to
        #               word_indices if it is in the vocabulary.
        try:
            idx = vocabList.index(w)
        except ValueError:
            idx = -1
        if idx is not -1:
            word_indices.append(idx)
        # ===========================================
    return word_indices

def emailFeatures(word_indices):
    n = 1899
    x = np.zeros((1,n))
    # ============= YOUR CODE HERE =============
    # Instructions: Fill in this function to return a feature vector for the
    #               given email (word_indices).
    x[0, word_indices] = 1
    # ===========================================
    return x

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    # ==================== Part 1: Email Preprocessing ====================
    
    print('Preprocessing sample email (emailSample1.txt)')
    
    data_file = '../../data/ex6/emailSample1.txt'
    with open(data_file, 'r') as f:
        file_contents = f.read()
    word_indices = processEmail(file_contents)
    
    print('Word Indices:')
    print(word_indices)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Feature Extraction ===================

    print('Extracting features from sample email (emailSample1.txt)\n')
   
    features = emailFeatures(word_indices)

    print('Length of feature vector: %d' % features.size);
    print('Number of non-zero entries: %d' % np.sum(features > 0));
        
    raw_input('Program paused. Press enter to continue')

    # =================== Part 3: Train Linear SVM for Spam Classification ===================

    data_file = '../../data/ex6/spamTrain.mat'
    mat_content = sio.loadmat(data_file)
    
    X = mat_content['X']
    y = mat_content['y']
    y = y.ravel()

    print('Training Linear SVM (Spam Classification)')
    
    C = 0.1

    model = svm.SVC(C=C, kernel='linear', max_iter=200)
    model.fit(X, y)

    print('Training Accuracy: %f' % model.score(X, y))
    
    raw_input('Program paused. Press enter to continue')

    # =================== Part 4: Test Spam Classification ===================
    
    data_file = '../../data/ex6/spamTest.mat'
    mat_content = sio.loadmat(data_file)
    
    Xtest = mat_content['Xtest']
    ytest = mat_content['ytest']
    y = y.ravel()

    print('Training Accuracy: %f' % model.score(Xtest, ytest))

    raw_input('Program paused. Press enter to continue')

    # =================== Part 5: Top Predictors of Spam ===================
    
    vocabList = getVocabList()
    coef = model.coef_.ravel()
    
    indices = coef.argsort()
    print('\nTop predictors of spam:')
    for i in xrange(-1, -16, -1):
        index = indices[i]
        print("%-15s %f" % (vocabList[index], coef[index]))
    print('\n')

    raw_input('Program paused. Press enter to continue')
    
    # =================== Part 6: Try Your Own Emails ===================

    file_name = '../../data/ex6/spamSample1.txt'
    with open(file_name, 'r') as f:
        file_contents = f.read()

    word_indices = processEmail(file_contents)
    x = emailFeatures(word_indices)    
    p = model.predict(x)
    
    print('\nProcessed %s\n\nSpam Classification: %d' % (file_name, p))
    print('(1 indicates spam, 0 indicates not spam)')

    
