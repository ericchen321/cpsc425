#Starter code prepared by Borna Ghotbi, Polina Zablotskaia, and Ariel Shann for Computer Vision
#based on a MATLAB code by James Hays and Sam Birch 

import numpy as np
from util import load, build_vocabulary, get_bags_of_sifts, convert_label_to_integer, convert_label_to_one_hot
from classifiers import nearest_neighbor_classify, svm_classify
import matplotlib.pyplot as plt
import pickle
import os
import glob

#For this assignment, you will need to report performance for sift features on two different classifiers:
# 1) Bag of sift features and nearest neighbor classifier
# 2) Bag of sift features and linear SVM classifier

#For simplicity you can define a "num_train_per_cat" vairable, limiting the number of
#examples per category. num_train_per_cat = 100 for intance.

#Sample images from the training/testing dataset. 
#You can limit number of samples by using the n_sample parameter.

build_new_model = True

train_image_paths = None
train_labels = None
train_classnames = None
test_image_paths = None
test_labels = None
test_classnames = None

if build_new_model:
    print('Getting paths and labels for all train and test data\n')
    train_image_paths, train_labels, train_classnames = load("sift/train")
    np.save('train_image_paths', train_image_paths)
    np.save('train_labels', train_labels)
    np.save('train_classnames', train_classnames)
    test_image_paths, test_labels, test_classnames = load("sift/test")
    np.save('test_image_paths', test_image_paths)
    np.save('test_labels', test_labels)
    np.save('test_classnames', test_classnames)
else:
    print('Retrieving paths and labels for all train and test data\n')
    train_image_paths = np.load('train_image_paths.npy')
    train_labels = np.load('train_labels.npy')
    train_classnames = np.load('train_classnames.npy')
    test_image_paths = np.load('test_image_paths.npy')
    test_labels = np.load('test_labels.npy')
    test_classnames = np.load('test_classnames.npy')
assert np.array_equal(train_classnames, test_classnames)

''' Step 1: Represent each image with the appropriate feature
 Each function to construct features should return an N x d matrix, where
 N is the number of paths passed to the function and d is the 
 dimensionality of each image representation. See the starter code for
 each function for more details. '''

vocab_size = 200
train_image_feats = None
test_image_feats = None
kmeans = None

if build_new_model:      
    print('Extracting SIFT features\n')
    # NOTE: You code build_vocabulary function in util.py
    print('Buildng vocabulary')
    kmeans = build_vocabulary(train_image_paths, vocab_size=vocab_size)
    # model save and load from
    # https://stackoverflow.com/questions/54879434/how-to-use-the-pickle-to-save-sklearn-model
    pickle.dump(kmeans, open("kmeans.pkl", "wb"))
    # NOTE: You code get_bags_of_sifts function in util.py
    print('Buildng histograms for training images')
    train_image_feats = get_bags_of_sifts(train_image_paths, kmeans)
    np.save('train_image_feats', train_image_feats)
    print('Buildng histograms for testing images')
    test_image_feats = get_bags_of_sifts(test_image_paths, kmeans)
    np.save('test_image_feats', test_image_feats)
else:
    print('Loading KMeans model')
    kmeans = pickle.load(open("kmeans.pkl", "rb"))
    print('Loading histograms for training images')
    train_image_feats = np.load('train_image_feats.npy')
    print('Loading histograms for testing images')
    test_image_feats = np.load('test_image_feats.npy')
    
#TODO: visualize histograms of 15 categories
assert train_image_feats.shape[0] == train_labels.shape[0]
# initialize sum histograms as a dictionary of 15 histogram vectors
sum_histos = {}
# iterate over all training images and sum up histograms
for i in range(0, train_image_feats.shape[0]):
    # extract image class index
    classindex = train_labels[i]
    # extract image histogram
    feats = train_image_feats[i, :] # (vocab_size, )
    # sum up histograms
    if sum_histos.get(classindex) is None:
        sum_histos.update({classindex: feats})
    else:
        sum_histos[classindex] = sum_histos[classindex] + feats
assert len(sum_histos.keys()) == 15
# average the sum histograms to produce per-class average
avg_histos = {}
for classindex, sum_feats in sum_histos.items():
    # here we take advantage of the fact that for each class, we have
    # 100 training instances
    avg_feats = sum_feats / 100.0
    assert np.linalg.norm(np.sum(avg_feats, axis=0)-1.0)<1e-9
    avg_histos.update({classindex: avg_feats})
# plot the histograms
# for each class, plot its average histogram
for classindex, histo in avg_histos.items():
    vocab_indices = np.arange(0, vocab_size)
    assert vocab_indices.shape == histo.shape
    fig_histo, ax_histo = plt.subplots()
    ax_histo.bar(vocab_indices, histo)
    classname = train_classnames[int(classindex)]
    ax_histo.set_title(f'Average histogram for class: index={classindex}, name={classname}; {vocab_size} vocabs')
    #plt.show()
    plt.savefig(f'avghistog_vocab_size={vocab_size}_cidx={classindex}_cname={classname}.png')
        
#If you want to avoid recomputing the features while debugging the
#classifiers, you can either 'save' and 'load' the extracted features
#to/from a file.

''' Step 2: Classify each test image by training and using the appropriate classifier
 Each function to classify test features will return an N x l cell array,
 where N is the number of test cases and each entry is a string indicating
 the predicted one-hot vector for each test image. See the starter code for each function
 for more details. '''

print('Using nearest neighbor classifier to predict test set categories\n')
#TODO: YOU CODE nearest_neighbor_classify function from classifers.py
pred_labels_knn = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
print(pred_labels_knn)

print('Using support vector machine to predict test set categories\n')
#TODO: YOU CODE svm_classify function from classifers.py
#pred_labels_svm_one_hot = svm_classify(train_image_feats, train_labels, test_image_feats)



print('---Evaluation---\n')
# Step 3: Build a confusion matrix and score the recognition system for 
#         each of the classifiers.
# In this step you will be doing evaluation. 
# TODO: 1) Calculate the total accuracy of your model by counting number
#   of true positives and true negatives over all. 
# calculate accuracy of KNN classifier
test_labels_one_hot = convert_label_to_one_hot(test_labels, 15)
pred_labels_knn_one_hot = convert_label_to_one_hot(pred_labels_knn, 15)
knn_correct_preds = np.sum(np.multiply(test_labels_one_hot, pred_labels_knn_one_hot))
print(f"Number of correct predictions: {knn_correct_preds}")
knn_acc = knn_correct_preds / test_labels_one_hot.shape[0]
print(f"KNN classifier accuracy: {knn_acc}")

# TODO: 2) Build a Confusion matrix and visualize it. 
#   You will need to convert the one-hot format labels back
#   to their category name format.


# Interpreting your performance with 100 training examples per category:
#  accuracy  =   0 -> Your code is broken (probably not the classifier's
#                     fault! A classifier would have to be amazing to
#                     perform this badly).
#  accuracy ~= .10 -> Your performance is chance. Something is broken or
#                     you ran the starter code unchanged.
#  accuracy ~= .40 -> Rough performance with bag of SIFT and nearest
#                     neighbor classifier. 
#  accuracy ~= .50 -> You've gotten things roughly correct with bag of
#                     SIFT and a linear SVM classifier.
#  accuracy >= .60 -> You've added in spatial information somehow or you've
#                     added additional, complementary image features. This
#                     represents state of the art in Lazebnik et al 2006.
#  accuracy >= .85 -> You've done extremely well. This is the state of the
#                     art in the 2010 SUN database paper from fusing many 
#                     features. Don't trust this number unless you actually
#                     measure many random splits.
#  accuracy >= .90 -> You used modern deep features trained on much larger
#                     image databases.
#  accuracy >= .96 -> You can beat a human at this task. This isn't a
#                     realistic number. Some accuracy calculation is broken
#                     or your classifier is cheating and seeing the test
#                     labels.