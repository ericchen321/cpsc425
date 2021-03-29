#Starter code prepared by Borna Ghotbi, Polina Zablotskaia, and Ariel Shann for Computer Vision
#based on a MATLAB code by James Hays and Sam Birch 

import numpy as np
from util import load, build_vocabulary, get_bags_of_sifts, convert_int_to_classname
from classifiers import nearest_neighbor_classify, svm_classify
import matplotlib.pyplot as plt
import pickle
import os
import glob
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix

#For this assignment, you will need to report performance for sift features on two different classifiers:
# 1) Bag of sift features and nearest neighbor classifier
# 2) Bag of sift features and linear SVM classifier

#For simplicity you can define a "num_train_per_cat" vairable, limiting the number of
#examples per category. num_train_per_cat = 100 for intance.

#Sample images from the training/testing dataset. 
#You can limit number of samples by using the n_sample parameter.

if(len(sys.argv)!=6):
        print("Wrong arguments, please check comments in the script for usage")
        sys.exit(1)

build_new_model = sys.argv[1]
vocab_size = int(sys.argv[2])
n_neighbors = int(sys.argv[3])
C = float(sys.argv[4])
repeat = int(sys.argv[5])

print(f"Experiment config:\nbuild_new_model={build_new_model}\nvocab_size={vocab_size}\nn_neighbors={n_neighbors}\nC={C}")

''' Step 0: Load data in '''

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
    plt.savefig(f'figs/histos/avghistog_vocab_size={vocab_size}_cidx={classindex}_cname={classname}_repeat={repeat}.png')
        
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
train_labels_classnames = convert_int_to_classname(train_labels, train_classnames)
model_knn, pred_labels_classnames_knn = nearest_neighbor_classify(train_image_feats, train_labels_classnames, test_image_feats, n_neighbors)
#print(pred_labels_classnames_knn)

print('Using support vector machine to predict test set categories\n')
#TODO: YOU CODE svm_classify function from classifers.py
model_svm, pred_labels_classnames_svm = svm_classify(train_image_feats, train_labels_classnames, test_image_feats, C)


print('---Evaluation---\n')
# Step 3: Build a confusion matrix and score the recognition system for 
#         each of the classifiers.
# In this step you will be doing evaluation. 
# TODO: 1) Calculate the total accuracy of your model by counting number
#   of true positives and true negatives over all. 
# calculate accuracy of KNN classifier
test_labels_classnames = convert_int_to_classname(test_labels, test_classnames)
knn_acc = accuracy_score(test_labels_classnames, pred_labels_classnames_knn) * 100.0
print(f"KNN classifier accuracy={knn_acc:.2f}%, vocab_size={vocab_size}, n_neighbors={n_neighbors}")
svm_acc = accuracy_score(test_labels_classnames, pred_labels_classnames_svm) * 100.0
print(f"SVM classifier accuracy={svm_acc:.2f}%, vocab_size={vocab_size}, C={C}")

# TODO: 2) Build a Confusion matrix and visualize it. 
#   You will need to convert the one-hot format labels back
#   to their category name format.
conf_mat_knn = confusion_matrix(test_labels_classnames, pred_labels_classnames_knn)
assert np.linalg.norm(np.sum(conf_mat_knn) - test_labels.shape[0]) < 1e-9
plot_confusion_matrix(model_knn, test_image_feats, test_labels_classnames, xticks_rotation="vertical")
plt.title(f'Confusion matrix, KNN, vocab_size={vocab_size}, n_neighbors={n_neighbors}')
plt.savefig(f'figs/conf_mats/conf_mat_knn_vocab_size={vocab_size}_n_neighbors={n_neighbors}_repeat={repeat}.png', bbox_inches='tight')

conf_mat_svm = confusion_matrix(test_labels_classnames, pred_labels_classnames_svm)
assert np.linalg.norm(np.sum(conf_mat_svm) - test_labels.shape[0]) < 1e-9
plot_confusion_matrix(model_svm, test_image_feats, test_labels_classnames, xticks_rotation="vertical")
plt.title(f'Confusion matrix, SVM, vocab_size={vocab_size}, C={C}')
plt.savefig(f'figs/conf_mats/conf_mat_svm_vocab_size={vocab_size}_C={C}_repeat={repeat}.png', bbox_inches='tight')

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
