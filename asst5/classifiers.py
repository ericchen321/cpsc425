 #Starter code prepared by Borna Ghotbi for computer vision
 #based on MATLAB code by James Hay

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

'''This function will predict the category for every test image by finding
the training image with most similar features. Instead of 1 nearest
neighbor, you can vote based on k nearest neighbors which will increase
performance (although you need to pick a reasonable value for k). '''

def nearest_neighbor_classify(train_image_feats, train_labels_classnames, test_image_feats, n_neighbors=7):

    '''
    Parameters
        ----------
        train_image_feats:  is an N x d matrix, where d is the dimensionality of the feature representation.
        train_labels_classnames: is an (N, ) cell array, where each entry is a string 
        			             indicating the ground truth class of each training image.
    	test_image_feats: is an M x d matrix, where d is the dimensionality of the
    					  feature representation. You can assume M = N unless you've modified the starter code.
        
    Returns
        -------
        knn: fitted model
    	predicted_labels_classnames: an (M, ) cell array, where each row is a one-hot vector 
                                     indicating the predicted category for each test image.

    Usefull funtion:
    	
    	# You can use knn from sci-kit learn.
        # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    '''
    # referenced implementation examples from https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_image_feats, train_labels_classnames)
    predicted_labels_classnames = knn.predict(test_image_feats) # (M, )
    assert predicted_labels_classnames.shape == (test_image_feats.shape[0], )
    return knn, predicted_labels_classnames



'''This function will train a linear SVM for every category (i.e. one vs all)
and then use the learned linear classifiers to predict the category of
very test image. Every test feature will be evaluated with all 15 SVMs
and the most confident SVM will "win". Confidence, or distance from the
margin, is W*X + B where '*' is the inner product or dot product and W and
B are the learned hyperplane parameters. '''

def svm_classify(train_image_feats, train_labels_classnames, test_image_feats, C):

    '''
    Parameters
        ----------
        train_image_feats:  is an N x d matrix, where d is the dimensionality of the feature representation.
        train_labels_classnames: is an (N, ) cell array, where each entry is a string 
        			             indicating the ground truth class of each training image.
    	test_image_feats: is an M x d matrix, where d is the dimensionality of the
    					  feature representation. You can assume M = N unless you've modified the starter code.
        
    Returns
        -------
        svm: fitted model
    	predicted_labels_classnames: an (M, ) cell array, where each row is a one-hot vector 
                                     indicating the predicted category for each test image.

    Usefull funtion:
    	
    	# You can use svm from sci-kit learn.
        # Reference: https://scikit-learn.org/stable/modules/svm.html

    '''

    # referenced implementation examples from https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.fit
    svm = LinearSVC(random_state=0, tol=1e-5, C=C, multi_class='ovr', max_iter=2000)
    svm.fit(train_image_feats, train_labels_classnames)
    predicted_labels_classnames = svm.predict(test_image_feats) # (M, )
    assert predicted_labels_classnames.shape == (test_image_feats.shape[0], )
    return svm, predicted_labels_classnames

