import numpy as np
import os
import glob
from sklearn.cluster import KMeans

def build_vocabulary(image_paths, vocab_size):
    """ Sample SIFT descriptors, cluster them using k-means, and return the fitted k-means model.
    NOTE: We don't necessarily need to use the entire training dataset. You can use the function
    sample_images() to sample a subset of images, and pass them into this function.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    vocab_size: the number of clusters desired.
    
    Returns
    -------
    kmeans: the fitted k-means clustering model.
    """
    n_image = len(image_paths)

    # Since want to sample tens of thousands of SIFT descriptors from different images, we
    # calculate the number of SIFT descriptors we need to sample from each image.
    n_each = int(np.ceil(10000 / n_image))

    # Initialize an array of features, which will store the sampled descriptors
    # keypoints = np.zeros((n_image * n_each, 2))
    descriptors = np.zeros((n_image * n_each, 128))

    for i, path in enumerate(image_paths):
        # Load features from each image
        features = np.loadtxt(path, delimiter=',',dtype=float)
        #sift_keypoints = features[:, :2] # (num_descriptors, 2)
        sift_descriptors = features[:, 2:] # (num_descriptors, 128)

        # TODO: Randomly sample n_each descriptors from sift_descriptor and store them into descriptors
        sampled_indices = np.random.randint(0, sift_descriptors.shape[0], size=(n_each,))
        sift_descriptors_sampled = sift_descriptors[sampled_indices, :]
        assert sift_descriptors_sampled.shape == (n_each, 128)
        descriptors[i*n_each:(i+1)*n_each, :] = sift_descriptors_sampled

    # TODO: pefrom k-means clustering to cluster sampled sift descriptors into vocab_size regions.
    # You can use KMeans from sci-kit learn.
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    kmeans = KMeans(n_clusters=vocab_size, random_state=0).fit(descriptors)
    
    return kmeans
    
def get_bags_of_sifts(image_paths, kmeans):
    """ Represent each image as bags of SIFT features histogram.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    kmeans: k-means clustering model with vocab_size centroids.

    Returns
    -------
    image_feats: an (n_image, vocab_size) matrix, where each row is a histogram.
    """
    n_image = len(image_paths)
    vocab_size = kmeans.cluster_centers_.shape[0]

    image_feats = np.zeros((n_image, vocab_size)) #(n_image, vocab_size)

    for i, path in enumerate(image_paths):
        # Load features from each image
        features = np.loadtxt(path, delimiter=',',dtype=float)

        # TODO: Assign each feature to the closest cluster center
        # Again, each feature consists of the (x, y) location and the 128-dimensional sift descriptor
        # You can access the sift descriptors part by features[:, 2:]
        sift_descriptors = features[:, 2:] # (num_descriptors, 128)
        sift_word_indices = kmeans.predict(sift_descriptors, sample_weight=None) # (num_descriptors, )

        # TODO: Build a histogram normalized by the number of descriptors
        histo = np.bincount(sift_word_indices, minlength=vocab_size) # (vocab_size, )
        assert histo.shape[0] == vocab_size
        histo = histo / sift_word_indices.shape[0]
        image_feats[i, :] = histo

    return image_feats

def convert_label_to_one_hot(feats, num_classes):
    """ Convert class labels from integer to one-hot format.

    Parameters
    ----------
    feats: (N, ) array of class indices where N is the number of samples
    num_classes: Total number of classes

    Returns
    ----------
    feats_one_hot: (N, num_classes) array of one-hot encoded labels of N
                   samples.
    """
    feats_one_hot = np.zeros((feats.shape[0], num_classes))
    feats_one_hot[range(0, feats.shape[0]), feats[range(0, feats.shape[0])].astype(np.int)] = 1
    return feats_one_hot

def convert_label_to_integer(feats_one_hot):
    """ Convert class labels from one-hot to integer format.

    Parameters
    ----------
    feats_one_hot: (N, num_classes) array of one-hot encoded labels of N
                   samples.
    Returns
    ----------
    feats: (N, ) array of integer encoded labels of N samples.
    """
    feats = np.argmax(feats_one_hot, axis=1)
    return feats

def load(ds_path):
    """ Load from the training/testing dataset.

    Parameters
    ----------
    ds_path: path to the training/testing dataset.
             e.g., sift/train or sift/test 
    
    Returns
    -------
    image_paths: a (n_sample, 1) array that contains the paths to the descriptors. 
    labels: class labels corresponding to each image
    classnames: a (l, ) array of classnames where l is the number of classes; 
                classnames[labels[i]] allows finding the classname corresponding to
                each class index.
    """
    # Grab a list of paths that matches the pathname
    files = glob.glob(os.path.join(ds_path, "*", "*.txt"))
    n_files = len(files)
    image_paths = np.asarray(files)
 
    # Get class labels
    classes = glob.glob(os.path.join(ds_path, "*"))
    labels = np.zeros(n_files)

    for i, path in enumerate(image_paths):
        folder, fn = os.path.split(path)
        labels[i] = np.argwhere(np.core.defchararray.equal(classes, folder))[0,0]
    
    classnames = []
    for classpath in classes:
        classnames.append(os.path.basename(classpath))

    # Randomize the order
    idx = np.random.choice(n_files, size=n_files, replace=False)
    image_paths = image_paths[idx]
    labels = labels[idx]

    return image_paths, labels, np.asarray(classnames)


if __name__ == "__main__":
    paths, labels = load("sift/train")
    #build_vocabulary(paths, 10)
