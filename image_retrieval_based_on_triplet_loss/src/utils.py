from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import manhattan_distances

import matplotlib.pyplot as plt
from matplotlib import offsetbox

from keras import backend as K


def triplet_loss(y_true, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), 
                            y_pred + margin))
                            #K.square(y_pred[:,0,0]) - 0.5*(K.square(y_pred[:,1,0]) + K.square(y_pred[:,2,0])) + margin))


def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])


def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def find_l2_distances(X,Y):
    intersection = -2.* np.dot(X,Y.T)
    X_sum = np.sum(X**2,axis=1)
    Y_sum = np.sum(Y**2,axis=1)
    XY_sum = X_sum[:, np.newaxis] + Y_sum
    return XY_sum + intersection


def find_cos_distances(X,Y):
    return (1.-np.dot(X, Y.T))/2.0


def max_distances(X,Y, dist_fun):
    results = np.zeros( (X.shape[0], Y.shape[0]), dtype=np.float32 )
    if dist_fun == 'max_l1':
        return cdist(X, Y, 'chebyshev')
    else: raise 'not implemented'


def recall_at_kappa_leave_one_out(test_emb, test_id, kappa, dist):
    unique_ids, unique_counts = np.unique(test_id,return_counts=True)
    unique_ids = unique_ids[unique_counts >= 2]
    good_test_indices = np.in1d(test_id,unique_ids)
    valid_test_embs = test_emb[good_test_indices]
    valid_test_ids = test_id[good_test_indices]
    n_correct_at_k = np.zeros(kappa)
    if dist == 'cos':
        distances = find_cos_distances(valid_test_embs,test_emb)
    elif dist == 'l2':
        distances = find_l2_distances(valid_test_embs, test_emb)
    elif dist == 'l1':
        distances = manhattan_distances(valid_test_embs, test_emb)
    elif dist == 'max_l1' or dist == 'max_l2':
        distances = max_distances(valid_test_embs, test_emb, dist)
    for idx, valid_test_id in enumerate(valid_test_ids):
        k_sorted_indices = np.argsort(distances[idx])[1:]
        first_correct_position = np.where(test_id[k_sorted_indices] == valid_test_id)[0][0]
        if first_correct_position < kappa:
            n_correct_at_k[first_correct_position:] += 1
    return 1.*n_correct_at_k / len(valid_test_ids)


def recall_at_kappa_support_query(x_support, y_support, x_query, y_query, kappa, dist):
    n_correct_at_k = np.zeros(kappa)
    if dist == 'cos':
        distances = find_cos_distances(x_query, x_support)
    elif dist == 'l2':
        distances = find_l2_distances(x_query, x_support)
    elif dist == 'l1':
        distances = manhattan_distances(x_query, x_support)
    elif dist == 'max_l1' or dist == 'max_l2':
        distances = max_distances(x_query, x_support, dist)
    for idx, valid_test_id in enumerate(y_query):
        k_sorted_indices = np.argsort(distances[idx])
        first_correct_position = np.where(y_support[k_sorted_indices] == valid_test_id)[0][0]
        if first_correct_position < kappa:
            n_correct_at_k[first_correct_position:] += 1
    return 1.*n_correct_at_k / len(y_query)


def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    #ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            #imagebox = offsetbox.AnnotationBbox(
                #offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                #offsetbox.OffsetImage(x_train[i], cmap=plt.cm.gray_r),
                #X[i])
            #ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
        
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    I refer to `https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1`
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)