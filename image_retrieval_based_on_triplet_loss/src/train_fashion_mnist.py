import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

import keras
from keras.datasets import fashion_mnist
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from model import TripletNet
from triplets import TripletGenerator
from utils import recall_at_kappa_support_query, plot_embedding, find_l2_distances, show_images

# Load the pre-shuffled train and test dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Pre-processing
x_train = np.expand_dims(x_train, axis=3); y_train = np.expand_dims(y_train, axis=3)
x_test = np.expand_dims(x_test, axis=3); y_test = np.expand_dims(y_test, axis=3)

# Break training set into training and validation sets
(x_train, x_valid, x_supp) = x_train[5000:], x_train[:3000], x_train[3000:5000]
(y_train, y_valid, y_supp) = y_train[5000:], y_train[:3000], y_train[3000:5000]

input_size = (28, 28, 1)
embedding_dimensions = 128
batch_size = 256
kappa = 1
dist = 'l2'

gen = TripletGenerator()
train_stream = gen.flow(x_train, y_train, batch_size=batch_size)
valid_stream = gen.flow(x_valid, y_valid, batch_size=batch_size)

t = TripletNet(shape=input_size, dimensions=embedding_dimensions,
                               pretrained=False, learning_rate=0.001)
t.summary()

#t.model.load_weights('model128_bin_batch2.weights.best.fashion-mnist.hdf5', by_name=False)
checkpoint = ModelCheckpoint(
        filepath='model128_bin_batch2.weights.best.fashion-mnist.hdf5',
        verbose=1, save_best_only=True
)


for i in range(1):
    t.model.fit_generator(
            train_stream, 2500, epochs=1, verbose=1,
            callbacks=[checkpoint],
            validation_data=valid_stream, validation_steps=20)

# Check Recall@K
#m, orig, residual = t.build_triplet_model(shape=input_size, dimensions=embedding_dimensions)
get_3rd_layer_output = K.function([t.orig, K.learning_phase()], [t.residual])
x_supp_emb = get_3rd_layer_output([x_supp, 0])[0]
x_valid_emb = get_3rd_layer_output([x_valid, 0])[0]

test_recall_at_one = np.mean(recall_at_kappa_support_query(x_supp_emb, y_supp, 
                                                       x_valid_emb, y_valid, 
                                                       kappa=kappa, dist=dist))
print("[INFO] Recall@{} is {}".format(kappa, test_recall_at_one))

# Check retrieval results (e.g. Recall@K)
if dist == 'l2':
    distances = find_l2_distances(x_valid_emb, x_supp_emb)
    
ind = 100
dist_sample = np.min(distances[ind])
dist_index = np.argsort(distances[ind])
    
# Query image
plt.title('Query image')
plt.imshow(x_valid[ind,:,:,0])
plt.axis('off')
plt.show()

# Retrieval images
show_images([x_supp[dist_index[0],:,:,0],
             x_supp[dist_index[1],:,:,0],
             x_supp[dist_index[2],:,:,0],
             x_supp[dist_index[3],:,:,0],
             x_supp[dist_index[4],:,:,0]], 
             titles=['The best retrieval image',
                     'Second retrieval image',
                     'Third retrieval image',
                     'Fourth retrieval image',
                     'Fifth retireval image'])


# Check retrieval results (e.g. Recall@K)
if dist == 'l2':
    distances = find_l2_distances(x_valid_emb, x_supp_emb)
    
ind = 1111
dist_sample = np.min(distances[ind])
dist_index = np.argmin(distances[ind])

plt.imshow(x_valid[ind,:,:,0])
plt.show()
plt.imshow(x_supp[dist_index,:,:,0])
plt.show()


# Qualitative results using various dim. reduction techniques
m, orig, residual = t.build_triplet_model(shape=input_size, 
                                          dimensions=embedding_dimensions)

get_3rd_layer_output = K.function([orig, K.learning_phase()],
                                  [residual])
layer_output = get_3rd_layer_output([x_train[:1000], 0])[0]


# 1. Random 2D映射
print("Computing random projection")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(layer_output)
plot_embedding(X_projected, np.squeeze(y_train[:1000], axis=1), "Random Projection of the digits")


# 2. PCA降维
print("Computing PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(layer_output)
plot_embedding(X_pca, np.squeeze(y_train[:1000], axis=1),
               "Principal Components projection of the digits (time %.2fs)" %
               (time() - t0))


# 3. 数据集Spectral embedding
print("Computing Spectral embedding")
embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                      eigen_solver="arpack")
t0 = time()
X_se = embedder.fit_transform(layer_output)

plot_embedding(X_se, np.squeeze(y_train[:1000], axis=1),
               "Spectral embedding of the digits (time %.2fs)" %
               (time() - t0))


# 4. t-SNE可视化
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(layer_output)

plot_embedding(X_tsne, np.squeeze(y_train[:1000], axis=1),
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))

plt.show()