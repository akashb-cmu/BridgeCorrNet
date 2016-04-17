__author__ = 'Sarath'

import sys

sys.path.append("../Model/")
# from Model.bridge_corr_net import *
from bridge_corr_net import *
from mnist import *

MNIST_DATA_PATH = "../mnist_images/"

src_folder = sys.argv[1]+"matpic1/"
tgt_folder = sys.argv[2]

batch_size = 100
training_epochs = 50
l_rate = 0.01
optimization = "rmsprop"
tied = True
n_visible_left = 392
n_visible_right = 392
n_hidden = 50
lamda = 2
hidden_activation = "sigmoid"
output_activation = "sigmoid"
loss_fn = "squarrederror"

# train_mnist_images, train_mnist_labels = load_mnist(dataset="training", digits=None, path=MNIST_DATA_PATH, asbytes=False, selection=None, return_labels=True, return_indices=False)


trainBridgeCorrNet(src_folder=src_folder, tgt_folder=tgt_folder, batch_size=batch_size,
             training_epochs=training_epochs, l_rate=l_rate, optimization=optimization,
             tied=tied, n_visible_left=n_visible_left, n_visible_right=n_visible_right, n_visible_pivot=n_visible_right,
             n_hidden=n_hidden, lamda=lamda, hidden_activation=hidden_activation,
             output_activation=output_activation, loss_fn=loss_fn)

