__author__ = 'Sarath'
import sys
import numpy as np

sys.path.append("../Model/")
# from Model.bridge_corr_net import *
from bridge_corr_net import *
from mnist import *

MNIST_DATA_PATH = "../mnist_images/"

# src_folder = sys.argv[1]+"matpic1/"
# tgt_folder = sys.argv[2]

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

def get_left_view(image):
    left_view = []
    for row in image:
        left_view.append(row[:len(row)/2])
    left_view = np.array(np.array(left_view).flatten())
    return(np.array(left_view))

def get_right_view(image):
    right_view = []
    for row in image:
        right_view.append(row[len(row)/2:])
    right_view = np.array(np.array(right_view).flatten())
    return(right_view)

def get_noise_view(image):
    rand_noise = np.random.uniform(high=0.01, low=-0.01, size=(image.shape[0], image.shape[1]))
    noise_view = np.array(np.array(image + rand_noise).flatten())
    return(noise_view)

def get_views(mnist_images, views=[LEFT, RIGHT, PIVOT]):
    left_views = []
    right_views = []
    noise_views = []
    for image in mnist_images:
        if LEFT in views:
            left_views.append(get_left_view(image))
        if RIGHT in views:
            right_views.append(get_right_view(image))
        if PIVOT in views:
            noise_views.append(get_noise_view(image))
    ret_list = []
    for view in views:
        if view == LEFT:
            ret_list.append(np.array(left_views))
        if view == RIGHT:
            ret_list.append(np.array(right_views))
        if view == PIVOT:
            ret_list.append(np.array(noise_views))
    return(ret_list)

train_mnist_images, train_mnist_labels = load_mnist(dataset="training", digits=None, path=MNIST_DATA_PATH, asbytes=False, selection=None, return_labels=True, return_indices=False)

tot_train_len = len(train_mnist_images)

[all_train_left, all_train_right, all_train_pivot] = get_views(train_mnist_images[:tot_train_len/6], views=[LEFT, RIGHT, PIVOT])
print(all_train_left.shape, all_train_right.shape, all_train_pivot.shape)
all_train = {LEFT: all_train_left, RIGHT: all_train_right, PIVOT: all_train_pivot}
n_visible_left = len(all_train_left[0])
n_visible_right = len(all_train_right[0])
n_visible_pivot = len(all_train_pivot[0])

[left_pivot_left, left_pivot_pivot] = get_views(train_mnist_images[tot_train_len/6:tot_train_len * 2/6], views=[LEFT, PIVOT])
print(left_pivot_left.shape, left_pivot_pivot.shape)
left_pivot = {LEFT: left_pivot_left, PIVOT: left_pivot_pivot}

[right_pivot_right, right_pivot_pivot] = get_views(train_mnist_images[tot_train_len*2/6 : tot_train_len * 3/6], views=[RIGHT, PIVOT])
print(right_pivot_right.shape, right_pivot_pivot.shape)
right_pivot = {RIGHT: right_pivot_right, PIVOT: right_pivot_pivot}

[right_only] = get_views(train_mnist_images[tot_train_len*3/6 : tot_train_len * 4/6], views=[RIGHT])
print(right_only.shape)
[left_only] = get_views(train_mnist_images[tot_train_len*4/6 : tot_train_len * 5/6], views=[LEFT])
print(left_only.shape)
[pivot_only] = get_views(train_mnist_images[tot_train_len*5/6 : ], views=[PIVOT])
print(pivot_only.shape)

# trainBridgeCorrNet(src_folder=src_folder, tgt_folder=tgt_folder, batch_size=batch_size,
#              training_epochs=training_epochs, l_rate=l_rate, optimization=optimization,
#              tied=tied, n_visible_left=n_visible_left, n_visible_right=n_visible_right, n_visible_pivot=n_visible_right,
#              n_hidden=n_hidden, lamda=lamda, hidden_activation=hidden_activation,
#              output_activation=output_activation, loss_fn=loss_fn)

trainBridgeCorrNet_with_mats(left_pivot_train=left_pivot, right_pivot_train=right_pivot, right_train=right_only, left_train=left_only,
                                 pivot_train=pivot_only, all_train=all_train, batch_size=batch_size, training_epochs=training_epochs,
                                 l_rate=l_rate, optimization=optimization, tied=tied, n_hidden=n_hidden, n_visible_left=n_visible_left,
                                 n_visible_right=n_visible_right,
                                 n_visible_pivot=n_visible_pivot, lamda=lamda,
                                 hidden_activation=hidden_activation,
                                 output_activation=output_activation, loss_fn=loss_fn, tgt_folder="./")