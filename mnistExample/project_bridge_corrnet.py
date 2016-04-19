__author__ = 'Sarath'

import numpy
import os
import sys
import pickle

sys.path.append("../Model/")
from bridge_corr_net import *

def create_folder(folder):

    if not os.path.exists(folder):
        os.makedirs(folder)

# src_folder = sys.argv[1]+"matpic/"
# tgt_folder = sys.argv[2]

data_folder = sys.argv[1]
model_folder = sys.argv[2]
projected_views_folder = sys.argv[3]

# data_folder = "/home/akashb/Desktop/Acads/Sem2/Projects/WMT/Corr_net_author/CorrNet/mnistExample/generated_views/"
# model_folder = "/home/akashb/Desktop/Acads/Sem2/Projects/WMT/Corr_net_author/CorrNet/Model/saved_model/"
# projected_views_folder = "./TGT_DIR/projected_views/"

LABELS_KEY = "labels"

model = BridgeCorrNet()
model.load(model_folder)

# create_folder(tgt_folder+"project/")
create_folder(projected_views_folder)

all_train = pickle.load(open(data_folder + "all_train.pkl"))
# left_pivot_train = pickle.load(open(data_folder + "left_pivot_train.pkl"))
# right_pivot_train = pickle.load(open(data_folder + "right_pivot_train.pkl"))
# right_train = pickle.load(open(data_folder + "right_only_train.pkl"))
# left_train = pickle.load(open(data_folder + "left_only_train.pkl"))
# pivot_train = pickle.load(open(data_folder + "pivot_only_train.pkl"))
#
# train_left_view = numpy.concatenate([all_train[LEFT], left_pivot_train[LEFT], left_train[LEFT]])
# train_right_view = numpy.concatenate([all_train[RIGHT], right_pivot_train[RIGHT], right_train[RIGHT]])
# train_pivot_view = numpy.concatenate([all_train[PIVOT], left_pivot_train[PIVOT], right_pivot_train[PIVOT], pivot_train[PIVOT]])
#
# train_left_labels = numpy.concatenate([all_train[LABELS_KEY], left_pivot_train[LABELS_KEY], left_train[LABELS_KEY]])
# train_right_labels = numpy.concatenate([all_train[LABELS_KEY], right_pivot_train[LABELS_KEY], right_train[LABELS_KEY]])
# train_pivot_labels = numpy.concatenate([all_train[LABELS_KEY], right_pivot_train[LABELS_KEY], left_pivot_train[LABELS_KEY], pivot_train[LABELS_KEY]])

def generate_projections(model, all_views_dict):
    left_view = all_views_dict[LEFT]
    right_view = all_views_dict[RIGHT]
    pivot_view = all_views_dict[PIVOT]
    labels = all_views_dict[LABELS_KEY]

    left_projected_view = model.proj_from_left(left_view)
    right_projected_view = model.proj_from_right(right_view)
    pivot_projected_view = model.proj_from_pivot(pivot_view)

    # Projecting from test_views

    left_pivot_projection = model.proj_from_left_pivot(left_view, pivot_view)
    right_pivot_projection = model.proj_from_right_pivot(right_view, pivot_view)
    right_left_projection = model.proj_from_left_right(right_view, left_view)
    return [left_projected_view, right_projected_view, pivot_projected_view, left_pivot_projection, right_pivot_projection, right_left_projection, labels]


[train_left_projected_view, train_right_projected_view, train_pivot_projected_view,
 train_left_pivot_projection, train_right_pivot_projection, train_right_left_projection, train_labels] = generate_projections(model, all_train)

all_test = pickle.load(open(data_folder + "test_views.pkl"))

[test_left_projected_view, test_right_projected_view, test_pivot_projected_view,
 test_left_pivot_projection, test_right_pivot_projection, test_right_left_projection, test_labels] = generate_projections(model, all_test)


def write_projections(postfix, projections_list):
    [left_projected_view, right_projected_view, pivot_projected_view,
     left_pivot_projection, right_pivot_projection, right_left_projection, labels] = projections_list
    pickle.dump(left_projected_view, open(projected_views_folder +  "left_proj_view_" + postfix + ".pkl", "w"))
    pickle.dump(right_projected_view, open(projected_views_folder + "right_proj_view_" + postfix + ".pkl", "w"))
    pickle.dump(pivot_projected_view, open(projected_views_folder + "pivot_proj_view_" + postfix + ".pkl", "w"))
    pickle.dump(left_pivot_projection, open(projected_views_folder + "left_pivot_proj_view_" + postfix + ".pkl", "w"))
    pickle.dump(right_pivot_projection, open(projected_views_folder + "right_pivot_proj_view_" + postfix + ".pkl", "w"))
    pickle.dump(right_left_projection, open(projected_views_folder + "right_left_proj_view_" + postfix + ".pkl", "w"))
    pickle.dump(labels, open(projected_views_folder + "labels_" + postfix + ".pkl", "w"))


write_projections(postfix="train",
                  projections_list=[train_left_projected_view, train_right_projected_view, train_pivot_projected_view,
                                    train_left_pivot_projection, train_right_pivot_projection,
                                    train_right_left_projection, train_labels])
write_projections(postfix="test", projections_list=[test_left_projected_view, test_right_projected_view, test_pivot_projected_view,
                                                    test_left_pivot_projection, test_right_pivot_projection, test_right_left_projection, test_labels])


#
# # Projection from test_views
#
# mat = numpy.load(src_folder+"train/view1.npy")
# new_mat = model.proj_from_left(mat)
# numpy.save(tgt_folder+"project/train-view1",new_mat)
#
# mat = numpy.load(src_folder+"train/view2.npy")
# # new_mat = model.proj_from_right(mat)
# new_mat = model.proj_from_pivot(mat)
# numpy.save(tgt_folder+"project/train-view2",new_mat)
#
# mat = numpy.load(src_folder+"train/labels.npy")
# numpy.save(tgt_folder+"project/train-labels",mat)
#
#
# mat = numpy.load(src_folder+"valid/view1.npy")
# new_mat = model.proj_from_left(mat)
# numpy.save(tgt_folder+"project/valid-view1",new_mat)
#
# mat = numpy.load(src_folder+"valid/view2.npy")
# # new_mat = model.proj_from_right(mat)
# new_mat = model.proj_from_pivot(mat)
# numpy.save(tgt_folder+"project/valid-view2",new_mat)
#
# mat = numpy.load(src_folder+"valid/labels.npy")
# numpy.save(tgt_folder+"project/valid-labels",mat)
#
#
# mat = numpy.load(src_folder+"test/view1.npy")
# new_mat = model.proj_from_left(mat)
# numpy.save(tgt_folder+"project/test-view1",new_mat)
#
# mat = numpy.load(src_folder+"test/view2.npy")
# # new_mat = model.proj_from_right(mat)
# new_mat = model.proj_from_pivot(mat)
# numpy.save(tgt_folder+"project/test-view2",new_mat)
#
# mat = numpy.load(src_folder+"test/labels.npy")
# numpy.save(tgt_folder+"project/test-labels",mat)

