__author__ = 'Akash'

import os
import sys
import pickle

sys.path.append("../Model/")
from bridge_corr_net import *

def create_folder(folder):

    if not os.path.exists(folder):
        os.makedirs(folder)

data_folder = sys.argv[1]
model_folder = sys.argv[2]
projected_views_folder = sys.argv[3]

LABELS_KEY = "labels"

model = BridgeCorrNet()
model.load(model_folder)

create_folder(projected_views_folder)

all_train = pickle.load(open(data_folder + "all_train.pkl"))

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