__author__ = 'Sarath'

import numpy
import math
from sklearn import svm
from sklearn.metrics import accuracy_score
import sys
import pickle

# projected_views_folder = "./TGT_DIR/projected_views/"
projected_views_folder = sys.argv[1]

def svm_classifier(train_x, train_y, valid_x, valid_y, test_x, test_y):

    clf = svm.LinearSVC()
    clf.fit(train_x,train_y)
    pred = clf.predict(valid_x)
    va = accuracy_score(numpy.ravel(valid_y),numpy.ravel(pred))
    pred = clf.predict(test_x)
    ta = accuracy_score(numpy.ravel(test_y),numpy.ravel(pred))
    return va, ta

def transfer_learning_5fold(folder):

    view1 = numpy.load(folder+"test-view1.npy")
    view2 = numpy.load(folder+"test-view2.npy")
    labels = numpy.load(folder+"test-labels.npy")

    perp = len(view1)/5

    print "view1 to view2"

    acc = 0
    for i in range(0,5):
        test_x = view2[i*perp:(i+1)*perp]
        test_y = labels[i*perp:(i+1)*perp]
        if i==0:
            train_x = view1[perp:len(view1)]
            train_y = labels[perp:len(view1)]
        elif i==4:
            train_x = view1[0:4*perp]
            train_y = labels[0:4*perp]
        else:
            train_x1 = view1[0:i*perp]
            train_y1 = labels[0:i*perp]
            train_x2 = view1[(i+1)*perp:len(view1)]
            train_y2 = labels[(i+1)*perp:len(view1)]
            train_x = numpy.concatenate((train_x1,train_x2))
            train_y = numpy.concatenate((train_y1,train_y2))
        va, ta = svm_classifier(train_x, train_y, test_x, test_y, test_x, test_y)
        acc += ta
    print acc/5
    print "view2 to view1"

    acc = 0
    for i in range(0,5):
        test_x = view1[i*perp:(i+1)*perp]
        test_y = labels[i*perp:(i+1)*perp]
        if i==0:
            train_x = view2[perp:len(view1)]
            train_y = labels[perp:len(view1)]
        elif i==4:
            train_x = view2[0:4*perp]
            train_y = labels[0:4*perp]
        else:
            train_x1 = view2[0:i*perp]
            train_y1 = labels[0:i*perp]
            train_x2 = view2[(i+1)*perp:len(view1)]
            train_y2 = labels[(i+1)*perp:len(view1)]
            train_x = numpy.concatenate((train_x1,train_x2))
            train_y = numpy.concatenate((train_y1,train_y2))
        va, ta = svm_classifier(train_x, train_y, test_x, test_y, test_x, test_y)
        acc += ta
    print acc/5

def transfer_learning_5fold(view1, view2, labels):

    perp = len(view1)/5

    print "view1 to view2"

    acc = 0
    for i in range(0,5):
        test_x = view2[i*perp:(i+1)*perp]
        test_y = labels[i*perp:(i+1)*perp]
        if i==0:
            train_x = view1[perp:len(view1)]
            train_y = labels[perp:len(view1)]
        elif i==4:
            train_x = view1[0:4*perp]
            train_y = labels[0:4*perp]
        else:
            train_x1 = view1[0:i*perp]
            train_y1 = labels[0:i*perp]
            train_x2 = view1[(i+1)*perp:len(view1)]
            train_y2 = labels[(i+1)*perp:len(view1)]
            train_x = numpy.concatenate((train_x1,train_x2))
            train_y = numpy.concatenate((train_y1,train_y2))
        va, ta = svm_classifier(train_x, train_y, test_x, test_y, test_x, test_y)
        acc += ta
    print acc/5
    print "view2 to view1"

    acc = 0
    for i in range(0,5):
        test_x = view1[i*perp:(i+1)*perp]
        test_y = labels[i*perp:(i+1)*perp]
        if i==0:
            train_x = view2[perp:len(view1)]
            train_y = labels[perp:len(view1)]
        elif i==4:
            train_x = view2[0:4*perp]
            train_y = labels[0:4*perp]
        else:
            train_x1 = view2[0:i*perp]
            train_y1 = labels[0:i*perp]
            train_x2 = view2[(i+1)*perp:len(view1)]
            train_y2 = labels[(i+1)*perp:len(view1)]
            train_x = numpy.concatenate((train_x1,train_x2))
            train_y = numpy.concatenate((train_y1,train_y2))
        va, ta = svm_classifier(train_x, train_y, test_x, test_y, test_x, test_y)
        acc += ta
    print acc/5



def correlation(folder):

    print "validation correlation"
    x = numpy.load(folder+"valid-view1.npy")
    y = numpy.load(folder+"valid-view2.npy")
    corr = 0
    for i in range(0,len(x[0])):
        x1 = x[:,i] - (numpy.ones(len(x))*(sum(x[:,i])/len(x))) # mean centering
        x2 = y[:,i] - (numpy.ones(len(y))*(sum(y[:,i])/len(y))) # mean centering
        nr = sum(x1 * x2)/(math.sqrt(sum(x1*x1))*math.sqrt(sum(x2*x2))) # dividing by variance
        corr+=nr
    print corr

    print "test correlation"
    x = numpy.load(folder+"test-view1.npy")
    y = numpy.load(folder+"test-view2.npy")
    corr = 0
    for i in range(0,len(x[0])):
        x1 = x[:,i] - (numpy.ones(len(x))*(sum(x[:,i])/len(x)))
        x2 = y[:,i] - (numpy.ones(len(y))*(sum(y[:,i])/len(y)))
        nr = sum(x1 * x2)/(math.sqrt(sum(x1*x1))*math.sqrt(sum(x2*x2)))
        corr+=nr
    print corr


def correlation_with_views(x, y):
    x = numpy.array(x)
    y = numpy.array(y)
    corr = 0
    for i in range(0,len(x[0])):
        x1 = x[:,i] - (numpy.ones(len(x))*(sum(x[:,i])/len(x)))
        x2 = y[:,i] - (numpy.ones(len(y))*(sum(y[:,i])/len(y)))
        nr = sum(x1 * x2)/(math.sqrt(sum(x1*x1))*math.sqrt(sum(x2*x2)))
        corr+=nr
    print corr
    return corr

def read_projections(projected_views_folder, postfix):
    left_projected_view = pickle.load(open(projected_views_folder +  "left_proj_view_" + postfix + ".pkl", "r"))
    right_projected_view = pickle.load(open(projected_views_folder + "right_proj_view_" + postfix + ".pkl", "r"))
    pivot_projected_view = pickle.load(open(projected_views_folder + "pivot_proj_view_" + postfix + ".pkl", "r"))
    left_pivot_projection = pickle.load(open(projected_views_folder + "left_pivot_proj_view_" + postfix + ".pkl", "r"))
    right_pivot_projection = pickle.load(open(projected_views_folder + "right_pivot_proj_view_" + postfix + ".pkl", "r"))
    right_left_projection = pickle.load(open(projected_views_folder + "right_left_proj_view_" + postfix + ".pkl", "r"))
    labels = pickle.load(open(projected_views_folder + "labels_" + postfix + ".pkl", "r"))
    return([left_projected_view, right_projected_view, pivot_projected_view,
            left_pivot_projection, right_pivot_projection, right_left_projection, labels])

#
# job = sys.argv[1]
#
# if job=="tl":
#     transfer_learning_5fold(sys.argv[2]+"project/")
# elif job=="corr":
#     correlation(sys.argv[2]+"project/")

[train_left_projected_view, train_right_projected_view, train_pivot_projected_view,
 train_left_pivot_projection, train_right_pivot_projection, train_right_left_projection, train_labels] = read_projections(projected_views_folder, "train")

[test_left_projected_view, test_right_projected_view, test_pivot_projected_view,
 test_left_pivot_projection, test_right_pivot_projection, test_right_left_projection, test_labels] = read_projections(projected_views_folder, "test")

print("Left_only and Right_only")
correlation_with_views(test_left_projected_view, test_right_projected_view)
transfer_learning_5fold(test_left_projected_view, test_right_projected_view, test_labels)
print("Left_only and pivot_only")
correlation_with_views(test_left_projected_view, test_pivot_projected_view)
transfer_learning_5fold(test_left_projected_view, test_pivot_projected_view, test_labels)
print("Right_only and pivot_only")
correlation_with_views(test_right_projected_view, test_pivot_projected_view)
transfer_learning_5fold(test_right_projected_view, test_pivot_projected_view, test_labels)
print("Left_pivot and right_pivot")
correlation_with_views(test_left_pivot_projection, test_right_pivot_projection)
transfer_learning_5fold(test_left_pivot_projection, test_right_pivot_projection, test_labels)
print("Left_pivot and right_only")
correlation_with_views(test_left_pivot_projection, test_right_projected_view)
transfer_learning_5fold(test_left_pivot_projection, test_right_projected_view, test_labels)
print("Right_pivot and left_only")
correlation_with_views(test_right_pivot_projection, test_left_projected_view)
transfer_learning_5fold(test_right_pivot_projection, test_left_projected_view, test_labels)