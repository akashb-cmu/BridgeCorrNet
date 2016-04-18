__author__ = 'Akash'
"""
Based on the two view CorrNet found at https://github.com/apsarath/CorrNet based on the paper: Correlation Neural Nets (http://arxiv.org/abs/1504.07225)

This code extends the two view implementation to 3 view (and possibly n view) bridge correlation net (http://arxiv.org/abs/1510.03519)
"""

import time
import pickle

from optimization import *
from Initializer import *
from NNUtil import *


import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

LEFT = "left"
RIGHT = "right"
PIVOT = "pivot"

pivot_left = set([PIVOT, LEFT])
pivot_right = set([PIVOT, RIGHT])
pivot_left_right = set([PIVOT, LEFT, RIGHT])

class BridgeCorrNet(object):

    def init(self, numpy_rng, theano_rng=None, l_rate=0.01, optimization="sgd",
             tied=False, n_visible_left=None, n_visible_right=None, n_visible_pivot=None, n_hidden=None, lamda=5,
             W_left=None, W_right=None, W_pivot=None, b=None, W_left_prime=None, W_right_prime=None, W_pivot_prime=None,
             b_prime_left=None, b_prime_right=None, b_prime_pivot=None, input_left=None, input_right=None, input_pivot=None,
             hidden_activation="sigmoid", output_activation="sigmoid", loss_fn = "squarrederror",
             op_folder=None):
        self.numpy_rng = numpy_rng
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        self.optimization = optimization
        self.l_rate = l_rate

        self.optimizer = get_optimizer(self.optimization, self.l_rate)
        self.Initializer = Initializer(self.numpy_rng)

        self.n_visible_left = n_visible_left
        self.n_visible_right = n_visible_right
        self.n_visible_pivot = n_visible_pivot
        self.n_hidden = n_hidden
        self.lamda = lamda
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss_fn = loss_fn
        self.tied = tied
        self.op_folder = op_folder

        self.W_left = self.Initializer.fan_based_sigmoid("W_left", W_left, n_visible_left, n_hidden)
        self.optimizer.register_variable("W_left",n_visible_left,n_hidden)

        self.W_right = self.Initializer.fan_based_sigmoid("W_right", W_right, n_visible_right, n_hidden)
        self.optimizer.register_variable("W_right",n_visible_right,n_hidden)

        self.W_pivot = self.Initializer.fan_based_sigmoid("W_pivot", W_pivot, n_visible_pivot, n_hidden)
        self.optimizer.register_variable("W_pivot", n_visible_pivot, n_hidden)

        if not tied:
            self.W_left_prime = self.Initializer.fan_based_sigmoid("W_left_prime", W_left_prime, n_hidden, n_visible_left)
            self.optimizer.register_variable("W_left_prime",n_hidden, n_visible_left)

            self.W_right_prime = self.Initializer.fan_based_sigmoid("W_right_prime", W_right_prime, n_hidden, n_visible_right)
            self.optimizer.register_variable("W_right_prime",n_hidden, n_visible_right)

            self.W_pivot_prime = self.Initializer.fan_based_sigmoid("W_pivot_prime", W_pivot_prime, n_hidden,
                                                                    n_visible_pivot)
            self.optimizer.register_variable("W_pivot_prime", n_hidden, n_visible_pivot)
        else:
            self.W_left_prime = self.W_left.T
            self.W_right_prime = self.W_right.T
            self.W_pivot_prime = self.W_pivot.T

        self.b = self.Initializer.zero_vector("b", b, n_hidden)
        self.optimizer.register_variable("b",1,n_hidden)

        self.b_prime_left = self.Initializer.zero_vector("b_prime_left", b_prime_left, n_visible_left)
        self.optimizer.register_variable("b_prime_left",1,n_visible_left)

        self.b_prime_right = self.Initializer.zero_vector("b_prime_right", b_prime_right, n_visible_right)
        self.optimizer.register_variable("b_prime_right",1,n_visible_right)

        self.b_prime_pivot = self.Initializer.zero_vector("b_prime_pivot", b_prime_pivot, n_visible_pivot)
        self.optimizer.register_variable("b_prime_pivot", 1, n_visible_pivot)


        """
        Probably want to do something when a modality is missing
        """
        if input_left is None:
            self.x_left = T.matrix(name='x_left')
        else:
            self.x_left = input_left

        if input_right is None:
            self.x_right = T.matrix(name='x_right')
        else:
            self.x_right = input_right

        if input_pivot is None:
            self.x_pivot = T.matrix(name='x_pivot')
        else:
            self.x_pivot = input_pivot

        if not tied:
            self.params = [self.W_left, self.W_right, self.W_pivot, self.b, self.b_prime_left, self.b_prime_right, self.b_prime_pivot, self.W_left_prime, self.W_right_prime, self.W_pivot_prime]
            self.param_names = ["W_left", "W_right", "W_pivot", "b", "b_prime_left", "b_prime_right", "b_prime_pivot", "W_left_prime", "W_right_prime", "W_pivot_prime"]
        else:
            self.params = [self.W_left, self.W_right, self.W_pivot, self.b, self.b_prime_left, self.b_prime_right, self.b_prime_pivot]
            self.param_names = ["W_left", "W_right", "W_pivot", "b", "b_prime_left", "b_prime_right", "b_prime_pivot"]

        self.proj_from_left = theano.function([self.x_left], self.project_from_sources([LEFT]))
        self.proj_from_right = theano.function([self.x_right], self.project_from_sources([RIGHT]))
        self.proj_from_pivot = theano.function([self.x_pivot], self.project_from_sources([PIVOT]))

        self.proj_from_left_pivot = theano.function([self.x_left, self.x_pivot], self.project_from_sources([LEFT, PIVOT]))
        self.proj_from_right_pivot = theano.function([self.x_right, self.x_pivot], self.project_from_sources([RIGHT, PIVOT]))
        self.proj_from_left_right = theano.function([self.x_right, self.x_left], self.project_from_sources([RIGHT, LEFT]))
        self.proj_from_all = theano.function([self.x_right, self.x_left, self.x_pivot], self.project_from_sources([RIGHT, LEFT, PIVOT]))

        self.recon_from_left = theano.function([self.x_left], self.reconstruct_from_sources([LEFT]))
        self.recon_from_right = theano.function([self.x_right], self.reconstruct_from_sources([RIGHT]))
        self.recon_from_pivot = theano.function([self.x_pivot], self.reconstruct_from_sources([PIVOT]))

        self.recon_from_left_pivot = theano.function([self.x_left, self.x_pivot], self.reconstruct_from_sources([LEFT, PIVOT]))
        self.recon_from_right_pivot = theano.function([self.x_right, self.x_pivot], self.reconstruct_from_sources([RIGHT, PIVOT]))
        self.recon_from_right_left = theano.function([self.x_right, self.x_left], self.reconstruct_from_sources([RIGHT, LEFT]))
        self.recon_from_all = theano.function([self.x_right, self.x_left, self.x_pivot], self.reconstruct_from_sources([RIGHT, LEFT, PIVOT]))


        self.save_params()

    def get_corr_loss(self, y1, y2):
        y1_mean = T.mean(y1, axis=0)  # Find mean for each dimension (columns) over all samples (rows)
        y1_centered = y1 - y1_mean
        y2_mean = T.mean(y2, axis=0)
        y2_centered = y2 - y2_mean
        corr_nr = T.sum(y1_centered * y2_centered, axis=0)
        # y1_centered * y2_centered finds the dimension-wise (column-wise) product of the mean centered value VECTORS
        # For a matrix, * does element wise ops
        # T.sum with axis=0 sums of the dimension-wise products over all samples
        corr_dr1 = T.sqrt(T.sum(y1_centered * y1_centered, axis=0) + 1e-8)
        corr_dr2 = T.sqrt(T.sum(y2_centered * y2_centered, axis=0) + 1e-8)
        corr_dr = corr_dr1 * corr_dr2
        corr = corr_nr / corr_dr  # dimension(column)-wise division
        L = T.sum(corr) * self.lamda  # adds each dimension's correlation together to get a scalar
        return(L)

    def train_common(self,mtype="1111"):
        [z1_right, z1_left, z1_pivot] = self.reconstruct_from_sources([LEFT])
        L_left_only = loss(z1_left, self.x_left, self.loss_fn) + loss(z1_right, self.x_right, self.loss_fn) + loss(z1_pivot, self.x_pivot, self.loss_fn)

        [z2_right, z2_left, z2_pivot] = self.reconstruct_from_sources([RIGHT])
        L_right_only = loss(z2_left, self.x_left, self.loss_fn) + loss(z2_right, self.x_right, self.loss_fn) + loss(z2_pivot, self.x_pivot, self.loss_fn)

        [z_right, z_left, z_pivot] = self.reconstruct_from_sources([PIVOT])
        L_pivot_only = loss(z_left, self.x_left, self.loss_fn) + loss(z_right, self.x_right, self.loss_fn) + loss(z_pivot,
                                                                                                          self.x_pivot,
                                                                                                          self.loss_fn)

        [z3_right, z3_left, z3_pivot] = self.reconstruct_from_sources([RIGHT, LEFT, PIVOT])
        L_all_recon = loss(z3_left, self.x_left, self.loss_fn) + loss(z3_right, self.x_right, self.loss_fn) + loss(z3_pivot, self.x_pivot, self.loss_fn)

        [z6_right, z6_left, z6_pivot] = self.reconstruct_from_sources([RIGHT, LEFT])
        L_right_left = loss(z6_left, self.x_left, self.loss_fn) + loss(z6_right, self.x_right, self.loss_fn) + loss(z6_pivot, self.x_pivot, self.loss_fn)

        [z7_right, z7_left, z7_pivot] = self.reconstruct_from_sources([RIGHT, PIVOT])
        L_right_pivot = loss(z7_left, self.x_left, self.loss_fn) + loss(z7_right, self.x_right, self.loss_fn) + loss(z7_pivot,
                                                                                                          self.x_pivot,
                                                                                                          self.loss_fn)

        [z8_right, z8_left, z8_pivot] = self.reconstruct_from_sources([LEFT, PIVOT])
        L_left_pivot = loss(z8_left, self.x_left, self.loss_fn) + loss(z8_right, self.x_right, self.loss_fn) + loss(z8_pivot,
                                                                                                          self.x_pivot,
                                                                                                          self.loss_fn)
        #L3 = L_all_recon + L_pivot_only + L_right_left + L_right_pivot + L_left_pivot
        L3 = L_pivot_only + L_all_recon + L_right_left + L_right_pivot + L_left_pivot

        y1 = self.project_from_sources([LEFT])
        y2 = self.project_from_sources([RIGHT])
        y3 = self.project_from_sources([PIVOT])

        L_corr_right_left = self.get_corr_loss(y1, y2)
        L_corr_left_pivot = self.get_corr_loss(y1, y3)
        L_corr_right_pivot = self.get_corr_loss(y2, y3)

        L4 = L_corr_right_left + L_corr_left_pivot + L_corr_right_pivot

        L5 = loss(z1_pivot, self.x_pivot, self.loss_fn) + loss(z1_right, self.x_right, self.loss_fn) + \
             loss(z2_pivot, self.x_pivot, self.loss_fn) + loss(z2_left, self.x_left, self.loss_fn) + \
             loss(z_left, self.x_left, self.loss_fn) + loss(z_right, self.x_right, self.loss_fn)

        if mtype=="1111":
            print "1111"
            L = L_left_only + L_right_only + L3 - L4 # L5 not needed since it is already covered in L_left_ony, L_right_only
            # and L_pivot_only
        elif mtype=="1110":
            print "1110"
            L = L_left_only + L_right_only + L3
        elif mtype=="1101":
            print "1101"
            L = L_left_only + L_right_only - L4
        elif mtype == "0011":
            print "0011"
            L = L3 - L4
        elif mtype == "1100":
            print "1100"
            L = L_left_only + L_right_only
        elif mtype == "0010":
            print "0010"
            L = L3
        elif mtype == "euc":
            print "euc"
            L = L5
        elif mtype == "euc-cor":
            print "euc-cor"
            L = L5 - L4
        elif mtype == "all":
            L = L_left_only

        cost = T.mean(L)

        gradients = T.grad(cost, self.params)
        updates = []
        for p,g,n in zip(self.params, gradients, self.param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)

        return cost, updates

    def train_left_pivot(self, mtype="1111"):
        [z1_right, z1_left, z1_pivot] = self.reconstruct_from_sources([LEFT])
        L_left = loss(z1_left, self.x_left, self.loss_fn) + loss(z1_pivot, self.x_pivot, self.loss_fn)

        [z_right, z_left, z_pivot] = self.reconstruct_from_sources([PIVOT])
        L_pivot = loss(z_left, self.x_left, self.loss_fn) + loss(z_pivot, self.x_pivot, self.loss_fn)

        [z8_right, z8_left, z8_pivot] = self.reconstruct_from_sources([LEFT, PIVOT])
        L_left_pivot = loss(z8_left, self.x_left, self.loss_fn) + loss(z8_pivot, self.x_pivot, self.loss_fn)

        y1 = self.project_from_sources([LEFT])
        y3 = self.project_from_sources([PIVOT])

        L_corr_left_pivot = self.get_corr_loss(y1, y3)

        L4 = L_corr_left_pivot

        L5 = loss(z1_pivot, self.x_pivot, self.loss_fn) + loss(z_left, self.x_left, self.loss_fn)

        if mtype == "1111":
            print "1111"
            L = L_left + L_pivot + L_left_pivot - L4
        elif mtype == "1110":
            print "1110"
            L = L_left + L_pivot + L_left_pivot
        elif mtype == "1101":
            print "1101"
            L = L_left + L_pivot - L4
        elif mtype == "0011":
            print "0011"
            L = L_left_pivot - L4
        elif mtype == "1100":
            print "1100"
            L = L_left + L_pivot
        elif mtype == "0010":
            print "0010"
            L = L_left_pivot
        elif mtype == "euc":
            print "euc"
            L = L5
        elif mtype == "euc-cor":
            print "euc-cor"
            L = L5 - L4
        elif mtype == "all":
            L = L_left

        cost = T.mean(L)

        if self.tied:
            curr_params = [self.W_left, self.W_pivot, self.b, self.b_prime_left,  self.b_prime_pivot]
            curr_param_names = ["W_left", "W_pivot", "b", "b_prime_left", "b_prime_pivot"]
        else:
            curr_params = [self.W_left, self.W_pivot, self.b, self.b_prime_left, self.b_prime_pivot, self.W_left_prime, self.W_pivot_prime]
            curr_param_names = ["W_left", "W_pivot", "b", "b_prime_left", "b_prime_pivot" "W_left_prime", "W_pivot_prime"]

        gradients = T.grad(cost, curr_params)
        updates = []
        for p, g, n in zip(curr_params, gradients, curr_param_names):
            gr, upd = self.optimizer.get_grad_update(n, g) # upd is empty list
            updates.append((p, p + gr))
            updates.extend(upd)

        return cost, updates


    def train_right_pivot(self, mtype="1111"):
        [z1_right, z1_left, z1_pivot] = self.reconstruct_from_sources([RIGHT])
        L_right = loss(z1_right, self.x_right, self.loss_fn) + loss(z1_pivot, self.x_pivot, self.loss_fn)

        [z_right, z_left, z_pivot] = self.reconstruct_from_sources([PIVOT])
        L_pivot = loss(z_right, self.x_right, self.loss_fn) + loss(z_pivot, self.x_pivot, self.loss_fn)

        [z8_right, z8_left, z8_pivot] = self.reconstruct_from_sources([RIGHT, PIVOT])
        L_right_pivot = loss(z8_right, self.x_right, self.loss_fn) + loss(z8_pivot, self.x_pivot, self.loss_fn)

        y1 = self.project_from_sources([RIGHT])
        y3 = self.project_from_sources([PIVOT])

        L_corr_right_pivot = self.get_corr_loss(y1, y3)

        L4 = L_corr_right_pivot

        L5 = loss(z1_pivot, self.x_pivot, self.loss_fn) + loss(z_right, self.x_right, self.loss_fn)

        if mtype == "1111":
            print "1111"
            L = L_right + L_pivot + L_right_pivot - L4
        elif mtype == "1110":
            print "1110"
            L = L_right + L_pivot + L_right_pivot
        elif mtype == "1101":
            print "1101"
            L = L_right + L_pivot - L4
        elif mtype == "0011":
            print "0011"
            L = L_right_pivot - L4
        elif mtype == "1100":
            print "1100"
            L = L_right + L_pivot
        elif mtype == "0010":
            print "0010"
            L = L_right_pivot
        elif mtype == "euc":
            print "euc"
            L = L5
        elif mtype == "euc-cor":
            print "euc-cor"
            L = L5 - L4
        elif mtype == "all":
            L = L_right

        cost = T.mean(L)

        if self.tied:
            curr_params = [self.W_right, self.W_pivot, self.b, self.b_prime_right, self.b_prime_pivot]
            curr_param_names = ["W_right", "W_pivot", "b", "b_prime_right", "b_prime_pivot"]
        else:
            curr_params = [self.W_right, self.W_pivot, self.b, self.b_prime_right, self.b_prime_pivot, self.W_right_prime,
                           self.W_pivot_prime]
            curr_param_names = ["W_right", "W_pivot", "b", "b_prime_right", "b_prime_pivot" "W_right_prime", "W_pivot_prime"]

        gradients = T.grad(cost, curr_params)
        updates = []
        for p, g, n in zip(curr_params, gradients, curr_param_names):
            gr, upd = self.optimizer.get_grad_update(n, g) # Just clips the gradient and multiplies the learning rate to
            #  the gradients. upd is empty list here
            updates.append((p, p + gr))
            updates.extend(upd)

        return cost, updates

    def train_left(self):
        [z_right, z_left, z_pivot] = self.reconstruct_from_sources([LEFT])
        L = loss(z_left, self.x_left, self.loss_fn)
        cost = T.mean(L)

        if self.tied:
            curr_params = [self.W_left, self.b, self.b_prime_left]
            curr_param_names = ["W_left", "b", "b_prime_left"]
        else:
            curr_params = [self.W_left, self.b, self.b_prime_left, self.W_left_prime]
            curr_param_names = ["W_left", "b", "b_prime_left", "W_left_prime"]

        gradients = T.grad(cost, curr_params)
        updates = []
        for p,g,n in zip(curr_params, gradients, curr_param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)
        return cost, updates


    def train_right(self):
        [z_right, z_left, z_pivot] = self.reconstruct_from_sources([RIGHT])
        L = loss(z_right, self.x_right, self.loss_fn)
        cost = T.mean(L)

        if self.tied:
            curr_params = [self.W_right, self.b, self.b_prime_right]
            curr_param_names = ["W_right", "b", "b_prime_right"]
        else:
            curr_params = [self.W_right, self.b, self.b_prime_right, self.W_right_prime]
            curr_param_names = ["W_right", "b", "b_prime_right", "W_right_prime"]

        gradients = T.grad(cost, curr_params)
        updates = []
        for p,g,n in zip(curr_params, gradients, curr_param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)
        return cost, updates

    def train_pivot(self):
        [z_right, z_left, z_pivot] = self.reconstruct_from_sources([PIVOT])
        L = loss(z_pivot, self.x_pivot, self.loss_fn)
        cost = T.mean(L)

        if self.tied:
            curr_params = [self.W_pivot, self.b, self.b_prime_pivot]
            curr_param_names = ["W_pivot", "b", "b_prime_pivot"]
        else:
            curr_params = [self.W_pivot, self.b, self.b_prime_pivot, self.W_pivot_prime]
            curr_param_names = ["W_pivot", "b", "b_prime_pivot", "W_pivot_prime"]

        gradients = T.grad(cost, curr_params)
        updates = []
        for p, g, n in zip(curr_params, gradients, curr_param_names):
            gr, upd = self.optimizer.get_grad_update(n, g)
            updates.append((p, p + gr))
            updates.extend(upd)
        return cost, updates

    def project_from_sources(self, source_list):
        y_pre = self.b
        for source in source_list:
            if source == LEFT:
                y_pre += T.dot(self.x_left, self.W_left)
            elif source == RIGHT:
                y_pre += T.dot(self.x_right, self.W_right)
            elif source == PIVOT:
                y_pre += T.dot(self.x_pivot, self.W_pivot)
        y = activation(y_pre, self.hidden_activation)
        return y

    def reconstruct_from_sources(self, source_list):
        y = self.project_from_sources(source_list)
        recon_left = activation(T.dot(y, self.W_left_prime) + self.b_prime_left, self.output_activation)
        recon_right = activation(T.dot(y, self.W_right_prime) + self.b_prime_right, self.output_activation)
        recon_pivot = activation(T.dot(y, self.W_pivot_prime) + self.b_prime_pivot, self.output_activation)
        recon_list = [recon_right, recon_left, recon_pivot]
        return recon_list

    def get_lr_rate(self):
        return self.optimizer.get_l_rate()

    def set_lr_rate(self,new_lr):
        self.optimizer.set_l_rate(new_lr)

    def save_matrices(self):
        for p,nm in zip(self.params, self.param_names):
            numpy.save(self.op_folder+nm, p.get_value(borrow=True))

    def save_params(self):
        params = {}
        params["optimization"] = self.optimization
        params["l_rate"] = self.l_rate
        params["n_visible_left"] = self.n_visible_left
        params["n_visible_right"] = self.n_visible_right
        params["n_visible_pivot"] = self.n_visible_pivot
        params["n_hidden"] = self.n_hidden
        params["lamda"] = self.lamda
        params["hidden_activation"] = self.hidden_activation
        params["output_activation"] = self.output_activation
        params["loss_fn"] = self.loss_fn
        params["tied"] = self.tied
        params["numpy_rng"] = self.numpy_rng
        params["theano_rng"] = self.theano_rng
        pickle.dump(params, open(self.op_folder+"params.pck", "wb"), -1)


    def load(self, folder, input_left=None, input_right=None, input_pivot=None):

        plist = pickle.load(open(folder+"params.pck","rb"))

        self.init(plist["numpy_rng"], theano_rng=plist["theano_rng"], l_rate=plist["l_rate"],
                  optimization=plist["optimization"], tied=plist["tied"],
                  n_visible_left=plist["n_visible_left"], n_visible_right=plist["n_visible_right"], n_visible_pivot=plist["n_visible_pivot"],
                  n_hidden=plist["n_hidden"], lamda=plist["lamda"], W_left=folder+"W_left",
                  W_right=folder+"W_right", W_pivot=folder+"W_pivot", b=folder+"b", W_left_prime=folder+"W_left_prime",
                  W_right_prime=folder+"W_right_prime", W_pivot_prime=folder+"W_pivot_prime", b_prime_left=folder+"b_prime_left",
                  b_prime_right=folder+"b_prime_right", b_prime_pivot=folder+"b_prime_pivot", input_left=input_left, input_right=input_right,
                  input_pivot=input_pivot, hidden_activation=plist["hidden_activation"], output_activation=plist["output_activation"],
                  loss_fn = plist["loss_fn"], op_folder=folder)



def trainBridgeCorrNet(src_folder, tgt_folder, batch_size = 20, training_epochs=40,
                 l_rate=0.01, optimization="sgd", tied=False, n_visible_left=None,
                 n_visible_right=None, n_visible_pivot=None, n_hidden=None, lamda=5,
                 W_left=None, W_right=None, W_pivot=None, b=None, W_left_prime=None, W_right_prime=None, W_pivot_prime=None,
                 b_prime_left=None, b_prime_right=None, b_prime_pivot=None, hidden_activation="sigmoid",
                 output_activation="sigmoid", loss_fn = "squarrederror"):

    index = T.lscalar()
    x_left = T.matrix('x_left')
    x_right = T.matrix('x_right')
    x_pivot = T.matrix('x_pivot')
    # The above three theano tensors indicate the input to the Bridge Corr Net

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    n_visible_pivot = n_visible_left # Pivot = left view

    model = BridgeCorrNet()
    model.init(numpy_rng=rng, theano_rng=theano_rng, l_rate=l_rate, optimization=optimization, tied=tied,
               n_visible_left=n_visible_left, n_visible_right=n_visible_right, n_visible_pivot=n_visible_pivot,
               n_hidden=n_hidden, lamda=lamda, W_left=W_left, W_right=W_right, W_pivot=W_pivot, b=b,
               W_left_prime=W_left_prime, W_right_prime=W_right_prime, W_pivot_prime=W_pivot_prime,
               b_prime_left=b_prime_left, b_prime_right=b_prime_right, b_prime_pivot=b_prime_pivot,
               input_left=x_left, input_right=x_right, input_pivot=x_pivot, hidden_activation=hidden_activation,
               output_activation=output_activation, loss_fn =loss_fn, op_folder=tgt_folder)
    #model.load(tgt_folder,x_left,x_right)
    start_time = time.clock()
    train_set_x_left = theano.shared(numpy.asarray(numpy.zeros((1000,n_visible_left)), dtype=theano.config.floatX), borrow=True)
    train_set_x_right = theano.shared(numpy.asarray(numpy.zeros((1000,n_visible_right)), dtype=theano.config.floatX), borrow=True)
    train_set_x_pivot = theano.shared(numpy.asarray(numpy.zeros((1000, n_visible_pivot)), dtype=theano.config.floatX),
                                      borrow=True)
    # Note: The above theano shared variable assignments are just place holders and their actual values will be populated
    # before actually calling the train method

    # common_cost, common_updates = model.train_common("1111")
    common_cost, common_updates = model.train_right_pivot("1111")
    mtrain_common = theano.function([index], common_cost,updates=common_updates,
                                    givens=[(x_right, train_set_x_right[index * batch_size:(index + 1) * batch_size]),
                                            (x_pivot, train_set_x_pivot[index * batch_size:(index + 1) * batch_size])])

    right_cost, right_updates = model.train_right()
    mtrain_right = theano.function([index], right_cost,updates=right_updates,givens=[(x_right, train_set_x_right[index * batch_size:(index + 1) * batch_size])])

    #pivot_cost, pivot_updates = model.train_right()
    pivot_cost, pivot_updates = model.train_pivot()
    mtrain_pivot = theano.function([index], pivot_cost,updates=pivot_updates,givens=[(x_pivot, train_set_x_pivot[index * batch_size:(index + 1) * batch_size])])

    """
        Note on what givens does:
        Given usually takes a dict of { <tensor variable used in the model> : <value assigned to this tensor at run time/function call time
                                                                               as a function of other variables/constants> }
        Basically, it separates the actual tensor in the computation graph (part of the model definition) from a value to be
        assigned to this variable (definition/assignment of the input variable).

        Consequently, with the same model definition, using givens, at runtime we can substitute a node with a value that is
        computed as a function of other (shared) variables, in this case: the input corresponding to that batch.

        Why not just specify x_right/x_left/x_pivot as the input tensor to a theano compiled function and keep making calls to this
        function by providing the appropriate batch training data matrix as the argument? Memory management can be optimized
        by using a shared variable that will be loaded into the GPU initially, instead of passing a slice of data to the
        theano function every time, which would require a transfer from CPU RAM to GPU each time. However, the model was defined
        using x_right/x_left/x_pivot which are theano tensors. How do we make use of a shared variable to populate those tensors
        without passing their value as input to the theano function? Using givens we can specify a slice of a theano shared variable (dataset)
        to use for a specific function call, as a function of the batch index and the already loaded dataset living in a shared
        variable on the GPU.
    """


    diff = 0
    flag = 1
    detfile = open(tgt_folder+"details.txt","w")
    detfile.close()
    oldtc = float("inf")

    for epoch in xrange(training_epochs):

        print "in epoch ", epoch
        c = []
        ipfile = open(src_folder+"train/ip.txt","r")
        for line in ipfile:
            next = line.strip().split(",")
            if(next[0]=="xy"):
                if(next[1]=="dense"):
                    denseTheanoloader(next[2]+"_left",train_set_x_left, "float32")
                    denseTheanoloader(next[2]+"_right",train_set_x_right, "float32")
                    denseTheanoloader(next[2] + "_left", train_set_x_pivot, "float32")
                else:
                    sparseTheanoloader(next[2]+"_left",train_set_x_left,"float32",1000,n_visible_left)
                    sparseTheanoloader(next[2]+"_right",train_set_x_right, "float32", 1000, n_visible_right)
                    sparseTheanoloader(next[2] + "_left", train_set_x_pivot, "float32", 1000, n_visible_pivot)
                for batch_index in range(0,int(next[3])/batch_size):
                    c.append(mtrain_common(batch_index))
            elif(next[0]=="x"):
                if(next[1]=="dense"):
                    denseTheanoloader(next[2]+"_left",train_set_x_left,"float32")
                    denseTheanoloader(next[2] + "_left", train_set_x_pivot, "float32")
                else:
                    sparseTheanoloader(next[2]+"_left",train_set_x_left,"float32",1000,n_visible_left)
                    sparseTheanoloader(next[2] + "_left", train_set_x_pivot, "float32", 1000, n_visible_pivot)
                for batch_index in range(0,int(next[3])/batch_size):
                    c.append(mtrain_pivot(batch_index))
            elif(next[0]=="y"):
                if(next[1]=="dense"):
                    denseTheanoloader(next[2]+"_right",train_set_x_right,"float32")
                else:
                    sparseTheanoloader(next[2]+"_right",train_set_x_right,"float32",1000,n_visible_right)
                for batch_index in range(0,int(next[3])/batch_size):
                    c.append(mtrain_right(batch_index))


        if(flag==1):
            flag = 0
            diff = numpy.mean(c)
            di = diff
        else:
            di = numpy.mean(c) - diff
            diff = numpy.mean(c)

        print 'Difference between 2 epochs is ', di
        print 'Training epoch %d, cost ' % epoch, diff

        ipfile.close()

        detfile = open(tgt_folder+"details.txt","a")
        detfile.write("train\t"+str(diff)+"\n")
        detfile.close()
        # save the parameters for every 5 epochs
        if((epoch+1)%5==0):
            model.save_matrices()

    end_time = time.clock()
    training_time = (end_time - start_time)
    print ' code ran for %.2fm' % (training_time / 60.)
    model.save_matrices()

def trainBridgeCorrNet_with_mats(left_pivot_train=None, right_pivot_train=None, right_train=None, left_train=None, pivot_train=None, all_train=None, batch_size=20, training_epochs=40,
                       l_rate=0.01, optimization="sgd", tied=False,  n_hidden=None, n_visible_left=None, n_visible_right=None,
                       n_visible_pivot=None, lamda=5,
                       W_left=None, W_right=None, W_pivot=None, b=None, W_left_prime=None, W_right_prime=None,
                       W_pivot_prime=None, b_prime_left=None, b_prime_right=None, b_prime_pivot=None, hidden_activation="sigmoid",
                       output_activation="sigmoid", loss_fn="squarrederror", tgt_folder="./"):
    """
    :param left_pivot_train: Dictionary {LEFT: left view train instances, PIVOT: pivot view train instances}
    :param right_pivot_train: Dictionary {RIGHT: right view train instances, PIVOT: pivot view train instances}
    :param right_train: Dictionary {RIGHT: right view train instances}
    :param left_train: Dictionary {LEFT: left view train instances}
    :param pivot_train: Dictionary {PIVOT: pivot view train instances}
    :param all_train: Dictionary {LEFT: left view train instances, PIVOT: pivot view train instances, RIGHT: right view train instances}
    """
    index = T.lscalar()
    x_left = T.matrix('x_left')
    x_right = T.matrix('x_right')
    x_pivot = T.matrix('x_pivot')
    # The above three theano tensors indicate the input to the Bridge Corr Net

    if left_pivot_train is not None or all_train is not None:
        assert n_visible_left is not None, "n_visible_left not specified"
        if left_pivot_train is not None:
            assert n_visible_left == left_pivot_train[LEFT].shape[1], "n_visible_left and left_pivot mat dimension don't match"
        if all_train is not None:
            assert n_visible_left == all_train[LEFT].shape[1], "n_visible_left and all_train mat dimension don't match"
        if left_train is not None:
            assert n_visible_left == left_train.shape[1], "n_visible_left and left_train mat dimension don't match"

    if right_pivot_train is not None or all_train is not None:
        assert n_visible_right is not None, "n_visible_right not specified"
        if right_pivot_train is not None:
            assert n_visible_right == right_pivot_train[RIGHT].shape[1], "n_visible_right and right_pivot mat dimension don't match"
        if all_train is not None:
            assert n_visible_right == all_train[RIGHT].shape[1], "n_visible_right and all_train mat dimension don't match"
        if right_train is not None:
            assert n_visible_right == right_train.shape[1], "n_visible_right and right_train mat dimension don't match"

    assert n_visible_pivot is not None, "n_visible_pivot MUST be specified"
    if right_pivot_train is not None:
        assert n_visible_pivot == right_pivot_train[PIVOT].shape[1], "n_visible_pivot and right_pivot mat dimension don't match"
    if left_pivot_train is not None:
        assert n_visible_pivot == left_pivot_train[PIVOT].shape[1], "n_visible_pivot and left_pivot don't match"
    if all_train is not None:
        assert n_visible_pivot == all_train[PIVOT].shape[1], "n_visible_pivot and all_train mat dimension don't match"
    if pivot_train is not None:
        assert n_visible_pivot == pivot_train.shape[1], "n_visible_pivot and pivot_train mat dimension don't match"


    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    model = BridgeCorrNet()
    model.init(numpy_rng=rng, theano_rng=theano_rng, l_rate=l_rate, optimization=optimization, tied=tied,
               n_visible_left=n_visible_left, n_visible_right=n_visible_right, n_visible_pivot=n_visible_pivot,
               n_hidden=n_hidden, lamda=lamda, W_left=W_left, W_right=W_right, W_pivot=W_pivot, b=b,
               W_left_prime=W_left_prime, W_right_prime=W_right_prime, W_pivot_prime=W_pivot_prime,
               b_prime_left=b_prime_left, b_prime_right=b_prime_right, b_prime_pivot=b_prime_pivot,
               input_left=x_left, input_right=x_right, input_pivot=x_pivot, hidden_activation=hidden_activation,
               output_activation=output_activation, loss_fn=loss_fn, op_folder=tgt_folder)
    # model.load(tgt_folder,x_left,x_right)
    start_time = time.clock()
    train_set_x_left = theano.shared(numpy.asarray(numpy.zeros((1000, n_visible_left)), dtype=theano.config.floatX),
                                     borrow=True)
    train_set_x_right = theano.shared(numpy.asarray(numpy.zeros((1000, n_visible_right)), dtype=theano.config.floatX),
                                      borrow=True)
    train_set_x_pivot = theano.shared(numpy.asarray(numpy.zeros((1000, n_visible_pivot)), dtype=theano.config.floatX),
                                      borrow=True)
    # Dummy variables for input data. Borrow=true flag ensures the values can be updated later.

    views = set()

    # Note: The above theano shared variable assignments are just place holders and their actual values will be populated
    # before actually calling the train method. This is also why borrow=True so that by changing the value of the nump mat
    # used to initialize the shared variable, the shared variables value changes.

    # common_cost, common_updates = model.train_common("1111")
    common_cost, common_updates = model.train_common("1111")
    mtrain_common = theano.function([index], common_cost, updates=common_updates,
                                    givens=[(x_right, train_set_x_right[index * batch_size:(index + 1) * batch_size]),
                                            (x_left, train_set_x_left[index * batch_size:(index + 1) * batch_size]),
                                            (x_pivot, train_set_x_pivot[index * batch_size:(index + 1) * batch_size])])

    right_pivot_cost, right_pivot_updates = model.train_right_pivot()
    mtrain_right_pivot = theano.function([index], right_pivot_cost, updates=right_pivot_updates,
                                   givens=[(x_right, train_set_x_right[index * batch_size:(index + 1) * batch_size]),
                                           (x_pivot, train_set_x_pivot[index * batch_size:(index + 1) * batch_size])])

    # left_pivot_cost, left_pivot_updates = model.train_right()
    left_pivot_cost, left_pivot_updates = model.train_left_pivot()
    mtrain_left_pivot = theano.function([index], left_pivot_cost, updates=left_pivot_updates,
                                   givens=[(x_left, train_set_x_left[index * batch_size:(index + 1) * batch_size]),
                                       (x_pivot, train_set_x_pivot[index * batch_size:(index + 1) * batch_size])])

    left_cost, left_updates = model.train_left()
    mtrain_left = theano.function([index], left_cost, updates=left_updates,
                                  givens=[(x_left, train_set_x_left[index * batch_size:(index + 1) * batch_size])])

    right_cost, right_updates = model.train_right()
    mtrain_right = theano.function([index], right_cost, updates=right_updates,
                                   givens=[(x_right, train_set_x_right[index * batch_size:(index + 1) * batch_size])])

    pivot_cost, pivot_updates = model.train_pivot()
    mtrain_pivot = theano.function([index], pivot_cost, updates=pivot_updates,
                                   givens=[(x_pivot, train_set_x_pivot[index * batch_size:(index + 1) * batch_size])])

    """
        Note on what givens does:
        Given usually takes a dict of { <tensor variable used in the model> : <value assigned to this tensor at run time/function call time
                                                                               as a function of other variables/constants> }
        Basically, it separates the actual tensor in the computation graph (part of the model definition) from a value to be
        assigned to this variable (definition/assignment of the input variable).

        Consequently, with the same model definition, using givens, at runtime we can substitute a node with a value that is
        computed as a function of other (shared) variables, in this case: the input corresponding to that batch.

        Why not just specify x_right/x_left/x_pivot as the input tensor to a theano compiled function and keep making calls to this
        function by providing the appropriate batch training data matrix as the argument? Memory management can be optimized
        by using a shared variable that will be loaded into the GPU initially, instead of passing a slice of data to the
        theano function every time, which would require a transfer from CPU RAM to GPU each time. However, the model was defined
        using x_right/x_left/x_pivot which are theano tensors. How do we make use of a shared variable to populate those tensors
        without passing their value as input to the theano function? Using givens we can specify a slice of a theano shared variable (dataset)
        to use for a specific function call, as a function of the batch index and the already loaded dataset living in a shared
        variable on the GPU.
    """

    diff = 0
    flag = 1
    detfile = open(tgt_folder + "details.txt", "w")
    detfile.close()
    oldtc = float("inf")

    for epoch in xrange(training_epochs):

        print "in epoch ", epoch
        c = []

        if left_pivot_train is not None:
            assert (left_pivot_train[LEFT].shape[0] == left_pivot_train[PIVOT].shape[0]), "No. of samples in left_pivot_train don't match!"
            train_set_x_left.set_value(left_pivot_train[LEFT], borrow=True)
            train_set_x_pivot.set_value(left_pivot_train[PIVOT], borrow=True)
            data_size = left_pivot_train[LEFT].shape[0]
            for batch_index in range(0, int(data_size / batch_size)):
                c.append(mtrain_left_pivot(batch_index))
        if all_train is not None:
            assert (all_train[LEFT].shape[0] == all_train[RIGHT].shape[0] and all_train[RIGHT].shape[0] == all_train[PIVOT].shape[0]), "No. of samples in all_train don't match!"
            train_set_x_left.set_value(all_train[LEFT], borrow=True)
            train_set_x_right.set_value(all_train[RIGHT], borrow=True)
            train_set_x_pivot.set_value(all_train[PIVOT], borrow=True)
            data_size = all_train[LEFT].shape[0]
            for batch_index in range(0, int(data_size / batch_size)):
                c.append(mtrain_common(batch_index))
        if right_pivot_train is not None:
            assert (right_pivot_train[RIGHT].shape[0] == right_pivot_train[PIVOT].shape[
                0]), "No. of samples in right_pivot_train don't match!"
            train_set_x_right.set_value(right_pivot_train[RIGHT], borrow=True)
            train_set_x_pivot.set_value(right_pivot_train[PIVOT], borrow=True)
            data_size = right_pivot_train[RIGHT].shape[0]
            for batch_index in range(0, int(data_size / batch_size)):
                c.append(mtrain_right_pivot(batch_index))
        if left_train is not None:
            train_set_x_left.set_value(left_train, borrow=True)
            data_size = left_train.shape[0]
            for batch_index in range(0, int(data_size / batch_size)):
                c.append(mtrain_left(batch_index))
        if right_train is not None:
            train_set_x_right.set_value(right_train, borrow=True)
            data_size = right_train.shape[0]
            for batch_index in range(0, int(data_size / batch_size)):
                c.append(mtrain_right(batch_index))
        if pivot_train is not None:
            train_set_x_pivot.set_value(pivot_train, borrow=True)
            data_size = pivot_train.shape[0]
            for batch_index in range(0, int(data_size / batch_size)):
                c.append(mtrain_pivot(batch_index))



        if (flag == 1):
            flag = 0
            diff = numpy.mean(c)
            di = diff
        else:
            di = numpy.mean(c) - diff
            diff = numpy.mean(c)

        print 'Difference between 2 epochs is ', di
        print 'Training epoch %d, cost ' % epoch, diff

        detfile = open(tgt_folder + "details.txt", "a")
        detfile.write("train\t" + str(diff) + "\n")
        detfile.close()
        # save the parameters for every 5 epochs
        if ((epoch + 1) % 5 == 0):
            model.save_matrices()

    end_time = time.clock()
    training_time = (end_time - start_time)
    print ' code ran for %.2fm' % (training_time / 60.)
    model.save_matrices()
