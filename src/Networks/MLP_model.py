"""
The model is a multiclass perceptron.
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import math
import json
from utils_MLP_model import *


#Model Initialization
# on initialise les underlying variables (celles qui garantissent la sparsity)
w_vars, b_vars, stable_var, sparse_vars = init_MLP_vars()
limit_0, limit_1, temperature, epsilon, lambd = init_sparsity_constants()

class Model(object):
  def __init__(self, num_classes, batch_size, network_size, subset_ratio, num_features, dropout = 1, l2 = 0, l0 = 0, rho=0, stored_weights=None):
    # RMQ : 
    # ttes les computations, ttes les valeurs calculées intermédiaires, ttes les losses calculées et métriques sont des attributs
    
    # on définit les hyperparams
    self.dropout = dropout
    self.subset_ratio = subset_ratio
    self.rho =  rho
    self.l2 = l2

    # on initialise les weights du modèle à partir des underlying variables (qui sont en fait les vraies weights)
    # (*)
    weights, biases, stab_weight, sparse_weights = init_MLP_weights(w_vars, b_vars, stable_var, sparse_vars)
    if stored_weights is not None:
      weights, biases, stab_weight, sparse_weights = reset_stored_weights(stored_weights)

    # on définit les noms associés à chaque module
    layer_sizes = [num_features] + network_size + [num_classes]
    layer_names = ["x_input"] + [str("h") + str(l) for l in range(len(network_size)+1)]
    mask_names = [w_vars[l] + str("_masked") for l in range(len(w_vars))]
    norm_names = [w_vars[l] + str("_norm") for l in range(len(w_vars))]

    # on définit les inputs du modèle
    self.x_input = tf.placeholder(tf.float32, shape = [None, num_features])
    self.y_input = tf.placeholder(tf.int64, shape = [None])
    # ajout de distillation:
    self.y_input_distil = tf.placeholder(tf.float32, shape = [None, num_classes])



    # DEFINE VARIABLES
    # ici, on BIND véritablement les weights au modèle en tant qu'attributs
    #Stability Variable
    setattr(self, stable_var, self._bias_variable([], stab_weight)) 

    for i in range(len(w_vars)):
      #Matrix Variables
      # le premier arg est la taille de la matrice de weights qu'on vise à reproduire avec nos sparse variables; 
      # le deuxième arg est la pré-initialisation (*) des weights obtenue via init_MLP_weights et les variables pré-initialisées
      setattr(self, w_vars[i], self._weight_variable([layer_sizes[i] , layer_sizes[i+1]], weights[i])) 
      #Vector Variables
      setattr(self, b_vars[i], self._bias_variable([layer_sizes[i+1]], biases[i])) 
      #Sparsity Variables
      setattr(self, sparse_vars[i], self._log_a_variable([layer_sizes[i] , layer_sizes[i+1]], sparse_weights[i])) 
      

    # DEFINE LAYERS
    # Complètement WTF : les opérations sont effectuées avec des méthodes de la classe Model (get_l0_norm_full,...)
    # et les résultats des calculs sont stockés en attribut
    for l in range(len(w_vars)):
      if l0 > 0:
        W_masked, W_norm = self.get_l0_norm_full(getattr(self, w_vars[l]), getattr(self, sparse_vars[l]))
        setattr(self, mask_names[l], W_masked)
        setattr(self, norm_names[l], W_norm)
        hidden_layer = tf.nn.relu(tf.matmul(getattr(self, layer_names[l]), getattr(self, mask_names[l])) + getattr(self, b_vars[l]))
        setattr(self, layer_names[l+1], hidden_layer) 
      else:
        hidden_layer = tf.nn.relu(tf.matmul(getattr(self, layer_names[l]), getattr(self, w_vars[l])) + getattr(self, b_vars[l]))
        setattr(self, layer_names[l+1], hidden_layer) 
      if l < len(network_size):
        setattr(self, layer_names[l+1], tf.nn.dropout(getattr(self,  layer_names[l+1]), self.dropout)) 


    # CALCUL DES LOSSES
    #Compute standard Cross Entropy (aka xent)
    # Encore une fois, les computations intermédiaires sont sauvées en tant qu'attributs
    self.pre_softmax = getattr(self, layer_names[len(layer_sizes)-1])
    # ici on n'a pas encore agrégé les résultats des différents exemples d'un mm batch
    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.pre_softmax)
    # juste pr pouvoir visualiser les logits
    self.logits = tf.nn.softmax(self.pre_softmax)
    # la loss elle-mm
    self.xent = tf.reduce_mean(y_xent)


    #Compute robust cross-entropy.
    # on check une autre loss possible; la robust cross-ent
    data_range = tf.range(tf.shape(self.y_input)[0])
    indices = tf.map_fn(lambda n: tf.stack([tf.cast(self.y_input[n], tf.int32), n]), data_range)
    pre_softmax_t = tf.transpose(self.pre_softmax)
    self.nom_exponent = pre_softmax_t -  tf.gather_nd(pre_softmax_t, indices)
    sum_exps = 0
    for i in range(num_classes):
      grad = tf.gradients(self.nom_exponent[i], self.x_input)
      exponent = rho*tf.reduce_sum(tf.abs(grad[0]), axis=1) + self.nom_exponent[i]
      sum_exps+=tf.exp(exponent)
    robust_y_xent = tf.log(sum_exps)
    # ici, on vient de finir de calculer cette nvelle loss => on la stocke en attribut, une fois calculée
    self.robust_xent = tf.reduce_mean(robust_y_xent)

    #Compute stable cross-entropy.
    # Encore une loss différente
    self.stable_data_loss = tf.nn.relu(y_xent - getattr(self, stable_var))
    self.stable_xent = getattr(self, stable_var) + 1/(self.subset_ratio) * tf.reduce_mean(self.stable_data_loss)

    #Compute stable robust cross-entropy.
    # Encore une loss différente
    self.rob_stable_data_loss = tf.nn.relu(robust_y_xent - getattr(self, stable_var))
    self.robust_stable_xent = getattr(self, stable_var) + 1/(self.subset_ratio) * tf.reduce_mean(self.rob_stable_data_loss)

    #Compute regularization terms
    l0_regularizer = 0
    l2_regularizer = 0

    for i in range(len(w_vars)):
      if l0 > 0:
        l0_regularizer += getattr(self, norm_names[i])
        l2_regularizer += tf.reduce_sum(tf.square(getattr(self, mask_names[i])))
      else:
        l2_regularizer += tf.reduce_sum(tf.square(getattr(self, w_vars[i])))

    self.regularizer = l2*l2_regularizer + l0*l0_regularizer


    # LOSSES WITH SELF-DISTILLATION:
    # try:
    #   # basic L2 distil loss
    #   self.distil_loss = tf.reduce_mean(tf.squared_difference(tf.nn.softmax(self.y_input_distil), self.logits))
    #   # robust xent distil loss
    #   y_xent_distil = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input_distil, logits=self.pre_softmax)
    #   data_range_ = tf.range(tf.shape(self.y_input_distil)[0])
    #   indices_ = tf.map_fn(lambda n: tf.stack([tf.cast(self.y_input_distil[n], tf.int32), n]), data_range_)
    #   pre_softmax_t_ = tf.transpose(self.pre_softmax)
    #   self.nom_exponent = pre_softmax_t_ -  tf.gather_nd(pre_softmax_t_, indices_)
    #   sum_exps_ = 0
    #   for i in range(num_classes):
    #     grad_ = tf.gradients(self.nom_exponent[i], self.x_input)
    #     exponent_ = rho*tf.reduce_sum(tf.abs(grad_[0]), axis=1) + self.nom_exponent[i]
    #     sum_exps_+=tf.exp(exponent_)
    #   distil_robust_y_xent = tf.log(sum_exps_)
    #   self.distil_robust_xent = tf.reduce_mean(distil_robust_y_xent)
    #   # stable xent distil loss
    #   self.distil_stable_data_loss = tf.nn.relu(y_xent_distil - getattr(self, stable_var))
    #   self.distil_stable_xent = getattr(self, stable_var) + 1/(self.subset_ratio) * tf.reduce_mean(self.distil_stable_data_loss)
    #   # robust stable xent distil loss
    #   self.distil_rob_stable_data_loss = tf.nn.relu(distil_robust_y_xent - getattr(self, stable_var))
    #   self.distil_robust_stable_xent = getattr(self, stable_var) + 1/(self.subset_ratio) * tf.reduce_mean(self.distil_rob_stable_data_loss)
    # except:
    #   print("No distillation")



    #Evaluation
    # on évalue pour chaque batch et on stocke en attribut du modèle
    self.y_pred = tf.argmax(self.pre_softmax, 1)
    correct_prediction = tf.equal(self.y_pred, self.y_input)
    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      


  @staticmethod
  def _weight_variable(shape, initial = None):
      if initial is None:
        W0 = tf.glorot_uniform_initializer()
        return tf.get_variable(shape=shape, initializer=W0, name=str(np.random.randint(1e10)))

      else:
        W0 = tf.constant(initial, shape = shape, dtype=tf.float32)
        return tf.Variable(W0)

  @staticmethod
  def _bias_variable(shape, initial = None):
      if initial is None:
        b0 = tf.constant(0.1, shape = shape)
        return tf.Variable(b0)
      else:
        b0 = tf.constant(initial, shape = shape, dtype=tf.float32)
        return tf.Variable(b0)

  @staticmethod
  def _log_a_variable(shape, initial = None):
      if initial is None:
        a0 = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01))
        return a0
      else:
        a0 = tf.constant(initial, shape = shape, dtype=tf.float32)
        return tf.Variable(a0)

  def get_l0_norm_full(self, x, log_a):

    shape = x.get_shape()
    # sample u
    u = tf.random_uniform(shape)
    # compute hard concrete distribution
    # i.e., implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution
    y = tf.sigmoid((tf.log(u) - tf.log(1.0 - u) + log_a) / temperature)
    # stretch hard concrete distribution
    s_bar = y * (limit_1 - limit_0) + limit_0
    l0_norm = tf.reduce_sum(tf.clip_by_value(
        tf.sigmoid(log_a - temperature * math.log(-limit_0 / limit_1)),
        epsilon, 1-epsilon))

    # get mask for calculating sparse version of tensor
    mask = hard_sigmoid(s_bar)
    # return masked version of tensor and l0 norm
    return tf.multiply(x, mask), l0_norm

def hard_sigmoid(x):
    return tf.minimum(tf.maximum(x, tf.zeros_like(x)), tf.ones_like(x))

