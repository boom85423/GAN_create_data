import tensorflow as tf
import numpy as np
from sklearn.datasets import load_boston
import pandas as pd
import random as rd
from scipy import stats

tf.set_random_seed(123)
np.random.seed(123)

BATCH_SIZE = 32
LR_G = 0.001
LR_D = 0.001

boston = load_boston()
boston = pd.DataFrame(boston['data'], columns=boston['feature_names'])
x = boston['AGE'].values

N_IDEAS = 5
ART_COMPONENTS = 30 
PAINT_POINTS = np.vstack([np.random.choice(x, ART_COMPONENTS, replace=True) for i in range(BATCH_SIZE)]) # bootstrap : (32,30)

with tf.variable_scope('Generator'):
    G_in = tf.placeholder(tf.float32, [None,N_IDEAS])
    G_l1 = tf.layers.dense(G_in, 100, tf.nn.sigmoid)
    G_out = tf.layers.dense(G_l1, ART_COMPONENTS, tf.nn.sigmoid) # converge to (0,1)

with tf.variable_scope('Discriminator'):
    real_art = tf.placeholder(tf.float32, [None,ART_COMPONENTS])
    D_l0 = tf.layers.dense(real_art, 100, tf.nn.relu, name='l')
    prob_artist0 = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out') # power
    D_l1 = tf.layers.dense(G_out, 100, tf.nn.relu, name='l', reuse=True) # reuse th param from genuine discriminator
    prob_artist1 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True) # type I error

G_loss = tf.reduce_mean(tf.log(1-prob_artist1))
D_loss = -tf.reduce_mean(tf.log(prob_artist0) + tf.log(1-prob_artist1))
train_G = tf.train.AdamOptimizer(LR_G).minimize(G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))
train_D = tf.train.AdamOptimizer(LR_D).minimize(D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        artist_paintings = PAINT_POINTS
        G_ideas = np.random.randn(BATCH_SIZE, N_IDEAS) # (32,5)
        G_out_, prob_artist0_, prob_artist1_, D_loss_ = sess.run([G_out, prob_artist0, prob_artist1, D_loss, train_G, train_D], feed_dict={G_in:G_ideas, real_art:artist_paintings})[0:4]
        # if i%300 == 0:
        #     print(prob_artist0_.mean(), prob_artist1_.mean(), D_loss_)
    G_out_ = G_out_*100 # age from 0 to 100 
    x1 = G_out_[rd.randint(0,32), :] # random select one sample
    print("Mean of original data =", np.mean(x))
    print("Mean of created data =", np.mean(x1))

# Null hypothesis : two samples are same
res = stats.ttest_ind(x, x1)
print('P-value = {}'.format(res[1])) 
