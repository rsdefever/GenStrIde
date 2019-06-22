#/usr/bin/python

import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import sys, argparse
import os

from utils import tf_util

class PointNet:
    def __init__(self, lr=0.001, epochs=75, \
        batch_size=16, disp_step=1, n_points=25, n_input=3, \
        n_classes=4, dropout=0, load=0, save=0, verbose=0, \
        weights_dir='/scratch3/ctargon/weights/r2.0/r2'):

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.display_step = disp_step
        self.n_points = n_points
        self.n_input = n_input
        self.n_classes = n_classes
        self.load = load
        self.save = save
        self.dropout = dropout
        self.verbose = verbose
        self.weights_file = weights_dir + '/model'

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Model definition for pointnet
    def pointnet(self, point_cloud, is_training, bn=True, bn_decay=None):
        """ Classification PointNet, input is BxNx3, output Bxn where n is num classes """
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value

        input_image = tf.expand_dims(point_cloud, -1)

        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=bn, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=bn, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=bn, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=bn, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=bn, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point,1],
                                 padding='VALID', scope='maxpool')
        
        # MLP on global point cloud vector
        net = tf.layers.flatten(net)
        net = tf_util.fully_connected(net, 512, bn=bn, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 256, bn=bn, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                              scope='dp1')
        net = tf_util.fully_connected(net, self.n_classes, activation_fn=None, scope='fc3')

        return net


    # Get the loss from predictions vs labels
    def get_loss(self, pred, label):
        """ pred: B*NUM_CLASSES,
            label: B, """
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label)
        classify_loss = tf.reduce_mean(loss)
        tf.summary.scalar('classify loss', classify_loss)
        return classify_loss


    # Function modified from https://github.com/charlesq34/pointnet/blob/master/provider.py
    def rotate_point_cloud(self, batch_data):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            angles = np.random.uniform(size=(3)) * 2 * np.pi
            cosval = np.cos(angles)
            sinval = np.sin(angles)

            x_rot_mat = np.array([[1, 0, 0],
                                  [0, cosval[0], -sinval[0]],
                                  [0, sinval[0], cosval[0]]])

            y_rot_mat = np.array([[cosval[1], 0, sinval[1]],
                                  [0, 1, 0],
                                  [-sinval[1], 0, cosval[1]]])

            z_rot_mat = np.array([[cosval[2], -sinval[2], 0],
                                  [sinval[2], cosval[2], 0],
                                  [0, 0, 1]])

            # Overall rotation calculated from x,y,z -->
            # order matters bc matmult not commutative 
            overall_rot = np.dot(z_rot_mat,np.dot(y_rot_mat,x_rot_mat))
            # Transposes bc overall_rot operates on col. vec [[x,y,z]]
            rotated_pc = np.dot(overall_rot,batch_data[k,:,:3].T).T
            rotated_data[k] = np.concatenate((rotated_pc, batch_data[k,:,3:]), axis=1)

        return rotated_data

    # Method to train the model
    def run(self, dataset, conf_matrix=True):
        tf.reset_default_graph()

        pc_pl = tf.placeholder(tf.float32, [None, self.n_points, self.n_input])
        y_pl = tf.placeholder(tf.float32, [None, self.n_classes])
        is_training_pl = tf.placeholder(tf.bool, shape=())  

        # Construct model
        pred = self.pointnet(pc_pl, is_training_pl)

        loss = self.get_loss(pred, y_pl)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        
        saver = tf.train.Saver()

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        sess = tf.Session()
        sess.run(init)

        if self.load:
            saver.restore(sess, '/tmp/cnn')

        total_batch = int(dataset.train.num_examples/self.batch_size)

        is_training = True

        # Training cycle
        for epoch in range(self.epochs):
            avg_cost = 0.
            
            dataset.shuffle()
            
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = dataset.train.next_batch(self.batch_size, i)
                batch_x = self.rotate_point_cloud(batch_x)
                _, c = sess.run([optimizer, loss], feed_dict={pc_pl: batch_x, 
                                                              y_pl: batch_y,
                                                              is_training_pl: is_training})

                # Compute average loss
                avg_cost += c / total_batch

            if self.verbose:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

            if (epoch+1) % 10 == 0 and self.save:
                saver.save(sess, self.weights_file)

        if self.save:
            saver.save(sess, self.weights_file)

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_pl, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        accs = []
        cm_preds = []
        cm_labels = []
        is_training = False
        total_test_batch = int(dataset.test.num_examples / self.batch_size)
        for i in range(total_test_batch):
            batch_x, batch_y = dataset.test.next_batch(self.batch_size, i)
            #batch_x = self.rotate_point_cloud(batch_x)
            acc, p = sess.run([accuracy, pred], feed_dict={pc_pl: batch_x, 
                                       y_pl: batch_y,
                                       is_training_pl: is_training}) 
            accs.append(acc)
            cm_preds.append(p)
            cm_labels.append(batch_y)           

        if conf_matrix:
            cm_preds = np.vstack(cm_preds)
            cm_preds = np.argmax(cm_preds, axis=1)
            cm_labs = np.vstack(cm_labels)
            cm_labs = np.argmax(cm_labs, axis=1)
            self.confusion_matrix(cm_preds, cm_labs, sess)

        sess.close()

        return sum(accs) / float(len(accs))

    # Run inference and output accuracy with confusion matrix
    # I.e., we are testing the network here. We have labeled data
    def inference(self, dataset, conf_matrix=True):
        tf.reset_default_graph()

        pc_pl = tf.placeholder(tf.float32, [None, self.n_points, self.n_input])
        y_pl = tf.placeholder(tf.float32, [None, self.n_classes])
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Construct model
        pred = self.pointnet(pc_pl, is_training_pl)

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_pl, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Load from weights file
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, self.weights_file)

        accs = []
        cm_preds = []
        cm_labels = []
        is_training = False
        total_test_batch = int(dataset.test.num_examples / self.batch_size)
        for i in range(total_test_batch):
            batch_x, batch_y = dataset.test.next_batch(self.batch_size, i)
            acc, p = sess.run([accuracy, pred], feed_dict={pc_pl: batch_x,
                                                y_pl: batch_y,
                                                is_training_pl: is_training})
            accs.append(acc)
            cm_preds.append(p)
            cm_labels.append(batch_y)

        # confusion matrix
        if conf_matrix:
            cm_preds = np.vstack(cm_preds)
            cm_preds = np.argmax(cm_preds, axis=1)
            cm_labs = np.vstack(cm_labels)
            cm_labs = np.argmax(cm_labs, axis=1)
            self.confusion_matrix(cm_preds, cm_labs, sess)

        sess.close()

        return sum(accs) / float(len(accs))

    # Function to calculate and write confusion matrix
    def confusion_matrix(self, preds, labels, sess):
        cm = tf.confusion_matrix(labels=labels, predictions=preds, num_classes=self.n_classes)
        conf_mat = sess.run(cm)
        print(conf_mat)

    # Function to perform inference on data with no labels
    # i.e., using the PointNet to identify structure of unknown atoms
    def infer_nolabel(self, dataset):

        tf.reset_default_graph()
        pc_pl = tf.placeholder(tf.float32, [None, self.n_points, self.n_input])
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Construct model
        pred = self.pointnet(pc_pl, is_training_pl)
        pred_ndx = tf.argmax(pred,1)

        # Load from weights file
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, self.weights_file)

        results = []
        is_training = False
        total_test_batch = int(dataset.shape[0] / self.batch_size)

        for i in range(total_test_batch+1):
            batch_x = self.next_test_batch(dataset,self.batch_size,i)
            if batch_x is not None:
                results.extend(pred_ndx.eval({pc_pl: batch_x,
                                       is_training_pl: is_training},
                                       session=sess))
        sess.close()
        return results

    def next_test_batch(self,dataset, batch_size, index):
        idx = index * batch_size
        n_idx = index * batch_size + batch_size
        if n_idx < dataset.shape[0]:
            return dataset[idx:n_idx, :]
        elif idx < dataset.shape[0]:
            return dataset[idx: , :]
        else:
            return None

