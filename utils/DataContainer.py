import numpy as np
import random
import sys

class data_t(object):
    def __init__(self, data, labels=None):
        self.labels = labels
        self.data = data
        self.num_examples = data.shape[0]

    def next_batch(self, batch_size, index):
        idx = index * batch_size
        n_idx = index * batch_size + batch_size
        return self.data[idx:n_idx, :], self.labels[idx:n_idx, :]

# expects a numpy array of data and a corresponding numpy array of labels
# samples on the rows, features on the columns
class DataContainer:
    def __init__(self, data, labels, train_split=0.8, test_split=0.2):
        assert(data.shape[0] == labels.shape[0])
        self.num_classes = labels.shape[1]
        self.class_counts = {}
        self.train, self.test = self.partition(data, labels, train_split, test_split)

    # Shuffle training dataset (when creating dataset)
    def shuffle_and_transform(self, data, labels):
        stacked_d = np.vstack(data)
        stacked_l = np.vstack(labels)

        samples = random.sample(range(stacked_d.shape[0]),stacked_d.shape[0])

        # convert lists to numpy arrays
        stacked_d = stacked_d[samples]
        stacked_l = stacked_l[samples]

        return data_t(stacked_d, stacked_l)

    #   Shuffle training dataset (in between epochs)
    def shuffle(self):
        idxs = np.arange(self.train.data.shape[0])
        np.random.shuffle(idxs)
        self.train.data = np.squeeze(self.train.data[idxs])
        self.train.labels = np.squeeze(self.train.labels[idxs])

    def partition(self, data, labels, train_split=0.8, test_split=0.2):
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        for i in range(self.num_classes):
            # find where the labels are equal to the certain class
            idxs = np.where(np.argmax(labels, axis=1) == i)[0]
            np.random.shuffle(idxs)

            # record the class count information
            self.class_counts[str(i)] = idxs.shape[0]

            # get the int that splits the train/test sets
            split = int(train_split * idxs.shape[0])

            # append class data to respective lists
            x_train.append(data[idxs[:split]])
            y_train.append(labels[idxs[:split]])

            x_test.append(data[idxs[split:]])
            y_test.append(labels[idxs[split:]])

        # format into datacontainer 
        train = self.shuffle_and_transform(x_train, y_train)
        test = self.shuffle_and_transform(x_test, y_test)

        return [train, test]

