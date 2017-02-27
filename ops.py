#! /usr/bin/env python2.7

from random import shuffle

def load_csv(path):
    data = []
    labels = []
    for line in open(path):
        line = line.split(',')
        data.append([ float(x) for x in line[:-1] ])
        labels.append(int(line[-1].rstrip()))
    print data
    print labels
    return data, labels


def get_iris():
    #load data from sklearn
    iris_data, iris_target = load_csv('data/iris.csv')
    #shuffle data
    c = list(zip(iris_data, iris_target))
    shuffle(c)
    iris_data, iris_target = zip(*c)

    #split data
    train_iris_data = iris_data[:120]
    test_iris_data = iris_data[120:]

    train_iris_labels = [ convert_to_one_hot(x, 3) for x in iris_target[:120] ]
    test_iris_labels = [ convert_to_one_hot(x, 3) for x in iris_target[120:] ]
        
    return train_iris_data, train_iris_labels, test_iris_data, test_iris_labels

#split the data
def convert_to_one_hot(val, categories):
    one_hot = [ 0 for x in range(categories) ]
    one_hot[val] = 1
    return one_hot
