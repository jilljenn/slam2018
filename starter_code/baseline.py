"""
Duolingo SLAM Shared Task - Baseline Model

This baseline model loads the training and test data that you pass in via --train and --test arguments for a particular
track (course), storing the resulting data in InstanceData objects, one for each instance. The code then creates the
features we'll use for logistic regression, storing the resulting LogisticRegressionInstance objects, then uses those to
train a regularized logistic model with SGD, and then makes predictions for the test set and dumps them to a CSV file
specified with the --pred argument, in a format appropriate to be read in and graded by the eval.py script.

We elect to use two different classes, InstanceData and LogisticRegressionInstance, to delineate the boundary between
the two purposes of this code; the first being to act as a user-friendly interface to the data, and the second being to
train and run a baseline model as an example. Competitors may feel free to use InstanceData in their own code, but
should consider replacing the LogisticRegressionInstance with a class more appropriate for the model they construct.

This code is written to be compatible with both Python 2 or 3, at the expense of dependency on the future library. This
code does not depend on any other Python libraries besides future.
"""

import argparse
from scipy.sparse import lil_matrix, save_npz
import numpy as np
from collections import defaultdict, namedtuple, Counter
from io import open
import math
import os
from random import shuffle, uniform
import pickle
import yaml
import pandas as pd
from filters import follow_spec
import sys

from future.builtins import range
from future.utils import iteritems

# Sigma is the L2 prior variance, regularizing the baseline model. Smaller sigma means more regularization.
_DEFAULT_SIGMA = 20.0

# Eta is the learning rate/step size for SGD. Larger means larger step size.
_DEFAULT_ETA = 0.1


def main():
    """
    Executes the baseline model. This loads the training data, training labels, and dev data, then trains a logistic
    regression model, then dumps predictions to the specified file.

    Modify the middle of this code, between the two commented blocks, to create your own model.
    """

    parser = argparse.ArgumentParser(description='Duolingo shared task baseline model')
    parser.add_argument('--dataset', help='Dataset name', required=True)
    parser.add_argument('--spec', help='Specifications', required=True)
    # parser.add_argument('--train', help='Training file name', required=True)
    # parser.add_argument('--test', help='Test file name, to make predictions on', required=True)
    # parser.add_argument('--pred', help='Output file name for predictions, defaults to test_name.pred')
    args = parser.parse_args()

    args_train = '../data_{:s}/{:s}.slam.20171218.train'.format(args.dataset, args.dataset)
    args_valid = '../data_{:s}/{:s}.slam.20171218.dev'.format(args.dataset, args.dataset)
    args_test = '../data_{:s}/{:s}.slam.20171218.test'.format(args.dataset, args.dataset)
    DATASET_DIR = os.path.join('data', '{:s}_{:s}'.format(args.spec, args.dataset))
    if not os.path.isdir(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    df_valid = pd.read_csv(args_valid + '.key', sep=' ', names='kv')
    valid_labels = dict(zip(df_valid['k'], df_valid['v']))

    # if not args_pred:
    args_pred = args_test + '.pred'

    assert os.path.isfile(args_train)
    assert os.path.isfile(args_valid)
    assert os.path.isfile(args_test)

    # Assert that the train course matches the test course
    print(os.path.basename(args_train), os.path.basename(args_valid), os.path.basename(args_test))
    assert os.path.basename(args_train)[:5] == os.path.basename(args_valid)[:5] == os.path.basename(args_test)[:5]

    entities = defaultdict(set)
    if not os.path.isfile('{:s}.pickle'.format(args.dataset)):
        training_data, training_labels = load_data(args_train)
        valid_data = load_data(args_valid)
        test_data = load_data(args_test)

        with open('{:s}.pickle'.format(args.dataset), 'wb') as f:
            pickle.dump({
                'training_data': training_data,
                'training_labels': training_labels,
                'valid_data': valid_data,
                'test_data': test_data
            }, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('{:s}.pickle'.format(args.dataset), 'rb') as f:
            backup = pickle.load(f)
            training_data = backup['training_data']
            training_labels = backup['training_labels']
            valid_data = backup['valid_data']
            test_data = backup['test_data']

    with open('{:s}.keys'.format(args.dataset), 'w') as f:
        for instance_data in test_data:
            f.write(instance_data.instance_id + '\n')

    for instance_data in training_data + valid_data + test_data:
        instance = instance_data.__dict__
        for key in instance:
            if key in {'time', 'days'}:
                entities[key].add(instance[key])
            elif key not in {'morphological_features', 'countries'}:
                entities[key].add('{:s}={:s}'.format(key, str(instance[key])))
            elif key == 'countries':
                entities[key].update(['countries={:s}'.format(country) for country in instance[key]])
            else:
                for subkey in instance[key]:
                    entities[subkey].add(None)
                    entities[subkey].add('{:s}={:s}'.format(subkey, str(instance[key][subkey])))
    print(list(training_labels.values())[:5])
    all_entities = set()  # defaultdict(dict)
    # interesting_keys = ['user', 'token', 'part_of_speech', 'Definite', 'Gender', 'Number', 'fPOS', 'dependency_label', 'exercise_index', 'token_index', 'countries', 'client', 'session', 'format', 'Person', 'PronType', 'Mood', 'Tense', 'VerbForm']
    interesting_keys = ['user', 'token', 'part_of_speech', 'dependency_label', 'exercise_index', 'countries', 'client', 'session', 'format']
    for key in entities:
        print(key, len(entities[key]))
        if len(entities[key]) < 100:
            print(entities[key])
        elif key in {'time', 'days'}:
            values = sorted(list(filter(None, entities[key])))
            print(values[:6], values[-10:])
            if key == 'time':
                max_time = max(values)
        if key in interesting_keys:
            all_entities.update(entities[key])
    all_entities.add('time')
    all_entities.add('days')
    encode = dict(zip(all_entities, range(len(all_entities))))

    nb = Counter()
    train = []
    Xi_train = []
    Xv_train = []
    y_train = []
    wins_train = []
    fails_train = []
    for instance_data in training_data:
        instance = instance_data.__dict__
        if not follow_spec(args.spec, instance):
            continue
        instance['countries'] = instance['countries'][0]  # Only keep first country
        try:
            ids = [encode[key + '=' + str(instance.get(key))] for key in interesting_keys]  # user_id, item_id, etc.
        except KeyError:
            print('Erreur', instance)
            raise Exception
        assert None not in ids
        # Xi_train.append(ids + [0, 0])
        # Xi_train.append(ids + [encode['time'], encode['days']])
        Xi_train.append(ids)
        user_id, item_id = ids[:2]
        this_time = instance['time'] if instance['time'] is not None else 0
        # print('this time has type', type(this_time))
        # line = [1] * len(ids) + [this_time / max_time, instance['days']]
        line = [1] * len(ids)
        Xv_train.append(line)
        is_correct = 1 - int(training_labels[instance_data.instance_id])
        y_train.append(is_correct)
        train.append(ids + [is_correct, nb[(user_id, item_id, 0)], nb[(user_id, item_id, 1)]])
        nb[(user_id, item_id, is_correct)] += 1
    os.chdir(DATASET_DIR)
    np.save('Xi_train.npy', np.array(Xi_train))
    np.save('Xv_train.npy', np.array(Xv_train))
    np.save('y_train.npy', np.array(y_train))
    pd.DataFrame(train).to_csv('train.csv', index=False, header=False)

    valid = []
    Xi_valid = []
    Xv_valid = []
    y_valid = []
    for instance_data in valid_data:
        instance = instance_data.__dict__
        if not follow_spec(args.spec, instance):
            continue
        instance['countries'] = instance['countries'][0]  # Only keep first country
        ids = [encode[key + '=' + str(instance.get(key))] for key in interesting_keys]  # user_id, item_id, etc.
        assert None not in ids
        # Xi_valid.append(ids + [0, 0])
        # Xi_valid.append(ids + [encode['time'], encode['days']])
        Xi_valid.append(ids)
        user_id, item_id = ids[:2]
        this_time = instance['time'] if instance['time'] is not None else 0
        # line = [1] * len(ids) + [this_time / max_time, instance['days']]
        line = [1] * len(ids)
        assert None not in line
        Xv_valid.append(line)
        is_correct = 1 - int(valid_labels[instance_data.instance_id])
        y_valid.append(is_correct)
        valid.append(ids + [is_correct, nb[(user_id, item_id, 0)], nb[(user_id, item_id, 1)]])
        nb[(user_id, item_id, is_correct)] += 1
    np.save('Xi_valid.npy', np.array(Xi_valid))
    np.save('Xv_valid.npy', np.array(Xv_valid))
    np.save('y_valid.npy', np.array(y_valid))
    pd.DataFrame(valid).to_csv('valid.csv', index=False, header=False)

    test = []
    Xi_test = []
    Xv_test = []
    test_keys = []
    for instance_data in test_data:
        instance = instance_data.__dict__
        if not follow_spec(args.spec, instance):
            continue
        instance['countries'] = instance['countries'][0]  # Only keep first country
        ids = [encode[key + '=' + str(instance.get(key))] for key in interesting_keys]  # user_id, item_id, etc.
        assert None not in ids
        user_id, item_id = ids[:2]
        # Xi_test.append(ids + [0, 0])
        # Xi_test.append(ids + [encode['time'], encode['days']])
        Xi_test.append(ids)
        this_time = instance['time'] if instance['time'] is not None else 0
        # line = [1] * len(ids) + [this_time / max_time, instance['days']]
        line = [1] * len(ids)
        assert None not in line
        Xv_test.append(line)
        test_keys.append(instance['instance_id'])
    np.save('Xi_test.npy', np.array(Xi_test))
    np.save('Xv_test.npy', np.array(Xv_test))
    with open('{:s}_test.keys'.format(args.dataset), 'w') as f:
        f.write('\n'.join(test_keys))
    pd.DataFrame(test).to_csv('test.csv', index=False, header=False)

    print('train', len(train), 'valid', len(valid), 'test', len(test))

    with open('config.yml', 'w') as f:
        config = {
            'NUM': {key: len(entities[key]) for key in interesting_keys},
            'NB_CLASSES': 2,
            'BATCH_SIZE': 0
        }
        f.write(yaml.dump(config, default_flow_style=False))

    ####################################################################################
    # Here is the delineation between loading the data and running the baseline model. #
    # Replace the code between this and the next comment block with your own.          #
    ####################################################################################

    training_instances = [LogisticRegressionInstance(features=instance_data.to_features(),
                                                     label=training_labels[instance_data.instance_id],
                                                     name=instance_data.instance_id
                                                     ) for instance_data in training_data]

    test_instances = [LogisticRegressionInstance(features=instance_data.to_features(),
                                             label=None,
                                             name=instance_data.instance_id
                                             ) for instance_data in test_data]

    features = set()
    for instance in training_instances:
        features.update(instance.features)
    for instance in test_instances:
        features.update(instance.features)

    # with open('features.rst', 'w') as f:
    #     f.write('\n'.join(features))
    # encode_features = dict(zip(features, range(len(features))))
    # print('oh le nombre de features', len(features))

    # X_train = lil_matrix((len(training_instances), len(features)))
    # for i, instance in enumerate(training_instances):
    #     activated_features = [encode_features[feature] for feature in instance.features]
    #     X_train[i, activated_features] = [1] * len(activated_features)
    # save_npz('X_train.npz', X_train.tocsr())

    # X_test = lil_matrix((len(test_instances), len(features)))
    # for i, instance in enumerate(test_instances):
    #     activated_features = [encode_features[feature] for feature in instance.features]
    #     X_test[i, activated_features] = [1] * len(activated_features)
    # save_npz('X_test.npz', X_test.tocsr())

    logistic_regression_model = LogisticRegression()
    # logistic_regression_model.train(training_instances, iterations=10)

    predictions = logistic_regression_model.predict_test_set(test_instances)

    ####################################################################################
    # This ends the baseline model code; now we just write predictions.                #
    ####################################################################################

    os.chdir('../..')
    with open(args_pred, 'wt') as f:
        for instance_id, prediction in iteritems(predictions):
            f.write(instance_id + ' ' + str(prediction) + '\n')


def load_data(filename):
    """
    This method loads and returns the data in filename. If the data is labelled training data, it returns labels too.

    Parameters:
        filename: the location of the training or test data you want to load.

    Returns:
        data: a list of InstanceData objects from that data type and track.
        labels (optional): if you specified training data, a dict of instance_id:label pairs.
    """

    # 'data' stores a list of 'InstanceData's as values.
    data = []

    # If this is training data, then 'labels' is a dict that contains instance_ids as keys and labels as values.
    training = False
    if filename.find('train') != -1:
        training = True

    if training:
        labels = dict()

    num_exercises = 0
    print('Loading instances...')

    with open(filename, 'rt') as f:
        for line in f:
            line = line.strip()

            # If there's nothing in the line, then we're done with the exercise. Print if needed, otherwise continue
            if len(line) == 0:
                num_exercises += 1
                if num_exercises % 100000 == 0:
                    print('Loaded ' + str(len(data)) + ' instances across ' + str(num_exercises) + ' exercises...')

            # If the line starts with #, then we're beginning a new exercise
            elif line[0] == '#':
                list_of_exercise_parameters = line[2:].split()
                instance_properties = dict()
                for exercise_parameter in list_of_exercise_parameters:
                    [key, value] = exercise_parameter.split(':')
                    if key == 'countries':
                        value = value.split('|')
                    elif key == 'days':
                        value = float(value)
                    elif key == 'time':
                        if value == 'null':
                            value = None
                        else:
                            assert '.' not in value
                            value = int(value)
                            if value < 0:
                                print('Oops, negative value in', filename)
                                value = None
                    instance_properties[key] = value

            # Otherwise we're parsing a new Instance for the current exercise
            else:
                line = line.split()
                if training:
                    assert len(line) == 7
                else:
                    assert len(line) == 6
                assert len(line[0]) == 12

                instance_properties['instance_id'] = line[0]

                instance_properties['token'] = line[1]
                instance_properties['part_of_speech'] = line[2]

                instance_properties['morphological_features'] = dict()
                for l in line[3].split('|'):
                    [key, value] = l.split('=')
                    if key == 'Person':
                        value = int(value)
                    instance_properties['morphological_features'][key] = value

                instance_properties['dependency_label'] = line[4]
                instance_properties['dependency_edge_head'] = int(line[5])
                if training:
                    label = float(line[6])
                    labels[instance_properties['instance_id']] = label
                data.append(InstanceData(instance_properties=instance_properties))

        print('Done loading ' + str(len(data)) + ' instances across ' + str(num_exercises) +
              ' exercises.\n')

    if training:
        return data, labels
    else:
        return data


class InstanceData(object):
    """
    A bare-bones class to store the included properties of each instance. This is meant to act as easy access to the
    data, and provides a launching point for deriving your own features from the data.
    """
    def __init__(self, instance_properties):

        # Parameters specific to this instance
        self.instance_id = instance_properties['instance_id']
        self.token = instance_properties['token']
        self.part_of_speech = instance_properties['part_of_speech']
        self.morphological_features = instance_properties['morphological_features']
        self.dependency_label = instance_properties['dependency_label']
        self.dependency_edge_head = instance_properties['dependency_edge_head']

        # Derived parameters specific to this instance
        self.exercise_index = int(self.instance_id[8:10])
        self.token_index = int(self.instance_id[10:12])

        # Derived parameters specific to this exercise
        self.exercise_id = self.instance_id[:10]

        # Parameters shared across the whole session
        self.user = instance_properties['user']
        self.countries = instance_properties['countries']
        self.days = instance_properties['days']
        self.client = instance_properties['client']
        self.session = instance_properties['session']
        self.format = instance_properties['format']
        self.time = instance_properties['time']

        # Derived parameters shared across the whole session
        self.session_id = self.instance_id[:8]

    def to_features(self):
        """
        Prepares those features that we wish to use in the LogisticRegression example in this file. We introduce a bias,
        and take a few included features to use. Note that this dict restructures the corresponding features of the
        input dictionary, 'instance_properties'.

        Returns:
            to_return: a representation of the features we'll use for logistic regression in a dict. A key/feature is a
                key/value pair of the original 'instance_properties' dict, and we encode this feature as 1.0 for 'hot'.
        """
        to_return = dict()

        to_return['bias'] = 1.0
        to_return['user:' + self.user] = 1.0
        to_return['format:' + self.format] = 1.0
        to_return['token:' + self.token.lower()] = 1.0

        to_return['part_of_speech:' + self.part_of_speech] = 1.0
        for morphological_feature in self.morphological_features:
            to_return['morphological_feature:' + morphological_feature] = 1.0
        to_return['dependency_label:' + self.dependency_label] = 1.0

        return to_return


class LogisticRegressionInstance(namedtuple('Instance', ['features', 'label', 'name'])):
    """
    A named tuple for packaging together the instance features, label, and name.
    """
    def __new__(cls, features, label, name):
        if label:
            if not isinstance(label, (int, float)):
                raise TypeError('LogisticRegressionInstance label must be a number.')
            label = float(label)
        if not isinstance(features, dict):
            raise TypeError('LogisticRegressionInstance features must be a dict.')
        return super(LogisticRegressionInstance, cls).__new__(cls, features, label, name)


class LogisticRegression(object):
    """
    An L2-regularized logistic regression object trained using stochastic gradient descent.
    """

    def __init__(self, sigma=_DEFAULT_SIGMA, eta=_DEFAULT_ETA):
        super(LogisticRegression, self).__init__()
        self.sigma = sigma  # L2 prior variance
        self.eta = eta  # initial learning rate
        self.weights = defaultdict(lambda: uniform(-1.0, 1.0)) # weights initialize to random numbers
        self.fcounts = None # this forces smaller steps for things we've seen often before

    def predict_instance(self, instance):
        """
        This computes the logistic function of the dot product of the instance features and the weights.
        We truncate predictions at ~10^(-7) and ~1 - 10^(-7).
        """
        a = min(17., max(-17., sum([float(self.weights[k]) * instance.features[k] for k in instance.features])))
        return 1. / (1. + math.exp(-a))

    def error(self, instance):
        return instance.label - self.predict_instance(instance)

    def reset(self):
        self.fcounts = defaultdict(int)

    def training_update(self, instance):
        if self.fcounts is None:
            self.reset()
        err = self.error(instance)
        for k in instance.features:
            rate = self.eta / math.sqrt(1 + self.fcounts[k])
            # L2 regularization update
            if k != 'bias':
                self.weights[k] -= rate * self.weights[k] / self.sigma ** 2
            # error update
            self.weights[k] += rate * err * instance.features[k]
            # increment feature count for learning rate
            self.fcounts[k] += 1

    def train(self, train_set, iterations=10):
        for it in range(iterations):
            print('Training iteration ' + str(it+1) + '/' + str(iterations) + '...')
            shuffle(train_set)
            for instance in train_set:
                self.training_update(instance)
        print('\n')

    def predict_test_set(self, test_set):
        return {instance.name: self.predict_instance(instance) for instance in test_set}


if __name__ == '__main__':
    main()
