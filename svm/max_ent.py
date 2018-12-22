import svm

import numpy as np

from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def train_and_validate_lr(train_data, train_labels, test_data, test_labels, \
                           penalty='l2', solver='sag', C=1.0, verbose=0, \
                           class_weight='balanced', max_iter=1000):
    print('training max_ent with penalty=%s, solver=%s, C=%s, and class_weight=%s' % (penalty, solver, C, class_weight))
    lr = LR(penalty=penalty, solver=solver, C=C, verbose=verbose,
            class_weight=class_weight, max_iter=max_iter)
    lr.fit(train_data, train_labels)
    predicted_labels = lr.predict(test_data)
    print('test', test_labels[:10], test_labels[-10:])
    print('pred', predicted_labels[:10], predicted_labels[-10:])
    mse = mean_squared_error(test_labels, predicted_labels)
    print('mean squared error:', mse)
    pearson = pearsonr(test_labels, predicted_labels)
    print('pearson coefficient:', pearson)
    f_score = f1_score(test_labels, predicted_labels)
    print('f-score:', f_score)
    return mse, pearson, f_score

def cross_validate_lr(data, labels, \
                       penalty='l2', solver='sag', C=1.0, verbose=0, \
                       class_weight='balanced', max_iter=1000):
    print('cross-validating max-ent...')
    num_cross_validation_trials = 10
    kfold = KFold(num_cross_validation_trials, True, 1)

    mses = []
    pearsons = []
    f_scores = []
    for trial_index, (train, val) in enumerate(kfold.split(data)):
        print((" Trial %d of %d" % (trial_index+1, num_cross_validation_trials)).center(80, '-'))

        mse, (pearson_r, pearson_p), f_score = \
         train_and_validate_lr(data[train], labels[train], data[val], labels[val], \
                                penalty, solver, C, verbose, class_weight, max_iter)
        mses.append(mse)
        pearsons.append(pearson_r)
        f_scores.append(f_score)
    svm.print_results_of_trials(mses, pearsons, f_scores)
    return mses, pearsons
