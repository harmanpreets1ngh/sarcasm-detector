import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def train_and_validate_svm(train_data, train_labels, test_data, test_labels, \
                           kernel='rbf', C=1.0, gamma='scale', verbose=False, class_weight='balanced'):
    print('training svm with kernel=%s, C=%s, gamma=%s, class_weight=%s' % (kernel, C, gamma, class_weight))
    svm = SVC(kernel=kernel, C=C, gamma='scale', verbose=verbose, \
              class_weight=class_weight, cache_size=1000)
    svm.fit(train_data, train_labels)
    predicted_labels = svm.predict(test_data)
    print('test', test_labels[:10], test_labels[-10:])
    print('pred', predicted_labels[:10], predicted_labels[-10:])
    mse = mean_squared_error(test_labels, predicted_labels)
    print('mean squared error:', mse)
    pearson = pearsonr(test_labels, predicted_labels)
    print('pearson coefficient:', pearson)
    f_score = f1_score(test_labels, predicted_labels)
    print('f-score:', f_score)
    return mse, pearson, f_score

def cross_validate_svm(data, labels, \
                       kernel='linear', C=1.0, gamma='scale', verbose=False, class_weight='balanced'):
    print('cross-validating svm...')
    num_cross_validation_trials = 10
    kfold = KFold(num_cross_validation_trials, True, 1)

    mses = []
    pearsons = []
    f_scores = []
    for trial_index, (train, val) in enumerate(kfold.split(data)):
        print((" Trial %d of %d" % (trial_index+1, num_cross_validation_trials)).center(80, '-'))

        mse, (pearson_r, pearson_p), f_score = \
         train_and_validate_svm(data[train], labels[train], data[val], labels[val], \
                                kernel, C, gamma, verbose, class_weight)
        mses.append(mse)
        pearsons.append(pearson_r)
        f_scores.append(f_score)
    print_results_of_trials(mses, pearsons, f_scores)
    return mses, pearsons

def print_results_of_trials(mses, pearsons, f_scores):
    print_metric_results_of_trials(mses, "Mean Squared Error")
    print_metric_results_of_trials(pearsons, "Pearson Coefficients")
    print_metric_results_of_trials(f_scores, "F-Scores")

def print_metric_results_of_trials(metrics_over_trials, metric_name):
    np_metrics = np.array(metrics_over_trials)

    min_metric = np_metrics.min()
    avg_metric = np.average(np_metrics)
    max_metric = np_metrics.max()

    print(("> %s over all trials <" % metric_name).center(80, '='))
    print("%s min:  %7.4f" % (metric_name, min_metric))
    print("%s mean: %7.4f" % (metric_name, avg_metric))
    print("%s max:  %7.4f" % (metric_name, max_metric))
    print(80*'=')
