from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
import hyperopt
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.datasets import load_iris
import numpy as np
iris=load_iris()
space = {
    'C': hp.uniform('C', 0.001, 1000),
    'shrinking': hp.choice('shrinking', [True, False]),
    'kernel': hp.choice('kernel', [
        {
            'name': 'rbf',
            'gamma': hp.choice('rbf_gamma', ['auto', hp.uniform('rbf_gamma_uniform',0.0001, 8)])
        },
        {
            'name': 'linear',
        },
        {
            'name': 'sigmoid',
            'gamma': hp.choice('sigmoid_gamma', ['auto', hp.uniform('sigmoid_gamma_uniform',0.0001, 8)]),
            'coef0': hp.uniform('sigmoid_coef0', 0, 10)
        },
        {
            'name': 'poly',
            'gamma': hp.choice('poly_gamma', ['auto', hp.uniform('poly_gamma_uniform',0.0001, 8)]),
            'coef0': hp.uniform('poly_coef0', 0, 10),
            'degree': hp.uniformint('poly_degree', 1, 5),
        }
    ])
}


def svm_from_cfg(cfg):
    kernel_cfg=cfg.pop('kernel')
    kernel=kernel_cfg.pop('name')
    cfg.update(kernel_cfg)
    cfg['kernel']=kernel
    clf=svm.SVC(**cfg)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    return {
        'loss':1 - np.mean(scores),
        'status':STATUS_OK
    }
trials = Trials()
best = fmin(svm_from_cfg,
    space=space,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials)
print(best)
# print(hyperopt.pyll.stochastic.sample(space))
