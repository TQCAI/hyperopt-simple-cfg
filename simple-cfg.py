from collections import namedtuple
from copy import deepcopy
from importlib import import_module
from typing import Union

import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

from hyperopt import STATUS_OK, Trials, fmin, tpe, STATUS_FAIL

iris = load_iris()


class FacadeMixin():
    def get_kwargs(self):
        raise NotImplementedError()

    def get_function(self):
        if 'hyperopt' not in locals():
            hyperopt = import_module('hyperopt')
        fname = f'hyperopt.hp.{self.__class__.__name__}'
        return eval(fname)


class uniform(namedtuple('uniform', ['low', 'high']), FacadeMixin):
    def get_kwargs(self):
        return {'low': self.low, 'high': self.high}


class choice(namedtuple('choice', ['options']), FacadeMixin):
    def get_kwargs(self):
        return {'options': self.options}


class uniformint(namedtuple('uniformint', ['low', 'high']), FacadeMixin):
    def get_kwargs(self):
        return {'low': self.low, 'high': self.high}


class condition_option(dict):
    def __init__(self, name, **kwargs):
        super(condition_option, self).__init__()
        self.name = name
        self.update(kwargs)
        self.update({f'condition': name})

    def __str__(self):
        return f'condition_option({super().__str__()})'

    def __repr__(self):
        return f'condition_option({super().__repr__()})'


class estimator_option(dict):
    def __init__(self, estimator, **kwargs):
        super(estimator_option, self).__init__()
        self.estimator = estimator
        self.update(kwargs)
        self.update({'estimator': estimator})

    def __str__(self):
        return f'estimator_option({super().__str__()})'

    def __repr__(self):
        return f'estimator_option({super().__repr__()})'


def __get_prefix(prefix, name):
    if not prefix:
        return str(name)
    else:
        return f'{prefix}_{name}'


def to_hp(x: Union[dict, list, tuple, FacadeMixin], prefix=''):
    if isinstance(x, dict):
        for k, v in x.items():
            cur_prefix = __get_prefix(prefix, k)
            if isinstance(v, (FacadeMixin, dict, list, tuple)):
                x[k] = to_hp(v, cur_prefix)
        return x
    elif isinstance(x, FacadeMixin):
        kwargs: dict = x.get_kwargs()
        for sk, sv in kwargs.items():
            if isinstance(sv, (FacadeMixin, dict, list, tuple)):  # such as options arguments maybe
                cur_prefix = prefix
                if isinstance(sv, FacadeMixin):
                    cur_prefix = __get_prefix(prefix, sv.__class__.__name__)
                sv = to_hp(sv, cur_prefix)
            kwargs[sk] = sv
        values = list(kwargs.values())
        return x.get_function()(prefix, *values)
    elif isinstance(x, (tuple, list)):
        cls = x.__class__
        lst = []
        for ix, ele in enumerate(x):
            if isinstance(ele, (FacadeMixin, dict, list, tuple)):
                cur_prefix = __get_prefix(prefix, ix)
                ele = to_hp(ele, cur_prefix)
            lst.append(ele)
        return cls(lst)
    else:
        raise ValueError(f'Invalid type ({type(x)}) in recursion ')


space = estimator_option(
    svm.SVC,
    C=uniform(0.001, 1000),
    shrinking=choice([True, False]),
    kernel=choice([
        condition_option(
            'rbf',
            gamma=choice(['auto', uniform(0.0001, 8)])
        ),
        condition_option(
            'linear'
        ),
        condition_option(
            'sigmoid',
            gamma=choice(['auto', uniform(0.0001, 8)]),
            coef0=uniform(0, 10)
        ),
        condition_option(
            'poly',
            gamma=choice(['auto', uniform(0.0001, 8)]),
            coef0=uniform(0, 10),
            degree=uniformint(1, 5)
        )
    ])
)

sp = to_hp(space)


def estimator_from_cfg(cfg: estimator_option):
    cfg_ = deepcopy(cfg)
    estimator = cfg_['estimator']
    cfg_.pop('estimator')
    for k, v in deepcopy(cfg).items():
        if isinstance(v, dict) and 'condition' in v.keys():
            value_name = v['condition']
            v.pop('condition')
            key_name = k
            cfg_.pop(k)
            cfg_.update(v)
            cfg_.update({key_name: value_name})
    try:
        clf = estimator(**cfg_)
        scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    except:
        print('fail')
        return {'loss': np.inf, 'status': STATUS_FAIL}
    return {
        'loss': 1 - np.mean(scores),
        'status': STATUS_OK,
        'cfg': cfg
    }


trials = Trials()
best = fmin(estimator_from_cfg,
            space=sp,
            algo=tpe.suggest,
            max_evals=2000,
            trials=trials)
print(best)
