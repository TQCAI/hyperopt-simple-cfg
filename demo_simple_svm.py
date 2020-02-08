from hyperopt import Trials, fmin, tpe
from sklearn import svm
from simple_cfg import estimator_option, uniform, choice, condition_option, uniformint, to_hp, estimator_from_cfg

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
trials = Trials()
best = fmin(estimator_from_cfg,
            space=sp,
            algo=tpe.suggest,
            max_evals=2000,
            trials=trials)
print(best)