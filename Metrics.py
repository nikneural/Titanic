import pandas as pd
import numpy as np
from pygam import PoissonGAM, s
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ConvertData import ConvertData as cd
from typing import List


class Metrics:
    def __init__(self, df_cp_best: pd.DataFrame, test_interval: str):
        self.result = None
        self.df_cp_best = df_cp_best
        self.test_interval = test_interval

    @staticmethod
    def wape(y, predict):
        return sum(abs(y - predict)) / sum(abs(y))

    def GAM(self, X, Y, params=None, n_splines=20, folds=5):
        # set up GAM
        formula = s(0, n_splines, constraints='monotonic_dec')
        for j in range(1, 6):
            formula = formula + s(j, n_splines, constraints='monotonic_dec')
        for i in range(6, 9):
            formula = formula + s(i, n_splines)
        gam = PoissonGAM(formula)
        gam.fit(X, X.iloc[:, 0])

        # run full model
        GAM_results = {}
        for name, y in Y.iteritems():
            print("\nFitting for %s\n" % name)
            CV = KFold(folds)
            pred = np.zeros(y.shape[0])
            for train, test in CV.split(X, y):
                Xtrain = X.iloc[train, :]
                ytrain = y.iloc[train]
                Xtest = X.iloc[test, :]
                gam = PoissonGAM(formula)
                gam.gridsearch(Xtrain, ytrain, **params)

                # out of fold
                p = gam.predict(Xtest)
                if len(p.shape) > 1:
                    p = p[:, 0]
                pred[test] = p

            cv_scores = [{'r': np.corrcoef(y, pred)[0, 1],
                          'R2': np.corrcoef(y, pred)[0, 1] ** 2,
                          'MAE': mean_absolute_error(y, pred),
                          'Wape': self.wape(y, pred),
                          'RMSE': np.sqrt(mean_squared_error(y, pred, squared=True))}]

            # insample
            gam.gridsearch(X, y, **params)
            in_pred = gam.predict(X)
            in_scores = [{'r': np.corrcoef(y, in_pred)[0, 1],
                          'R2': np.corrcoef(y, in_pred)[0, 1] ** 2,
                          'MAE': mean_absolute_error(y, in_pred),
                          'Wape': self.wape(y, in_pred),
                          'RMSE': (mean_squared_error(y, pred, squared=False))}]
            GAM_results[name] = {'scores_cv': cv_scores,
                                 'scores_insample': in_scores,
                                 'pred_vars': X.columns,
                                 'model': gam,
                                 'pred': pred}
            gam.summary()
        return GAM_results

    def fit_gam(self, max_iter: List[int, int, int], lam: List[int, int, int]):
        df_cp_train_best = self.df_cp_best[self.df_cp_best.orig_date < self.test_interval].copy()
        df_train = cd.add_features_cpt(df_cp_train_best)
        X = df_train
        Y = X['m_orders']
        X.drop(['company', 'orig_date', 'm_price', 'price', 'm_orders'], axis=1, inplace=True)
        X.reset_index(drop=True, inplace=True)
        Y.reset_index(drop=True, inplace=True)
        Y = pd.DataFrame(Y, columns=['m_orders'])
        max_iter_start, max_iter_end, max_iter_step = max_iter
        lam_start, lam_end, lam_step = lam
        params = {
            'max_iter': range(max_iter_start, max_iter_end, max_iter_step),
            'lam': np.arange(lam_start, lam_end, lam_step),
        }
        result = self.GAM(X, Y, params)
        self.result = result
