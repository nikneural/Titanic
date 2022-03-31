import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import PoissonGAM, s
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

from ConvertData import ConvertData as cd, ConvertData
from Repricer import Repricer


class Preprocessing:
    def __init__(self, path, target_product, company, train_interval):
        self.df_cp = None
        self.repricer = None
        self.data_rw = None
        self.path = path
        self.target_product = target_product
        self.company = company
        self.train_interval = train_interval
        self.y_column = 'm_orders'

    def run_GAM(self, X, Y, params=None, get_importance=False, n_splines=20, folds=5):
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
                ytest = y.iloc[test]
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
            #         gam.fit(X, y)
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
        return GAM_results

    def load_data(self):
        """Return two DataFrames: data_rw and data_rw_best"""
        data_rw = pd.read_csv(self.path)
        return data_rw

    @staticmethod
    def wape(y, predict):
        return sum(abs(y - predict)) / sum(abs(y))

    def RepricerWork(self):
        data_rw = self.load_data()
        df_cp = cd.init_cpt(data_rw, self.target_product, mean_days=5, predict_forward_days=0)
        self.df_cp = df_cp
        df_fcp = cd.add_features_cpt(ConvertData, df_cp=df_cp, fill_last_na=False)
        order_cols = df_fcp.columns.drop([self.y_column, 'orig_date', 'company', 'price', 'm_price'])
        df_cp_train = df_cp[df_cp.orig_date <= self.train_interval].copy()
        df_cp_test = df_cp[df_cp.orig_date > self.train_interval].copy()
        gam = PoissonGAM(s(0, 20, constraints='monotonic_dec') + s(1, 20, constraints='monotonic_dec') +
                         s(2, 20, constraints='monotonic_dec') + s(3, 20, constraints='monotonic_dec') +
                         s(4, 20, constraints='monotonic_dec') + s(5, 20, constraints='monotonic_dec') +
                         s(6, 20) + s(7, 20) + s(8, 20))
        repricer = Repricer(df_cp_train, df_cp_test, self.y_column, self.company, gam, order_cols)
        repricer.fit()
        repricer.get_price(k_max=2, n_price_samples=15)
        return repricer

    def fill_gaps(self, data):
        repricer = self.RepricerWork()
        date = []
        for dt in data.loc[(data['company'] == self.company) &
                           (data['orig_date'] > self.train_interval) &
                           (data['name'] == self.target_product), ['orig_date']].values:
            date.append(dt[0])

        index = []
        for idx in repricer.best_prices.index:
            index.append(idx)

        values = []
        for dt in date:
            if dt not in index:
                values.append(data.loc[(data['company'] == self.company) &
                                       (data['orig_date'] > '2021-11-01') &
                                       (data['name'] == self.target_product) &
                                       (data['orig_date'] == dt)]['price'].values[0])
            else:
                values.append(repricer.best_prices[dt])

        if len(data.loc[(data['company'] == self.company) & (data.orig_date > '2021-11-01')
                        & (data.name == self.target_product), ['price']]) == len(repricer.best_prices):
            data.loc[(data['company'] == self.company) & (data.orig_date > '2021-11-01')
                     & (data.name == self.target_product), ['price']] = repricer.best_prices.values
        else:
            data.loc[(data['company'] == self.company) & (data.orig_date > '2021-11-01')
                     & (data.name == self.target_product), ['price']] = values
        return data

    def WorkWithBest(self):
        data_rw_best = self.load_data()
        self.fill_gaps(data_rw_best)
        df_cp_best = cd.init_cpt(data_rw_best, self.target_product, mean_days=5, predict_forward_days=0)
        df_cp_train_best = df_cp_best[df_cp_best.orig_date <= self.train_interval].copy()
        df_cp_test_best = df_cp_best[df_cp_best.orig_date > self.train_interval].copy()
        df_train = cd.add_features_cpt(df_cp_train_best)
        test_data = cd.add_features_cpt(df_cp_test_best)
        return df_train, test_data

    def fit(self):
        X, _ = self.WorkWithBest()
        Y = X[self.y_column]
        X.drop([self.y_column, 'company', 'orig_date', 'm_price', 'price'], axis=1, inplace=True)
        X.reset_index(drop=True, inplace=True)
        Y.reset_index(drop=True, inplace=True)
        Y = pd.DataFrame(Y, columns=[self.y_column])
        params = {
            'max_iter': range(50, 500, 50),
            'lam': np.arange(1, 1000, 100),
        }
        result = self.run_GAM(X, Y, params)
        return result

    def predict(self):
        result = self.fit()
        gam = result['m_orders']['model']
        _, test_data = self.WorkWithBest()
        p_orders = test_data.copy()
        p_orders = p_orders.reset_index(drop=True)
        p_orders.drop(['company', 'm_price', 'orig_date', 'price'], axis=1, inplace=True)
        y = p_orders[self.y_column]
        X = p_orders.drop(self.y_column, axis=1)
        preds = gam.predict(X)
        test_data['p_orders'] = preds
        return test_data

    def plot_predict_mean_orders(self, plot_h, plot_l, printed_cols):
        test_data = self.predict()
        for i in test_data.index:
            corr_sum = test_data.iloc[i, 'p_orders'].sum()
            test_data.loc[i, 'p_orders_corr'] = test_data.loc[i, 'p_orders'] / corr_sum
            test_data.loc[:, 'p_orders_corr'] = test_data.loc[:, 'p_orders_corr'].fillna(0)

