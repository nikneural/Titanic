import numpy as np
import pandas as pd


class ConvertData:
    """
    Basic functions for converting the initial DataFrame.
    """

    def __init__(self):
        pass

    @staticmethod
    def init_cpt(df_raw: pd.DataFrame, target_product: str, mean_days: int = 7, predict_forward_days: int = 0):
        predict_forward_days = np.abs(predict_forward_days)
        df_cp = df_raw[df_raw.name == target_product].copy()
        cp_companies = np.unique(df_cp.company)
        df_cp.index = df_cp.orig_date

        df_cp['m_orders'] = np.NaN
        df_cp['m_orders_l1'] = np.NaN
        df_cp['m_price'] = np.NaN
        df_cp['g_m_orders'] = np.NaN
        df_cp['g_c_orders'] = np.NaN
        df_cp['g_rank1'] = np.NaN
        df_cp['g_rank1_min'] = np.NaN
        df_cp['g_rank1_max'] = np.NaN

        for i in cp_companies:
            df_cp['m_orders'][df_cp.company == i] = df_cp['c_orders'][df_cp.company == i].rolling(mean_days).mean().shift(-predict_forward_days)
            df_cp['m_orders_l1'][df_cp.company == i] = df_cp['m_orders'][df_cp.company == i].shift(1)
            df_cp['m_price'][df_cp.company == i] = df_cp['price'][df_cp.company == i].rolling(mean_days).mean()

        df_cp.loc[df_cp['m_orders'] == 0, 'm_orders'] = 0.0001
        df_cp.loc[df_cp['m_orders_l1'] == 0, 'm_orders_l1'] = 0.0001

        for i in np.unique(df_cp.index):
            df_cp['g_c_orders'].loc[i] = df_cp['c_orders'].loc[i].sum()
            df_cp['g_m_orders'].loc[i] = df_cp['m_orders'].loc[i].sum()
            df_cp['g_rank1'].loc[i] = df_cp['rank1'].loc[i].mean()
            df_cp['g_rank1_min'].loc[i] = df_cp['rank1'].loc[i].min()
            df_cp['g_rank1_max'].loc[i] = df_cp['rank1'].loc[i].max()

        return df_cp

    @staticmethod
    def algo_price(data: pd.DataFrame, orders_weighted: bool = False):
        data = data.copy()
        data = data.dropna(subset=['m_price'])
        res = {'p_mean': data.m_price.mean(), 'p_med': data.m_price.median(), 'p_max': data.m_price.max(),
               'p_min': data.m_price.min(), 'p_rw': (data['m_price'] * (1 / data['rank1'].fillna(np.inf))).sum() / (
                    1 / data['rank1'].fillna(np.inf)).sum()}
        if orders_weighted:
            res['p_sw'] = (data['m_price'] * (data['m_orders_l1'].fillna(0))).sum() / (
                data['m_orders_l1'].fillna(0)).sum()
        return res

    def add_features_cpt(self, df_cp: pd.DataFrame, fill_last_na: bool = False):
        df_cp = df_cp.copy()
        df_cp['g_p_mean'] = np.NaN
        df_cp['g_p_med'] = np.NaN
        df_cp['g_p_rw'] = np.NaN
        df_cp['g_p_sw'] = np.NaN
        df_cp['g_p_min'] = np.NaN
        df_cp['g_p_max'] = np.NaN
        for i in np.unique(df_cp.index):
            prices = self.algo_price(df_cp.loc[i], orders_weighted=True)
            df_cp['g_p_mean'].loc[i] = prices['p_mean']
            df_cp['g_p_med'].loc[i] = prices['p_med']
            df_cp['g_p_rw'].loc[i] = prices['p_rw']
            df_cp['g_p_sw'].loc[i] = prices['p_sw']
            df_cp['g_p_min'].loc[i] = prices['p_min']
            df_cp['g_p_max'].loc[i] = prices['p_max']

        df_cp['rank1_min'] = df_cp['rank1'] / df_cp['g_rank1_min']
        df_cp['rank1_max'] = df_cp['rank1'] / df_cp['g_rank1_max']
        df_cp['rank1'] = df_cp['rank1'] / df_cp['g_rank1']
        df_cp['m_orders'] = df_cp['m_orders'] / df_cp['g_m_orders']
        df_cp['p_mean'] = df_cp['price'] / df_cp['g_p_mean']
        df_cp['p_med'] = df_cp['price'] / df_cp['g_p_med']
        df_cp['p_rw'] = df_cp['price'] / df_cp['g_p_rw']
        df_cp['p_sw'] = df_cp['price'] / df_cp['g_p_sw']
        df_cp['p_min'] = df_cp['price'] / df_cp['g_p_min']
        df_cp['p_max'] = df_cp['price'] / df_cp['g_p_max']

        df_cp.index = df_cp.orig_date
        if fill_last_na:
            df_cp.loc[df_cp.index[-1], ['m_orders']] = df_cp.loc[df_cp.index[-1], ['m_orders']].fillna(0)
        df_cp = df_cp[
            ['company', 'm_orders', 'm_price', 'orig_date', 'p_max', 'p_mean', 'p_med', 'p_min', 'p_rw', 'p_sw',
             'price', 'rank1', 'rank1_max', 'rank1_min']]
        df_cp = df_cp.dropna()
        df_cp = df_cp.sort_index(axis=1)

        return df_cp
