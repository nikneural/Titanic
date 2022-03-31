import pandas as pd
import numpy as np
from ConvertData import ConvertData as cd
from sklearn.metrics import r2_score, mean_absolute_error


class Repricer:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, y_column: str, company: str, model, lag_cols):
        """Train - DataFrame the algorithm will train on.\n
           test - Dataframe on which the algorithm will make predictions.\n
           y_column - The column by which the algorithm will make predictions, by default, m_orders.\n
           company - Company for which m_orders will be taken.\n
           model - A model that will make predictions. By default, PoissonGAM.\n
           lag_cols - Columns to be removed from the dataframe.
           """
        super().__init__()
        self.best_prices = None
        self.profit_df = None
        self.prop_df = None
        self.train = train.copy()
        self.test = test.copy()
        self.y_column = y_column
        self.company = company
        self.msrp = self.test['msrp'][self.test.company == company]
        self.lag_cols = lag_cols
        self.model = model
        self.df_cpf_train = cd.add_features_cpt(self.train)
        self.df_cpf_test = cd.add_features_cpt(self.test, fill_last_na=True)

    def fit(self, verbose: bool = True):
        self.model.fit(self.df_cpf_train[self.lag_cols], self.df_cpf_train[self.y_column])
        if verbose:
            res = self.model.predict(self.df_cpf_train[self.lag_cols])
            print('train WAPE:',
                  mean_absolute_error(res, self.df_cpf_train[self.y_column]) / self.df_cpf_train[self.y_column].mean(),
                  'MAE:', mean_absolute_error(res, self.df_cpf_train[self.y_column]),
                  'RMSE:', r2_score(res, self.df_cpf_train[self.y_column]))

    def predict(self, price: pd.DataFrame = None, corr_prop: bool = True):
        test = self.test.copy()
        if price is not None:
            test['price'][test.company == self.company] = price

        df_cpf_test = cd.add_features_cpt(test, fill_last_na=True)

        company_mask = df_cpf_test.company == self.company
        prediction = self.model.predict(df_cpf_test[self.lag_cols])
        prediction[prediction < 0] = 0
        prediction = pd.Series(prediction, index=df_cpf_test.index)
        if corr_prop:
            for i in np.unique(prediction.index):
                corr_sum = prediction.loc[i].sum()
                prediction.loc[i] = prediction.loc[i] / corr_sum
        prediction = prediction[company_mask]

        return prediction.rename(self.y_column)

    def get_price(self, corr_prop: bool = True, n_price_samples: int = 10, k_max: float = 1.5):
        msrp_max = self.msrp.max()
        p_min = self.train['price'].dropna().min()
        print(f'min price {p_min}, msrp_max {msrp_max}')
        p_min = p_min if p_min > msrp_max else msrp_max * 1.01
        print('min price res', p_min)
        p_max = self.train['price'].dropna().max()
        print(f'max price {p_max}, msrp_max* k_max {msrp_max * k_max}')
        #         p_max = p_max if p_max < msrp_max * k_max else msrp_max * k_max
        p_max = msrp_max * k_max
        print('max price res', p_max)
        prop_df = pd.DataFrame([])
        price_line = np.linspace(p_min, p_max, n_price_samples)
        for p in price_line:
            price_res = self.predict(p, corr_prop)
            prop_df = pd.concat([prop_df, price_res.rename(p)], axis=1)
        msrp = self.msrp.loc[prop_df.index]
        price_grid = pd.DataFrame(np.full((len(prop_df.index), n_price_samples), price_line), index=prop_df.index)

        n_col = len(prop_df.columns)
        for i in range(0, len(prop_df.index)):
            min_price = prop_df.iloc[i, 0]
            max_price = prop_df.iloc[i, n_col - 1]
            for j in range(0, n_col - 1):
                if (prop_df.iloc[i, j] == min_price) and (prop_df.iloc[i, j + 1] == min_price):
                    prop_df.iloc[i, j] = np.NaN
                else:
                    break
            for j in range(n_col - 1, 0, -1):
                if (prop_df.iloc[i, j] == max_price) and (prop_df.iloc[i, j - 1] == max_price):
                    prop_df.iloc[i, j] = np.NaN
                else:
                    break

        profit_df = prop_df * (price_grid - msrp.values.reshape(-1, 1)).values

        self.prop_df = prop_df
        self.profit_df = profit_df
        # Маска для лучшей цены. Определяется по максимальной прибыли за день
        price_mask = profit_df.apply(lambda x: x == x.max(), axis=1)
        # Маска для максимальной цены. Используется, когда нету лучшей цены
        max_mask = np.full_like(price_mask.columns, False, dtype=bool)
        max_mask[-1] = True
        # Поиск мест, где нет лучшей цены, т.е. прогноз везде 0 и прибыль 0
        price_mask.iloc[price_mask.all(axis=1) == True] = max_mask
        self.best_prices = price_mask.idxmax(axis=1)
