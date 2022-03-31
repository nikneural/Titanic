import matplotlib.pyplot as plt
import pandas as pd


class RepricingStatistics:
    def __init__(self, df_cp: pd.DataFrame, df_cp_best: pd.DataFrame, target_product: str, main_company: str,
                 period: int, test_start: str, test_end: str):
        self.prices_df = None
        assert (df_cp.index == df_cp_best.index).all(), 'index is not equal'
        assert (df_cp[['company', 'name']] == df_cp_best[
            ['company', 'name']]).all().all(), 'company and name is not equal'
        self.df_cp = df_cp
        self.df_cp_best = df_cp_best
        self.target_product = target_product
        self.main_company = main_company
        self.period = period
        self.test_start = test_start
        self.test_end = test_end
        self.allow_plot = True
        self.total_df = None
        self.mean_profit_best = None
        self.mean_profit = None
        self.mean_prop_best = None
        self.mean_prop = None
        self.mean_orders_sum = None
        self.mean_orders_best = None
        self.mean_orders = None
        self.total_summary()

    @staticmethod
    def tall_to_flat(data: pd.DataFrame):
        data = data.copy()
        data = data.set_index('orig_date')
        data = data.pivot(columns=['name', 'company'])
        data.columns = data.columns.swaplevel(0, 2)
        data = data.sort_index(axis=1, level=[0, 1])
        return data

    def plot_data(self, data: pd.DataFrame, title: str, legend: plt.legend):
        fig, ax = plt.subplots(1, 1, figsize=(15, 3))
        data.plot(ax=ax)
        ax.set_title(title)
        ax.legend(legend, loc='upper left')
        if not self.allow_plot:
            plt.close()
        return fig

    def generate_plots(self, allow_plot: bool = True):
        self.allow_plot = allow_plot
        idx = pd.IndexSlice
        mean_orders_sum = self.total_df.loc[:, idx[:, 'm_orders']].sum(axis=1)
        prices_df = self.total_df.loc[:, idx[:, 'price']]
        prices_df.loc[:, idx['Best price', 'price']] = self.total_df.loc[:, idx[self.main_company, 'best_price']]

        # Plots
        def_legend = self.total_df.columns.levels[0]
        self.mean_orders_sum = self.plot_data(mean_orders_sum,
                                              title=f'Суммарные средние продажи с учетом конкурентов. MA({self.period})',
                                              legend='')
        self.mean_orders = self.plot_data(self.total_df.loc[:, idx[:, 'm_orders']],
                                          title=f'Средние продажи. MA({self.period})',
                                          legend=def_legend)
        self.mean_orders_best = self.plot_data(self.total_df.loc[:, idx[:, 'best_m_orders']],
                                               title=f'Средние продажи с учетом best price. MA({self.period})',
                                               legend=def_legend)
        self.mean_prop = self.plot_data(self.total_df.loc[:, idx[:, 'proportion']],
                                        title=f'Доли продаж производителей. MA({self.period})',
                                        legend=def_legend)
        self.mean_prop_best = self.plot_data(self.total_df.loc[:, idx[:, 'best_proportion']],
                                             title=f'Доли продаж производителей с учетом best price. MA({self.period})',
                                             legend=def_legend)
        self.prices_df = self.plot_data(prices_df.loc[:, idx[:, 'price']],
                                        title=f'Цены',
                                        legend=prices_df.columns.levels[0])
        self.mean_profit = self.plot_data(self.total_df.loc[:, idx[self.main_company, ['profit', 'best_profit']]],
                                          title=f'Средняя прибыль с учетом best_price. MA({self.period})',
                                          legend=['Ordinary price', 'Best Price'])

    def total_summary(self):
        idx = pd.IndexSlice
        total_df = self.df_cp[['orig_date', 'company', 'name', 'm_orders', 'price', 'msrp']].copy()
        total_df['best_m_orders'] = self.df_cp_best['m_orders']
        total_df['best_price'] = self.df_cp_best['price']
        total_df = self.tall_to_flat(total_df).copy()
        total_df.columns = total_df.columns.droplevel(1)
        total_df.columns = total_df.columns.remove_unused_levels()

        best_m_orders_sum = total_df.loc[:, idx[:, 'best_m_orders']].sum(axis=1)
        m_orders_sum = total_df.loc[:, idx[:, 'm_orders']].sum(axis=1)
        for company in total_df.columns.levels[0]:
            total_df.loc[:, idx[company, 'proportion']] = total_df.loc[:, idx[company, 'm_orders']] / m_orders_sum
            total_df.loc[:, idx[company, 'best_proportion']] = total_df.loc[:,
                                                               idx[company, 'best_m_orders']] / best_m_orders_sum
            total_df.loc[:, idx[company, 'profit']] = total_df.loc[:, idx[company, 'm_orders']] * \
                                                      (total_df.loc[:, idx[company, 'price']] - total_df.loc[:,
                                                                                                idx[company, 'msrp']])
            total_df.loc[:, idx[company, 'best_profit']] = total_df.loc[:, idx[company, 'best_m_orders']] * \
                                                           (total_df.loc[:, idx[company, 'best_price']] - total_df.loc[
                                                                                                          :, idx[
                                                                                                                 company, 'msrp']])

        total_df = total_df.iloc[:-1]
        total_df = total_df.fillna(0)
        total_df = total_df.sort_index(axis=1)
        total_df.index = total_df.index.astype('datetime64[ns]')
        self.total_df = total_df

    @staticmethod
    def calculate_value(interval: pd.DataFrame, m_orders: str, price: str,
                        proportion: str, msrp: str = 'msrp', desc: str = '') -> pd.DataFrame:
        real_profit = interval[m_orders] * (interval[price] - interval[msrp])

        summary = {'interval': [desc], 'from': [interval.index[0]], 'to': [interval.index[-1]],
                   'mean_m_orders': [interval[m_orders].mean()], 'median_m_orders': [interval[m_orders].median()],
                   'mean_price': [interval[price].mean()], 'median_price': [interval[price].median()],
                   'mean_proportion': [interval[proportion].mean()],
                   'median_proportion': [interval[proportion].median()], 'profit': [real_profit.sum()],
                   'profit_mean': [real_profit.mean()]}

        return pd.DataFrame.from_dict(summary)

    def summary_calculation(self) -> pd.DataFrame:
        mask = self.total_df.index.isin(pd.date_range(self.test_start, self.test_end, freq='1d'))
        train_interval = self.total_df.loc[~mask, self.main_company].copy()
        test_interval = self.total_df.loc[mask, self.main_company].copy()
        test_best_interval = self.total_df.loc[mask, self.main_company].copy()

        train_stat = self.calculate_value(train_interval, 'm_orders', 'price', 'proportion', desc='Train')
        test_stat = self.calculate_value(test_interval, 'm_orders', 'price', 'proportion', desc='Test')
        test_best_stat = self.calculate_value(test_best_interval, 'best_m_orders', 'best_price',
                                              'best_proportion', desc='Test best price')

        full = pd.concat([train_stat, test_stat, test_best_stat], ignore_index=True)
        return full
