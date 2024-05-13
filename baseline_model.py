import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from base_model import BaseModel
from typing import Dict, List, Optional
sns.set_style('darkgrid')


class MeansModel(BaseModel):
    '''
        Модель бейзлайна 
        
        colname_analog_price - поле цены аналога
        agg_function - функция аггрегации (считаем среднюю цену аналога)
        find_analogs_num - список значений для подбора максимального числа аналогов
        find_max_area_diff - список значений для подбора максимального отклонения по площади
        coverage -- выводим ли покрытие
        analogs_num - максимально число аналогов
        max_area_total_diff - максимальное отклонение площади аналога от объекта
        sort_by - список полей, по котором сортируем сгруппированные аналоги (по возрастанию)

    '''

    OBJECT_ID_NAME = 'a_offer_id'
    ANALOG_ID_NAME = 'b_offer_id'
    OBJECT_PRICE_NAME = 'a_price'
    ANALOG_PRICE_NAME = 'b_price'
    TARGET_NAME = 'target_price'
    GROUPBY_COLUMNS = ['a_offer_id', 'a_price']
    AGG_COLUMNS = ['b_price']
    RENAME_DICT = {'b_price': 'target_price'}

    def __init__(
        self,
        colname_analog_price: str = 'b_price',
        agg_function: str = 'mean',
        find_analogs_num: Optional[List[int]] = None,
        find_max_area_diff: Optional[List[int]] = None,
        coverage: bool = False,
        analogs_num: Optional[int] = None,
        max_area_total_diff: Optional[float] = None,
        sort_by: Optional[List[str]] = None,
    ):
        self.colname_analog_price = colname_analog_price
        self.agg_function = agg_function
        self.find_analogs_num = find_analogs_num
        self.find_max_area_diff = find_max_area_diff
        self.coverage = coverage
        self.analogs_num = analogs_num
        self.max_area_total_diff = max_area_total_diff
        self.sort_by = sort_by

    def filter_analogs_num(self, df, analogs_num):
        '''
            Берем максимально число аналогов
        '''

        df['b_offer_id_num'] = df.groupby(self.OBJECT_ID_NAME).cumcount() + 1
        return df[df['b_offer_id_num'] <= analogs_num]

    @staticmethod
    def filter_area_total_diff(df, max_area_total_diff):
        '''
            Фильтруем по максимальному отклонению по площади
        '''

        return df[df['area_total_diff'] <= max_area_total_diff]
    
    @staticmethod
    def plot_graph(x, y, title, x_title, y_title):
        '''
            Рисуем график
        '''
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=x, y=y)
        plt.title(title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.show()

    def fit(self, df_train: pd.DataFrame, title=str):
        '''
            Делаем перебор параметров на тренировочных данных
        '''
        df_train_modified = df_train
        if self.sort_by is not None:
            df_train_modified=df_train_modified.sort_values(
                by=self.sort_by, ascending=True)

        if self.find_analogs_num is not None:
            mape_grid=[]
            for analog_num in self.find_analogs_num:
                df_train_modified_new=self.filter_analogs_num(df_train_modified, analog_num)
                df_result=self.calculate_metric(self.predict(df_train_modified_new))
                mape_grid.append(df_result)
            self.plot_graph(self.find_analogs_num, mape_grid, title, 'Число аналогов', 'MAPE')
            return mape_grid

        elif self.find_max_area_diff is not None:
            mape_grid=[]
            coverage_grid=[]
            for max_area_diff in self.find_max_area_diff:
                len_before = len(df_train_modified['a_offer_id'].drop_duplicates())
                df_train_modified_new=self.filter_area_total_diff(df_train_modified, max_area_diff)
                len_after = len(df_train_modified_new['a_offer_id'].drop_duplicates())
                df_result=self.calculate_metric(self.predict(df_train_modified_new))
                mape_grid.append(df_result)
                coverage_grid.append(len_after / len_before * 100)
            if self.coverage:
                self.plot_graph(self.find_max_area_diff, coverage_grid, title, 'Отклонение по площади', 'Покрытие %')
            else:
                self.plot_graph(self.find_max_area_diff, mape_grid, title, 'Отклонение по площади', 'MAPE')
            return mape_grid, coverage_grid    
        return mape_grid

    def predict(self, df_test: pd.DataFrame) -> pd.DataFrame:
        '''
            Финальная модель бейзлайна с подобранными параметрами
        '''

        if self.sort_by is not None:
            df_test=df_test.sort_values(by=self.sort_by, ascending=True)

        if self.max_area_total_diff is not None:
            df_test=self.filter_area_total_diff(
                df_test, self.max_area_total_diff)

        elif self.analogs_num is not None:
            df_test=self.filter_analogs_num(df_test, self.analogs_num)

        return self.predict_mean(df_test)
