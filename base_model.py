import pandas as pd
from metric import mean_absolute_percentage_error_value, mean_absolute_percentage_error, calculate_mape

class BaseModel:
    def fit(self, df_train: pd.DataFrame):
        raise NotImplementedError

    def predict(self, df_test: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
    
    def calculate_metric(self, agg_data) -> int:
        agg_data = agg_data[[self.OBJECT_ID_NAME, self.OBJECT_PRICE_NAME, self.TARGET_NAME]].drop_duplicates()
        return mean_absolute_percentage_error(agg_data[self.OBJECT_PRICE_NAME], agg_data[self.TARGET_NAME])
    
    def predict_mean(self, df: pd.DataFrame) -> pd.DataFrame:   
        return df.groupby(self.GROUPBY_COLUMNS)[self.AGG_COLUMNS].agg(
            self.agg_function
        ).reset_index().rename(self.RENAME_DICT, axis=1)
