import pandas as pd
from catboost import Pool, CatBoostRegressor
import catboost
from base_model import BaseModel
from typing import Dict, List, Optional


class CorrectionsModel(BaseModel):
    '''
        Модель корректировок

        analogs_num - максимально число аналогов, отсортированных по коэффиценту корректироки
        params - словарь параметров для модели бустинга
    '''

    PARAMS = {
        'iterations': 100,
        'loss_function': 'MAE',
        'random_seed': 42
    }
    OBJECT_ID_NAME = 'a_offer_id'
    ANALOG_ID_NAME = 'b_offer_id'
    OBJECT_PRICE_NAME = 'a_price'
    ANALOG_PRICE_NAME = 'b_price'
    TARGET_NAME = 'target_price'
    GROUPBY_COLUMNS = ['a_offer_id', 'a_price']
    AGG_COLUMNS = ['price_corrected', 'correction_coeff']
    RENAME_DICT = {'price_corrected': 'target_price',
                   'correction_coeff': 'mean_correction_coeff'}


    def __init__(
        self, 
        params: Optional[Dict] = PARAMS,
        analogs_num: Optional[int] = 8,
        agg_function: str = 'mean',

    ):
        self.agg_function = agg_function
        self.model = CatBoostRegressor(**params)
        self.analogs_num = analogs_num
        self.agg_function = agg_function
        self.cat_features = None

    
    def fit(self, df_train: pd.DataFrame, df_validation: pd.DataFrame):
        train_label = pd.DataFrame()
        train_label['price'] = df_train['a_price'] / df_train['b_price']
        train = df_train.drop(columns=['a_offer_id', 'b_offer_id', 'a_price', 'b_price'])

        self.cat_features = [col for col in train.columns if train[col].dtype == 'object']

        train_pool = Pool(train, 
                          train_label,
                          cat_features=self.cat_features
            )
        
        validation_data = df_validation 
        validation_label = pd.DataFrame()
        validation_label['price'] = validation_data['a_price'] / validation_data['b_price']
        validation = validation_data.drop(columns=['a_offer_id', 'b_offer_id', 'a_price', 'b_price'])

        validation_pool = Pool(validation, 
                       validation_label,
                       cat_features=self.cat_features
                      )
        
        self.model.fit(train_pool,
                       eval_set=validation_pool, 
                        early_stopping_rounds=30,
                        )

        feature_importance = self.model.get_feature_importance(data=train_pool, type='PredictionValuesChange')
        feature_names = train.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        return feature_importance_df


    def predict(self, df_test) -> pd.DataFrame:
        test = df_test.drop(columns=['a_offer_id', 'b_offer_id', 'a_price', 'b_price'])

        test_pool = Pool(test,
                        cat_features=self.cat_features
            )
                            
        df_test['correction_coeff'] = self.model.predict(test_pool)
        df_test['price_corrected'] = df_test['correction_coeff'] * df_test['b_price']
        if self.analogs_num is not None:
            df_test['correction_coeff_diff'] = abs(df_test['correction_coeff'] - 1)
            df_test = df_test.sort_values(by='correction_coeff_diff', ascending=True)
            df_test['b_offer_id_num'] = df_test.groupby(self.OBJECT_ID_NAME).cumcount() + 1
            df_test = df_test[df_test['b_offer_id_num'] <= self.analogs_num]

        return self.predict_mean(df_test)
