import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from corrections_model import CorrectionsModel

def objective(trial, df):
    train_x, valid_x = train_test_split(df, test_size=0.2)
    valid_x, test_x = train_test_split(df, test_size=0.5)
    
    params = {
        "random_seed": 42,
        "iterations": trial.suggest_int('iterations', 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        'loss_function': 'MAE',
    }
    print(params)

    corr_model = CorrectionsModel(params=params, analogs_num=12)
    corr_model.fit(train_x, valid_x)

    preds = corr_model.predict(test_x)
    rmse = corr_model.calculate_metric(preds)
    
    return rmse

def optimize_catboost(df):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, df), n_trials=30)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    print('Best trial:', study.best_trial.params)
    
    return study.best_trial


