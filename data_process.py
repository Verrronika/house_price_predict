import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional

def is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False

def load_data() -> pd.DataFrame:
    '''
        Загружаем данные и создаем общую таблицу
    '''
    data1 = pd.read_csv('/Users/aksveronika/Desktop/Курсовая/analogs_2023-03-31_2023-04-30.csv')
    data2 = pd.read_csv('/Users/aksveronika/Desktop/Курсовая/analogs_2023-04-30_2023-05-31.csv')
    data3 = pd.read_csv('/Users/aksveronika/Desktop/Курсовая/analogs_2023-05-31_2023-06-30.csv')
    data4 = pd.read_csv('/Users/aksveronika/Desktop/Курсовая/analogs_2023-06-30_2023-07-31.csv')
    data5 = pd.read_csv('/Users/aksveronika/Desktop/Курсовая/analogs_2023-07-31_2023-08-31.csv')
    data6 = pd.read_csv('/Users/aksveronika/Desktop/Курсовая/analogs_2023-08-31_2023-09-30.csv')
    data7 = pd.read_csv('/Users/aksveronika/Desktop/Курсовая/analogs_2023-09-30_2023-10-31.csv')
    data = pd.concat([data1, data2, data3, data4, data5, data6, data7], ignore_index=True)

    # Переименовываем таргет
    data['a_price'] = data['a_target']
    data.drop('a_target', axis=1, inplace=True)

    # Добавляем поле коэффициента цены (для модели корректировок)
    data['price_coeff'] = data['a_price'] / data['b_price']
    
    return data


def process_missing_values(data: pd.DataFrame, info) -> pd.DataFrame:
    '''
        Обрабатываем все признаки на основе предварительного анализа данных -> заполняем пропуски
    '''

    # Создаем таблицу уникальных квартир
    a_columns = data.columns[(data.columns.str.startswith('a_')) & (~data.columns.str.startswith('a_actual_dt'))]
    b_columns = data.columns[(data.columns.str.startswith('b_')) & (~data.columns.str.startswith('b_valid'))]
    analogs = data.sort_values(by='b_valid_from', ascending=False).groupby('b_offer_id').head(1)

    rename_dict = {}
    for column in data.columns:
        rename_dict[column] = column[2:]

    objects = data[a_columns].rename(columns=rename_dict).drop_duplicates(ignore_index=True)
    analogs = analogs[b_columns].rename(columns=rename_dict).drop_duplicates(ignore_index=True)
    flats = pd.concat([objects, analogs[~analogs.offer_id.isin(objects.offer_id)]]).drop_duplicates(ignore_index=True).sort_values('offer_id')

    # Основное заполнение пропусков
    new_feature_names = ['a_actual_dt', 'b_valid_from', 'b_valid_to']
    for feature, metric in info.items():
        if metric == 'delete':
            continue
        elif metric == 'keep':
            new_feature_names.append('a_' + feature)
            new_feature_names.append('b_' + feature)
        elif feature in ('is_apartment_mean'):
            mean_values = flats.groupby('wall_material_fillna_sim_low')[feature].transform(metric)
            flats[feature] = flats[feature].fillna(mean_values)
            new_feature_names.append('a_' + feature)
            new_feature_names.append('b_' + feature)
        elif feature in ('living_area', 'kitchen_area'):
            # feature_name = feature + 'by_total_area'
            mean_coeff = (flats[feature] / flats['area_total']).mean()
            flats[feature] = flats[feature].fillna(flats['area_total'] * mean_coeff)
            new_feature_names.append('a_' + feature)
            new_feature_names.append('b_' + feature)
        elif metric in ('mean', 'median', 'mode'):
            if metric == 'mean':
                feature_metric = flats[feature].mean()
            elif metric == 'median':
                feature_metric = flats[feature].median()
            elif metric == 'mode':
                feature_metric = flats[feature].mode()[0]
            flats[feature] = flats[feature].fillna(feature_metric)
            new_feature_names.append('a_' + feature)
            new_feature_names.append('b_' + feature)
        else:
            feature_metric = metric
            flats[feature] = flats[feature].fillna(feature_metric)
            new_feature_names.append('a_' + feature)
            new_feature_names.append('b_' + feature)
    
    # Меняем исходные данные
    result = data.merge(flats, how='left', left_on='a_offer_id', right_on='offer_id')
    cols = [col for col in data.columns if col.startswith('a_')]
    for col in cols:
        if col[2:] in flats.columns and col in new_feature_names:
            result[col] = result[col].fillna(result[col[2:]])
    result = result[new_feature_names]

    result = result.merge(flats, how='left', left_on='b_offer_id', right_on='offer_id')
    cols = [col for col in data.columns if col.startswith('b_')]
    for col in cols:
        if col[2:] in flats.columns and col in new_feature_names:
            result[col] = result[col].fillna(result[col[2:]])
    result = result[new_feature_names]

    # Распарсить словари 
    for pre in ('a_', 'b_'):
        result[pre + 'kitchen_furniture'] = result[pre + 'amenities'].str.contains('Мебель на кухне|Мебельнакухне').fillna(0).astype(int)
        result[pre + 'furniture_in_rooms'] = result[pre + 'amenities'].str.contains('Мебель в комнатах|Мебельвкомнатах').fillna(0).astype(int)
        result[pre + 'washing_machine'] = result[pre + 'amenities'].str.contains('Стиральная машина|Стиральнаямашина').fillna(0).astype(int)
        result[pre + 'stove'] = result[pre + 'amenities'].str.contains('Плита').fillna(0).astype(int)
        result[pre + 'refrigerator'] = result[pre + 'amenities'].str.contains('Холодильник').fillna(0).astype(int)
        result[pre + 'dishwasher'] = result[pre + 'amenities'].str.contains('Посудомоечная машина|Посудомоечнаямашина').fillna(0).astype(int)
        result[pre + 'TV'] = result[pre + 'amenities'].str.contains('Телевизор').fillna(0).astype(int)
        result[pre + 'air_conditioner'] = result[pre + 'amenities'].str.contains('Кондиционер').fillna(0).astype(int)
        result[pre + 'microwave'] = result[pre + 'amenities'].str.contains('Микроволновая печь|Микроволноваяпечь').fillna(0).astype(int)
        result[pre + 'internet'] = result[pre + 'amenities'].str.contains('Интернет').fillna(0).astype(int)

        result[pre + 'school'] = result[pre + 'infrastructure'].str.contains('Школа').fillna(0).astype(int)
        result[pre + 'kindergarten'] = result[pre + 'infrastructure'].str.contains('Детский сад|Детскийсад').fillna(0).astype(int)
        result[pre + 'shopping_center'] = result[pre + 'infrastructure'].str.contains('Торговый центр|Торговыйцентр').fillna(0).astype(int)
        result[pre + 'park'] = result[pre + 'infrastructure'].str.contains('Парк').fillna(0).astype(int)
        result[pre + 'fitness'] = result[pre + 'amenities'].str.contains('Фитнес').fillna(0).astype(int)
    
    result = result.drop(['a_amenities', 'b_amenities', 'a_infrastructure', 'b_infrastructure'], axis=1)
    return result

def process_outliers(data: pd.DataFrame, max_object_price=None, max_analog_price=None, max_price_diff=None, max_rooms=None, max_ceiling_height_fillna_own=None) -> pd.DataFrame:
    '''
        Обрабатываем выбросы в данных на основе предварительного анализа данных -> создаем флаги выбросов

        max_object_price=172500, max_analog_price=232000, max_price_diff=220100, max_rooms=10, max_ceiling_height_fillna_own=7
    '''
    if max_object_price is not None:
        data = data[data['a_price'] <= max_object_price]

    if max_analog_price is not None:
        data = data[data['b_price'] <= max_analog_price]

    if max_price_diff is not None:
        data = data[abs(data['a_price'] - data['b_price']) <= max_price_diff]

    if max_rooms is not None:
        data = data[data['a_rooms'] <= max_rooms]
        data = data[data['b_rooms'] <= max_rooms]

    if max_ceiling_height_fillna_own is not None:
        data = data[data['b_ceiling_height_fillna_own'] <= max_ceiling_height_fillna_own]
        data = data[data['a_ceiling_height_fillna_own'] <= max_ceiling_height_fillna_own]
        
    return data

def format_features(data: pd.DataFrame) -> pd.DataFrame:
    '''
        Формируем новые признаки на основе предварительного анализа данных
    '''
    data['price_diff'] = abs(data['a_price'] - data['b_price']) # Разница в цене
    data['area_total_diff'] = abs(data['a_area_total'] - data['b_area_total']) # Разница в площади
    data['repair_class_id_diff'] = abs(data['a_repair_class_id'] - data['b_repair_class_id']) # Разница в ремонте
    data['quality_class_id_diff'] = abs(data['a_quality_class_id'] - data['b_quality_class_id']) # Разница в ремонте
    data['quality_diff'] = abs(data['a_quality'] - data['b_quality']) # Разница в ремонте
    return data


def split_data(data: pd.DataFrame) -> pd.DataFrame | pd.DataFrame | pd.DataFrame:
    '''
        Поделим данные на тренировочные, валидационные и тестовые по количеству уникальных объектов
    '''
    objects_sorted_dt = data.sort_values(by='a_actual_dt')['a_offer_id'].drop_duplicates()

    # Определяем точки, на которых нужно разделить данные в отношении 80/10/10
    train_size = int(len(objects_sorted_dt) * 0.8)  
    val_size = int(len(objects_sorted_dt) * 0.1)  

    obejcts_train = objects_sorted_dt[:train_size]
    obejcts_val = objects_sorted_dt[train_size:train_size + val_size]
    obejcts_test = objects_sorted_dt[train_size + val_size:]

    train = pd.merge(data, obejcts_train, on='a_offer_id', how='inner')
    test = pd.merge(data, obejcts_test, on='a_offer_id', how='inner')
    val = pd.merge(data, obejcts_val, on='a_offer_id', how='inner')

    print("Размер train выборки:", len(train))
    print("Размер val выборки:", len(val))
    print("Размер test выборки:", len(test))

    print("Уникальных объектов в train:", len(train['a_offer_id'].drop_duplicates()))
    print("Уникальных объектов в val:", len(val['a_offer_id'].drop_duplicates()))
    print("Уникальных объектов в test:", len(test['a_offer_id'].drop_duplicates()))

    return train, val, test


