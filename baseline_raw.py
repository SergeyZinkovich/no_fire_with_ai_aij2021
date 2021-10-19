# 1 Cell

import os
import pickle
import cfgrib
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from catboost import Pool, CatBoostClassifier
from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.overpass import overpassQueryBuilder, Overpass
from sklearn.model_selection import train_test_split

# модули из репозитория https://github.com/sberbank-ai/no_fire_with_ai_aij2021
import helpers, preprocessing, features_generation, prepare_train
from solution import FEATURES

import warnings
warnings.simplefilter("ignore")
plt.rcParams["figure.figsize"] = (16, 8)

# 2 Cell

ds = cfgrib.open_datasets('input/ERA5_data/temp_2018.grib')

# 3 Cell

print(len(ds))

# 4 Cell

print(ds[0].indexes['time'].min(), ds[0].indexes['time'].max())

# 6 Cell

ds[0].stl1[200].plot(cmap=plt.cm.coolwarm)

# 7 Cell

# Может понадобиться выполнение команд
# pip uninstall shapely
# pip install shapely --no-binary shapely

ax = plt.axes(projection=ccrs.Robinson())
# ax.coastlines(resolution='10m')
plot = ds[1].d2m[119].plot(cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree(), cbar_kwargs={'shrink': 0.6})
plt.title('Soil temperature on 2018-04-29');

# 8 Cell

times, latitudes, longitudes = preprocessing.parse_dims(ds)

# 9 Cell

print(latitudes)

# 10 Cell

print(longitudes)

# 11 Cell

train_raw = pd.read_csv('input/train_raw.csv', parse_dates=['dt'])
print(train_raw.shape)
print(train_raw.head())

# 12 Cell

print(train_raw.dt.min(), train_raw.dt.max())

# 13 Cell

lat_min = round(latitudes.min(), 1)
lat_max = round(latitudes.max(), 1)

lon_min = round(longitudes.min(), 1)
lon_max = round(longitudes.max(), 1)

print(lat_min, lat_max, lon_min, lon_max)

# 14 Cell

step = 0.2
array_of_lats = np.arange(lat_min, lat_max, step).round(1)
array_of_lons = np.arange(lon_min, lon_max, step).round(1)
print(len(array_of_lats), len(array_of_lons))

# 15 Cell

train = pd.read_csv('input/train.csv', parse_dates=['dt'])
print(train.shape)
train.head()

# 16 Cell

print(train.dt.min(), train.dt.max())

# 17 Cell

sample_test = pd.read_csv('input/sample_test.csv', parse_dates=['dt'])
print(sample_test.shape)
print(sample_test.head())

# 18 Cell

train, val, _, _ = train_test_split(train, train, test_size=0.25, random_state=41)
print(train.shape, val.shape)

# 19 Cell

print(train[['infire_day_1', 'infire_day_2', 'infire_day_3',
             'infire_day_4', 'infire_day_5', 'infire_day_6',
             'infire_day_7', 'infire_day_8']].apply(pd.Series.value_counts))

# 20 Cell

print(val[['infire_day_1', 'infire_day_2', 'infire_day_3',
           'infire_day_4', 'infire_day_5', 'infire_day_6',
           'infire_day_7', 'infire_day_8']].apply(pd.Series.value_counts))

# 23 Cell

cities_df = gpd.read_file('input/city_town_village.geojson')
cities_df = cities_df[['admin_level', 'name', 'population', 'population:date', 'place', 'geometry']]
cities_df = cities_df[cities_df.place != 'city_block'].reset_index(drop=True)
cities_df['lon'] = cities_df['geometry'].x
cities_df['lat'] = cities_df['geometry'].y

cities_df.loc[cities_df.lon < 0, 'lon'] += 360
cities_df.loc[cities_df.population.notna(), 'population'] = cities_df[cities_df.population.notna()] \
    .population.apply(helpers.split_string).str.replace(" ", "").astype(int)
print(cities_df.head())

# 24 Cell

cities_df = helpers.add_edges_polygon(cities_df)
cities_df = cities_df[(cities_df.lon_max <= lon_max) & \
                      (cities_df.lon_min >= lon_min) & \
                      (cities_df.lat_min >= lat_min) & \
                      (cities_df.lat_max <= lat_max)].reset_index(drop=True)

# 25 Cell

cities_df = helpers.get_grid_index(cities_df, array_of_lons, array_of_lats)

# 26 Cell

cities_df.rename(columns={'lon': 'city_lon',
                          'lat': 'city_lat'}, inplace=True)
print(cities_df.head())

# 27 Cell

PATH_TO_ADD_DATA = 'additional_data/'

grib_list = [el.split('.')[0] for el in os.listdir("input/ERA5_data") \
             if el.startswith(("temp", "wind",
                               "evaporation1", "evaporation2",
                               "heat1", "heat2", "vegetation")) and el.endswith(('2020.grib'))]

for file_name in grib_list:
    preprocessing.make_pool_features("input/ERA5_data",
                                     file_name, PATH_TO_ADD_DATA)

# 28 Cell

PATH_TO_ADD_DATA = 'additional_data/'
train = features_generation.add_pooling_features(train, PATH_TO_ADD_DATA, count_lag=3)
val = features_generation.add_pooling_features(val, PATH_TO_ADD_DATA, count_lag=3)

# 29 Cell

train = features_generation.add_cat_date_features(train)
val = features_generation.add_cat_date_features(val)

# 30 Cell

train = features_generation.add_geo_features(train, cities_df)
val = features_generation.add_geo_features(val, cities_df)

# 31 Cell

cat_features = ['month', 'day', 'weekofyear', 'dayofweek', 'place']
cat_features = train[FEATURES].columns.intersection(cat_features)
cat_features = [train[FEATURES].columns.get_loc(feat) for feat in cat_features]
print(cat_features)

# 32 Cell

def get_multiclass_target(df):
    df = df.copy()
    for i in range(8, 0, -1):
        df.loc[df[f'infire_day_{i}'] == 1, 'multiclass'] = i
    df.fillna(0, inplace=True)
    return df.multiclass

# 33 Cell

train_targets = train.iloc[:,11:11+8]
val_targets = val.iloc[:,11:11+8]

train_target_mc = get_multiclass_target(train_targets)
val_target_mc = get_multiclass_target(val_targets)

# 34 Cell

train_dataset_mc = Pool(data=train[FEATURES],
                    label=train_target_mc,
                    cat_features=cat_features)

eval_dataset_mc = Pool(data=val[FEATURES],
                    label=val_target_mc,
                    cat_features=cat_features)
model_mc = CatBoostClassifier(iterations=100, random_seed=8,
                              eval_metric='MultiClass', auto_class_weights="Balanced")
model_mc.fit(train_dataset_mc,
          eval_set=eval_dataset_mc,
          verbose=False)

# 35 Cell

train_targets = (
    train_targets.replace(0, np.nan).fillna(axis=1, method="ffill").fillna(0).astype(int)
)

val_targets = (
    val_targets.replace(0, np.nan).fillna(axis=1, method="ffill").fillna(0).astype(int)
)

# 36 Cell

models = []
for i in range(8):
    train_dataset = Pool(data=train[FEATURES],
                        label=train_targets.iloc[:,i],
                        cat_features=cat_features)

    eval_dataset = Pool(data=val[FEATURES],
                        label=val_targets.iloc[:,i],
                        cat_features=cat_features)
    model = CatBoostClassifier(iterations=100, random_seed=i+1, eval_metric='F1', auto_class_weights="Balanced")
    model.fit(train_dataset,
              eval_set=eval_dataset,
              verbose=False)
    models.append(model)

# 37 Cell

if not os.path.exists("models/"):
    os.mkdir("models/")
for idx, model in enumerate(models):
    path_to_model = f"models/model_{idx + 1}_day.pkl"

    with open(path_to_model, 'wb') as f:
        pickle.dump(model, f)

with open("models/model_mc.pkl", 'wb') as f:
    pickle.dump(model_mc, f)

# 38 Cell

# Bash script
# !zip -r sample_submission.zip *.py models/ metadata.json >/dev/null

# 39 Cell

pred = pd.DataFrame({'infire_day_1': [1, 0, 1, 0, 0],
                     'infire_day_2': [1, 0, 1, 0, 0],
                     'infire_day_3': [1, 0, 1, 0, 0],
                     'infire_day_4': [1, 0, 0, 0, 0],
                     'infire_day_5': [1, 1, 0, 0, 1],
                     'infire_day_6': [1, 0, 1, 0, 1],
                     'infire_day_7': [1, 0, 1, 0, 0],
                     'infire_day_8': [1, 0, 1, 0, 1],
                    })

print(pred)

# 40 Cell

gt = pd.DataFrame({'infire_day_1': [0, 0, 0, 0, 0],
                   'infire_day_2': [0, 0, 0, 0, 0],
                   'infire_day_3': [1, 0, 0, 0, 1],
                   'infire_day_4': [0, 1, 0, 0, 1],
                   'infire_day_5': [0, 0, 0, 0, 1],
                   'infire_day_6': [0, 0, 0, 0, 0],
                   'infire_day_7': [1, 0, 0, 0, 0],
                   'infire_day_8': [0, 0, 0, 0, 0],
                  })

print(gt)

# 41 Cell

helpers.competition_metric(gt, pred)