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

# модули из репозитория https://github.com/sberbank-ai/no_fire_with_ai_aij2021
import helpers, preprocessing, features_generation, prepare_train
from solution import FEATURES

import warnings
warnings.simplefilter("ignore")
plt.rcParams["figure.figsize"] = (16,8)

