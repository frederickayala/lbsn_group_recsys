import random
import pandas as pd
import io
import json
from dateutil import parser
from collections import OrderedDict
import time
import csv
import difflib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import sys, traceback
from matplotlib.legend_handler import HandlerLine2D
import graphlab as gl
import argparse
import logging
import ConfigParser
import configparser
from utils import general_utils
import shutil
from termcolor import colored
import matplotlib.pyplot as plt
from experiments import experiment_utils
import math
import powerlaw


def h1_h2(df_checkins):
    results = OrderedDict({})
    N = float(sum(df_checkins.R))
    n = float(len(df_checkins.R))

    #     lat_sign = np.sign(np.average(df_checkins.venue_lat))
    #     long_sign = np.sign(np.average(df_checkins.venue_long))

    #     df_checkins.venue_lat = np.abs(df_checkins.venue_lat)
    #     df_checkins.venue_long = np.abs(df_checkins.venue_long)

    lat_sum = 1 / N * sum(df_checkins.apply(lambda x: x.R * x.venue_lat, axis=1))
    lat_squared_diff = 1 / N * sum(df_checkins.apply(lambda x: pow((x.R * x.venue_lat) - lat_sum, 2), axis=1))

    h1 = 1.06 * pow(n, -.2) * math.sqrt(lat_squared_diff)

    long_sum = 1 / N * sum(df_checkins.apply(lambda x: x.R * x.venue_long, axis=1))
    long_squared_diff = 1 / N * sum(df_checkins.apply(lambda x: pow((x.R * x.venue_long) - long_sum, 2), axis=1))

    h2 = 1.06 * pow(n, -.2) * math.sqrt(long_squared_diff)

    results["h1"] = h1
    results["h2"] = h2
    return pd.Series(results)


def kh(l, li, h1, h2):
    lat = pow(l.venue_lat - li.venue_lat,2) / (2 * pow(h1,2))
    lon = pow(l.venue_long - li.venue_long,2) / (2 * pow(h2,2))
    res = 1 / (2 * math.pi * h1 * h2) * math.exp(-lat-lon)
    assert res > 0
    return res


def fgeo(l, N, df_checkins, h1, h2):
    s = sum(df_checkins.apply(lambda li: li.R * kh(l, li, h1, h2), axis=1))
    res = (1.0/N) * s
    assert res > 0
    return res


def g(fgeo_li):
    return np.power(np.prod(fgeo_li), 1.0/len(fgeo_li)) + .00000001


def fgeo_group(df_checkins):
    df_checkins["fgeo"] = df_checkins.apply(lambda l: fgeo(l, l.N, df_checkins, l.h1, l.h2), axis=1)
    return df_checkins


def h(checkin, alpha):
    return pow(pow(checkin["g"],-1) * checkin["fgeo"], -alpha)


def khh(l, li, h1, h2, hi):
    lat = pow(l.venue_lat - li.venue_lat,2) / (2 * pow(h1,2) * pow(hi,2))
    lon = pow(l.venue_long - li.venue_long,2) / (2 * pow(h2,2) * pow(hi,2))
    return 1 / (2 * math.pi * h1 * h2 * pow(h1,2)) * math.exp(-lat-lon)


def main_fgeo(l, df_checkins):
    N = df_checkins.head(1).N.values[0]
    s = sum(df_checkins.apply(lambda li: li.R * khh(l, li, li.h1, li.h2, li.hi), axis=1))
    return (1.0/N) * s


class Geo:
    """
    This class models the Geo part of GeoSoCa
    """
    def __init__(self, data_dic, using_key, alpha):
        assert using_key in ["user_id", "group_id"], "The key should be either user_id or group_id"

        self.using_key = using_key
        self.alpha = alpha

        self.df_checkins_train = data_dic["train"]
        self.df_checkins_test = data_dic["test"]
        self.df_venues_categories = data_dic["df_venues_categories"]
        self.df_venues = data_dic["df_venues"]
        self.df_venues_content = data_dic["df_venues_content"]
        self.df_checkin_group = data_dic["df_checkin_group"]
        self.df_checkins = data_dic["df_checkins"]
        self.df_all_groups = data_dic["df_all_groups"]
        self.df_users = data_dic["df_users"]

        self.rul = None
        self.df_fgeo = None

    def fit(self):
        # Step 0: Init step
        rul = self.df_checkins_train.groupby([self.using_key, "venue_id"]).count()[["checkin_id"]].reset_index()
        rul = rul.merge(self.df_venues[["venue_id", "venue_lat", "venue_long"]], on="venue_id")
        rul = rul.rename(columns={"checkin_id": "R"})

        df_h1h2 = rul.groupby(self.using_key).apply(h1_h2)
        df_h1h2 = df_h1h2.reset_index().drop_duplicates()
        rul = rul.merge(df_h1h2, on=self.using_key)
        df_n = rul.groupby(self.using_key).sum()[["R"]].reset_index().rename(columns={"R": "N"})
        rul = rul.merge(df_n, on=self.using_key)

        self.rul = rul

        # Step 1:
        df_fgeo = self.rul.query("h1 != 0").groupby(self.using_key).apply(fgeo_group)

        # Step 2
        # Compute G
        df_g = pd.DataFrame(df_fgeo.groupby(self.using_key)["fgeo"].aggregate(lambda x: g(x))).reset_index()
        df_g = df_g.rename(columns={"fgeo": "g"})
        df_fgeo = df_fgeo.merge(df_g, on="user_id")

        # Compute H
        alpha = .5
        df_fgeo["hi"] = df_fgeo.apply(lambda x: h(x, alpha), axis=1)
        self.df_fgeo = df_fgeo

    def score_sample(self, key, df_candidates):
        assert not self.rul is None and not self.df_fgeo is None, "Before scoring use the fit() method."
        # Compute Geo
        df_fgeo_users = self.df_fgeo.set_index(self.using_key)

        df_fgeo_user = df_fgeo_users.loc[key]
        df_candidates["geo"] = df_candidates.apply(lambda l: main_fgeo(l, df_fgeo_user), axis=1)

        return df_candidates