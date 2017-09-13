import random
import pandas as pd
import io
import json
from dateutil import parser
from collections import OrderedDict, deque
import time
import csv
import difflib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import sys, traceback
from matplotlib.legend_handler import HandlerLine2D
import multiprocessing
import argparse
import logging
import ConfigParser
import configparser
import math
from termcolor import colored
import graphlab as gl
import pickle
from mapbox import Static
from utils import general_utils
from utils.general_utils import *
from tabulate import tabulate
from geopy.distance import vincenty
import seaborn as sns
import powerlaw
import pickle
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import scipy
from pandas.tools.plotting import parallel_coordinates
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
from joblib import Parallel, delayed
import multiprocessing

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)
matplotlib.rcParams.update({'font.size': 32})
matplotlib.rcParams.update({'legend.fontsize': 20})
matplotlib.rcParams.update({'xtick.labelsize': 32, 'ytick.labelsize': 32, })

sns.set(font_scale=2)

sns.set(color_codes=True)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', multiprocessing.cpu_count())

def compare_preferences(user_id, user_checkins, group_checkins):
    res = {"user_id": user_id}
    top_user = user_checkins.groupby("second_category").count().sort_values(
        ["checkin_id"], ascending=False)[["checkin_id"]].reset_index()
    top_user["user_ranking"] = top_user.checkin_id.rank(ascending=False, method="first")
    top_user["user_ranking"] = top_user["user_ranking"].astype(int)

    top_group = group_checkins.groupby("second_category").count().sort_values(
        ["checkin_id"], ascending=False)[["checkin_id"]].reset_index()
    top_group["group_ranking"] = top_group.checkin_id.rank(ascending=False, method="first")
    top_group["group_ranking"] = top_group["group_ranking"].astype(int)

    top_merged = top_user.merge(top_group[["second_category", "group_ranking"]], on="second_category", how="left")

    # Assign a low rank to the categories that does not exists in the groups
    top_merged.group_ranking = top_merged.group_ranking.fillna(100)
    res["top_merged"] = top_merged
    kendelltau = scipy.stats.kendalltau(top_merged.user_ranking, top_merged.group_ranking)
    res["kendelltau"] = kendelltau[0]
    res["kendelltau_pvalue"] = kendelltau[1]
    return res


def get_clusters(checkins, eps, min_samples):
    res = scipy.spatial.distance.pdist(checkins[["venue_lat", "venue_long"]].values,
                                       lambda p1, p2: vincenty(p1, p2).kilometers)
    res = scipy.spatial.distance.squareform(res)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(res)
    checkins["cluster"] = db.labels_

    df_clusters = pd.DataFrame(db.labels_, columns=["cluster"])
    df_clusters["cluster_checkins"] = 1
    df_clusters = df_clusters.groupby("cluster").sum().reset_index()

    checkins = checkins.merge(df_clusters, on="cluster")

    cluster = checkins.groupby("cluster").apply(lambda x: get_representative_point(x)).reset_index()
    checkins_cluster = checkins[["venue_lat", "venue_long", "cluster", "cluster_checkins"]]
    checkins_cluster = checkins_cluster.merge(cluster, on="cluster")
    checkins_clean = checkins_cluster[
        ["venue_lat", "venue_long", "cluster", "cluster_checkins"]].drop_duplicates()
    checkins_centroids = checkins_cluster[
        ["cluster", "cluster_checkins", "centroid_lat", "centroid_long"]].drop_duplicates()
    return {"checkins_clean": checkins_clean,
            "checkins_centroids": checkins_centroids,
            "checkins": checkins}


def avg_distance_to_group(df_user_groups, checkins, user_id, eps, min_samples):
    user_cluster = get_clusters(checkins, eps, min_samples)

    df_user_groups = df_user_groups[["group_id", "cluster", "cluster_checkins",
                                     "centroid_lat", "centroid_long"]].drop_duplicates()

    if len(df_user_groups) > 0:
        # Compute the distance from user centroid to group centroids
        lines = deque()
        for i, v in user_cluster["checkins_centroids"].iterrows():
            for j, w in df_user_groups.iterrows():
                l = OrderedDict({})
                distance = vincenty((v["centroid_lat"], v["centroid_long"]),
                                    (w["centroid_lat"], w["centroid_long"])).kilometers
                l["distance"] = distance
                l["user_cluster_weight"] = v["cluster_checkins"]
                l["group_cluster_weight"] = w["cluster_checkins"]
                l["total_weight"] = l["user_cluster_weight"] * l["group_cluster_weight"]
                l["weighted_distance"] = distance * l["total_weight"]
                line_string = "LINESTRING(%s %s, %s %s)" % (
                    v["centroid_long"], v["centroid_lat"], w["centroid_long"], w["centroid_lat"])
                l["wkt"] = line_string
                lines.append(l)

        df_lines = pd.DataFrame(list(lines))
        return {"distance_average": df_lines.distance.sum() / (user_cluster["checkins_centroids"].shape[0]
                                                               * df_user_groups.shape[0]),
                "distance_weighted_average": df_lines.weighted_distance.sum() / df_lines.total_weight.sum(),
                "user_cluster": user_cluster,
                "user_id": user_id}
    else:
        return None


def get_distance(x):
    return math.ceil(vincenty((x.prev_venue_lat, x.prev_venue_long),(x.venue_lat, x.venue_long)).kilometers)


def get_best_fit(df_dist, best, R, p):
    df_diff = df_dist.query("dist == @best and dist != best").sort_values("R")
    if df_diff.shape[0]:
        best = df_diff.head(1).best.values[0]
        R = df_diff.head(1).R.values[0]
        p = df_diff.head(1).p.values[0]
        return get_best_fit(df_dist, best, R, p)
    else:
        return (best, R, p)


def get_representative_point(checkins):
    res = OrderedDict({})
    all_points = []
    for i,v in checkins.iterrows():
        all_points.append((v.venue_long, v.venue_lat))
    points = MultiPoint(all_points)
    rp = points.centroid
    res["centroid_long"] = rp.x
    res["centroid_lat"] = rp.y
    return pd.Series(res)


class Analysis:
    def __init__(self, data_dic, settings, logger):
        self.data_dic = data_dic
        self.data_dic_all = None
        self.settings = settings
        self.logger = logger
        self.analysis_folder = self.settings.get("settings", "analysis_folder")

    def get_summary(self):
        write_running_log("get_summary", self.logger)
        all_summary = []
        for ds in self.data_dic.keys():
            datasets = self.data_dic[ds]
            df_users = datasets["df_users"]
            df_checkins = datasets["df_checkins"]
            df_checkin_group = datasets["df_checkin_group"]
            df_checkins_with = datasets["df_checkins_with"]
            df_venues = datasets["df_venues"]
            df_venues_categories = datasets["df_venues_categories"]
            df_venues_content = datasets["df_venues_content"]
            df_group_user_checkins = datasets["df_group_user_checkins"]

            dataset_summary = OrderedDict({})
            dataset_summary["scope"] = ds
            dataset_summary["users"] = len(df_users.user_id.unique())
            dataset_summary["total_venues"] = len(df_venues.venue_id.unique())
            dataset_summary["categories"] = len(df_venues_categories.category_id.unique())
            dataset_summary["total_checkins"] = len(df_checkins.checkin_id.unique())
            dataset_summary["groups"] = len(df_checkin_group.group_id.unique())
            dataset_summary["group_checkins"] = len(df_checkins_with.checkin_id.unique())
            dataset_summary["group_venues"] = len(df_checkin_group.venue_id.unique())
            all_summary.append(dataset_summary)
        df_summary = pd.DataFrame(all_summary)
        df_summary = df_summary[["scope", "users", "total_venues", "categories",
             "groups", "total_checkins", "group_checkins", "group_venues"]]
        save2csv(df_summary, os.path.join(self.analysis_folder, "summary.csv"))
        self.format_summary()
        return df_summary

    def format_summary(self):
        df_summary = pd.read_csv(os.path.join(self.analysis_folder, "summary.csv"))
        df_summary = df_summary.sort_values(["total_checkins", "group_checkins"], ascending=False)
        df_summary["scope"] = df_summary["scope"].apply(lambda x: x.replace("_depth_3", "").replace("_", " ").title())
        for c in [c for c in df_summary.columns if "scope" not in c]:
            df_summary[c] = df_summary[c].apply(lambda x: "{:,}".format(x))
        df_summary.columns = [c.replace("_", " ").title() for c in df_summary.columns]
        print tabulate(df_summary, headers='keys', tablefmt='psql')
        return df_summary

    def analyze_inter_checkins(self):
        write_running_log("analyze_inter_checkins", self.logger)
        df_users_all = None
        df_checkins_all = None
        df_checkin_group_all = None
        df_checkins_with_all = None
        df_venues_all = None
        df_venues_categories_all = None

        for ds in self.data_dic.keys():
            datasets = self.data_dic[ds]
            df_users = datasets["df_users"]
            df_checkins = datasets["df_checkins"]
            df_checkin_group = datasets["df_checkin_group"]
            df_checkins_with = datasets["df_checkins_with"]
            df_venues = datasets["df_venues"]
            df_venues_categories = datasets["df_venues_categories"]

            df_users_all = pd.concat([df_users, df_users_all]) if not df_users_all is None else df_users
            df_checkins_all = pd.concat([df_checkins, df_checkins_all]) if not df_checkins_all is None else df_checkins
            df_checkin_group_all = pd.concat(
                [df_checkin_group, df_checkin_group_all]) if not df_checkin_group_all is None else df_checkin_group
            df_checkins_with_all = pd.concat(
                [df_checkins_with, df_checkins_with_all]) if not df_checkins_with_all is None else df_checkins_with
            df_venues_all = pd.concat([df_venues, df_venues_all]) if not df_venues_all is None else df_venues
            df_venues_categories_all = pd.concat([df_venues_categories,
                                                  df_venues_categories_all]) if not df_venues_categories_all is None else df_venues_categories

        df_checkins_all = df_checkins_all.drop_duplicates()
        df_venues_all = df_venues_all.drop_duplicates()

        # Use the latest check-ins information
        df_checkins_all = df_checkins_all.groupby("checkin_id").last().reset_index()

        # Use the latest venues information
        df_venues_all = df_venues_all.sort_values(["venue_id", "venue_checkinsCount"], ascending=True).groupby(
            "venue_id").last().reset_index()

        # Check if the inter_distance files exist already otherwise load it.
        file_user_inter_checkin = os.path.join(self.analysis_folder, "user_inter_checkin.csv")
        file_user_inter_checkin_clean = os.path.join(self.analysis_folder, "user_inter_checkin_clean.csv")
        file_group_inter_checkin = os.path.join(self.analysis_folder, "group_inter_checkin.csv")
        file_group_inter_checkin_clean = os.path.join(self.analysis_folder, "group_inter_checkin_clean.csv")

        if os.path.exists(file_user_inter_checkin) and os.path.exists(file_user_inter_checkin_clean) and \
                os.path.exists(file_group_inter_checkin) and os.path.exists(file_group_inter_checkin_clean):
            write_log("***** Using existing inter-distance and inter-time files, erase them to re-compute",
                      self.logger, True, False, print_color="green")
            user_inter_checkin = pd.read_csv(file_user_inter_checkin)
            user_inter_checkin_clean = pd.read_csv(file_user_inter_checkin_clean)
            group_inter_checkin = pd.read_csv(file_group_inter_checkin)
            group_inter_checkin_clean = pd.read_csv(file_group_inter_checkin_clean)
        else:
            write_log("***** Computing inter-distance and inter-time files",
                      self.logger, True, False, print_color="green")
            #Get the configuration for filtering
            filter_user_q_time = self.settings.getint("settings", "filter_user_q_time")
            filter_user_q_distance = self.settings.getint("settings", "filter_user_q_distance")
            filter_group_q_time = self.settings.getint("settings", "filter_group_q_time")
            filter_group_q_distance = self.settings.getint("settings", "filter_group_q_distance")
            filter_group_size = self.settings.getint("settings", "filter_group_size")

            # Analyze for users
            df_checkins_all_user = df_checkins_all[~df_checkins_all.checkin_id.isin(df_checkin_group_all.checkin_id)]
            user_inter_checkin = self.get_inter_df(False, df_checkins_all_user, df_venues_all)

            df_distance_median = user_inter_checkin.groupby("user_id").mean()[["inter_distance"]].dropna().reset_index()
            df_distance_median = df_distance_median.rename(columns={"inter_distance": "inter_distance_mean"})
            df_distance_median["q_inter_distance"] = pd.qcut(df_distance_median.inter_distance_mean, 4, labels=False)
            user_inter_checkin_clean = user_inter_checkin.merge(df_distance_median, on="user_id")
            user_inter_checkin_clean = user_inter_checkin_clean.query("q_inter_distance < @filter_user_q_distance")

            df_time_median = user_inter_checkin.groupby("user_id").mean()[["inter_time"]].dropna().reset_index()
            df_time_median = df_time_median.rename(columns={"inter_time": "inter_time_mean"})
            df_time_median["q_inter_time"] = pd.qcut(df_time_median.inter_time_mean, 4, labels=False)
            user_inter_checkin_clean = user_inter_checkin_clean.merge(df_time_median, on="user_id")
            user_inter_checkin_clean = user_inter_checkin_clean.query("q_inter_time < @filter_user_q_time")

            save2csv(user_inter_checkin, os.path.join(self.analysis_folder, "user_inter_checkin.csv"))
            save2csv(user_inter_checkin_clean, os.path.join(self.analysis_folder, "user_inter_checkin_clean.csv"))

            # Analyze for groups
            df_checkin_group_all["group_size"] = df_checkin_group_all["group_id"].apply(lambda x: len(x.split(",")))
            save2csv(df_checkin_group_all[["group_id", "group_size"]].drop_duplicates(),
                     os.path.join(self.analysis_folder, "group_sizes.csv"))

            group_ids = df_checkin_group_all.query("group_size <= @filter_group_size").checkin_id
            df_checkins_all_group = df_checkins_all[df_checkins_all.checkin_id.isin(group_ids)]
            df_checkins_all_group = df_checkins_all_group.merge(df_checkin_group_all[["checkin_id", "group_id"]],
                                                                on="checkin_id")
            group_inter_checkin = self.get_inter_df(True, df_checkins_all_group, df_venues_all)
            df_distance_median = group_inter_checkin.groupby("group_id").mean()[["inter_distance"]].dropna().reset_index()
            df_distance_median = df_distance_median.rename(columns={"inter_distance": "inter_distance_mean"})
            df_distance_median["q_inter_distance"] = pd.qcut(df_distance_median.inter_distance_mean, 4, labels=False)
            group_inter_checkin_clean = group_inter_checkin.merge(df_distance_median, on="group_id")
            group_inter_checkin_clean = group_inter_checkin_clean.query("q_inter_distance < @filter_group_q_distance")

            df_time_median = group_inter_checkin.groupby("group_id").mean()[["inter_time"]].dropna().reset_index()
            df_time_median = df_time_median.rename(columns={"inter_time": "inter_time_mean"})
            df_time_median["q_inter_time"] = pd.qcut(df_time_median.inter_time_mean, 4, labels=False)
            group_inter_checkin_clean = group_inter_checkin_clean.merge(df_time_median, on="group_id")
            group_inter_checkin_clean = group_inter_checkin_clean.query("q_inter_time < @filter_group_q_time")

            save2csv(group_inter_checkin, os.path.join(self.analysis_folder, "group_inter_checkin.csv"))
            save2csv(group_inter_checkin_clean, os.path.join(self.analysis_folder, "group_inter_checkin_clean.csv"))

        self.plot_hexbin(user_inter_checkin_clean, False)
        self.plot_hexbin(group_inter_checkin_clean, True)

        print "Descriptive statistics for inter analysis of users"
        print tabulate(user_inter_checkin.describe()[["inter_time", "inter_distance"]].transpose(),
                       headers='keys', tablefmt='psql')

        print "Descriptive statistics for inter analysis of users (clean)"
        print tabulate(user_inter_checkin_clean.describe()[["inter_time", "inter_distance"]].transpose(),
                       headers='keys', tablefmt='psql')

        print "Descriptive statistics for inter analysis of groups"
        print tabulate(group_inter_checkin.describe()[["inter_time", "inter_distance"]].transpose(),
                       headers='keys', tablefmt='psql')

        print "Descriptive statistics for inter analysis of groups (clean)"
        print tabulate(group_inter_checkin_clean.describe()[["inter_time", "inter_distance"]].transpose(),
                       headers='keys', tablefmt='psql')

        # Clean the datasets
        valid_checkins = set(user_inter_checkin_clean.checkin_id).union(group_inter_checkin_clean.checkin_id)
        df_checkins_all = df_checkins_all[df_checkins_all.checkin_id.isin(valid_checkins)]
        df_checkin_group_all = df_checkin_group_all[
            df_checkin_group_all.checkin_id.isin(group_inter_checkin_clean.checkin_id)]
        df_checkins_with_all = df_checkins_with_all[
            df_checkins_with_all.checkin_id.isin(group_inter_checkin_clean.checkin_id)]

        # Store the clean datasets
        clean_folder = os.path.join(self.analysis_folder, "clean_dataset")
        if not os.path.exists(clean_folder):
            os.mkdir(clean_folder)
            write_running_log("storing_clean_datasets", self.logger)

            save2csv(df_users_all, os.path.join(clean_folder, "users_clean.csv"))
            save2csv(df_checkins_all, os.path.join(clean_folder, "checkins_clean.csv"))
            save2csv(df_checkin_group_all, os.path.join(clean_folder, "checkin_group_clean.csv"))
            save2csv(df_checkins_with_all, os.path.join(clean_folder, "checkins_with_clean.csv"))
            save2csv(df_venues_all, os.path.join(clean_folder, "venues_clean.csv"))
            save2csv(df_venues_categories_all, os.path.join(clean_folder, "venues_categories_clean.csv"))
        else:
            write_log("***** Skipping storing the clean dataset. Erase " + clean_folder + " to generate them again.",
                      self.logger, True, False, print_color="green")

        # Get the group user relations
        write_running_log("groups_user_relations", self.logger)
        df_all_groups = deque()
        for group in df_checkin_group_all.group_id.unique():
            l_res = deque()
            for user in group.split(","):
                res = OrderedDict({})
                res["group_id"] = group
                res["user_id"] = int(user)
                l_res.append(res)
            df_all_groups.append(pd.DataFrame(list(l_res)))
        df_all_groups = pd.concat(df_all_groups)

        self.data_dic_all = {"user_inter_checkin": user_inter_checkin,
                "user_inter_checkin_clean": user_inter_checkin_clean,
                "group_inter_checkin": group_inter_checkin,
                "group_inter_checkin_clean": group_inter_checkin_clean,
                "users_all": df_users_all,
                "groups_all": df_all_groups,
                "checkins_all": df_checkins_all,
                "checkin_group_all": df_checkin_group_all,
                "checkins_with_all": df_checkins_with_all,
                "venues_all": df_venues_all,
                "venues_categories_all": df_venues_categories_all}

        return self.data_dic_all

    # def analyze_distribution(self, data):

    def get_inter_df(self, for_group, df_checkins_all, df_venues_all):
        for_id = "group_id" if for_group else "user_id"

        df_checkins_all = df_checkins_all.merge(df_venues_all, on="venue_id")
        df_checkins_all = df_checkins_all.sort_values([for_id, "checkin_created_at"], ascending=True)

        df_checkins_all["prev_for_id"] = df_checkins_all[for_id].shift(1)
        df_checkins_all["prev_checkin_at"] = df_checkins_all["checkin_created_at"].shift(1)
        df_checkins_all["prev_venue_lat"] = df_checkins_all["venue_lat"].shift(1)
        df_checkins_all["prev_venue_long"] = df_checkins_all["venue_long"].shift(1)

        df_checkins_all = df_checkins_all[~ df_checkins_all.prev_for_id.isnull()]

        if for_group:
            df_inter_checkin = df_checkins_all.query('group_id == prev_for_id')
        else:
            df_checkins_all.prev_user_id = df_checkins_all.prev_for_id.astype(int)
            df_inter_checkin = df_checkins_all.query('user_id == prev_for_id')

        df_inter_checkin = df_inter_checkin.sort_values([for_id, "checkin_created_at"], ascending=True)

        df_inter_checkin["inter_time"] = df_inter_checkin["checkin_created_at"] - df_inter_checkin["prev_checkin_at"]
        df_inter_checkin["inter_time"] = df_inter_checkin["inter_time"].apply(lambda x: math.ceil(x / 3600))
        df_inter_checkin["inter_distance"] = df_inter_checkin.apply(lambda x: get_distance(x), axis=1)

        return df_inter_checkin

    def plot_hexbin(self, df_inter_checkin, for_group):
        matplotlib.rcParams.update({'font.size': 32})
        matplotlib.rcParams.update({'legend.fontsize': 20})
        matplotlib.rcParams.update({'xtick.labelsize': 32, 'ytick.labelsize': 32, })
        # Get the configuration for filtering
        if for_group:
            xmax = 10
            ymax = 200
            gridsize = 15
            for_id = "groups"
            use_cmap = plt.cm.RdYlGn_r
            df_checkin_id = df_inter_checkin.groupby("group_id").count()[["checkin_id"]].reset_index()
            df_inter_time = df_inter_checkin.groupby("group_id").mean()[["inter_time"]].reset_index()
            df_inter_distance = df_inter_checkin.groupby("group_id").mean()[["inter_distance"]].reset_index()

            df_plot = df_checkin_id.merge(df_inter_time, on="group_id")
            df_plot = df_plot.merge(df_inter_distance, on="group_id")
        else:
            xmax = 20
            ymax = 200
            gridsize = 60
            for_id = "users"
            use_cmap = plt.cm.RdYlGn_r
            df_checkin_id = df_inter_checkin.groupby("user_id").count()[["checkin_id"]].reset_index()
            df_inter_time = df_inter_checkin.groupby("user_id").mean()[["inter_time"]].reset_index()
            df_inter_distance = df_inter_checkin.groupby("user_id").mean()[["inter_distance"]].reset_index()

            df_plot = df_checkin_id.merge(df_inter_time, on="user_id")
            df_plot = df_plot.merge(df_inter_distance, on="user_id")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        img = df_plot.plot.hexbin(ylim=(0, ymax),
                                  xlim=(0, xmax),
                                  y="inter_time",
                                  x="inter_distance",
                                  C="checkin_id",
                                  gridsize=gridsize,
                                  ax=ax,
                                  reduce_C_function=np.median,
                                  cmap=use_cmap)
        # ax.set_title(for_id.title() + " Check-ins Median", y=1.2)
        ax.set_ylabel("(hrs) between check-ins")
        ax.set_xlabel("(kms) between check-ins")
        ax.set_axis_bgcolor('white')
        general_utils.save_plot(fig, os.path.join(self.analysis_folder, "inter_distance_time_hexbin_median" + for_id))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        img = df_plot.plot.hexbin(ylim=(0, ymax),
                                  xlim=(0, xmax),
                                  y="inter_time",
                                  x="inter_distance",
                                  C="checkin_id",
                                  gridsize=gridsize,
                                  ax=ax,
                                  reduce_C_function=np.mean,
                                  cmap=use_cmap)
        # ax.set_title(for_id.title() + " Check-ins Mean", y=1.2)
        ax.set_ylabel("(hrs) between check-ins")
        ax.set_xlabel("(kms) between check-ins")
        ax.set_axis_bgcolor('white')
        general_utils.save_plot(fig, os.path.join(self.analysis_folder, "inter_distance_time_hexbin_mean" + for_id))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        img = df_plot.plot.hexbin(ylim=(0, ymax),
                                  xlim=(0, xmax),
                                  y="inter_time",
                                  x="inter_distance",
                                  C="checkin_id",
                                  gridsize=gridsize,
                                  ax=ax,
                                  reduce_C_function=np.sum,
                                  cmap=use_cmap)
        # ax.set_title(for_id.title() + " Check-ins Sum", y=1.2)
        ax.set_ylabel("(hrs) between check-ins")
        ax.set_xlabel("(kms) between check-ins")
        ax.set_axis_bgcolor('white')
        general_utils.save_plot(fig, os.path.join(self.analysis_folder, "inter_distance_time_hexbin_sum" + for_id))
        return fig

    def analyze_distribution(self, data, name, xlabel, xmin=None, xmax=None):
        matplotlib.rcParams.update({'font.size': 32})
        matplotlib.rcParams.update({'legend.fontsize': 20})
        matplotlib.rcParams.update({'xtick.labelsize': 32, 'ytick.labelsize': 32, })

        dist_folder = os.path.join(self.analysis_folder, "distribution_" + name)
        if not os.path.exists(dist_folder):
            os.mkdir(dist_folder)

        plt.clf()
        if not (xmax is None and xmin is None):
            fit = powerlaw.Fit(data, xmax=xmax, xmin=xmin)
        else:
            fit = powerlaw.Fit(data)

        # Plotting pdf
        fig_all = fit.plot_pdf(color="b", linewidth=2, label="Empirical Data")

        # ax1in = inset_axes(fig_all, width="30%", height="30%", loc=3)
        # ax1in.hist(data, normed=True, color='b')
        # ax1in.set_xticks([])
        # ax1in.set_yticks([])

        fit.power_law.plot_pdf(linewidth=1, ax=fig_all, color='r', linestyle="--", label="Power Law")
        fit.truncated_power_law.plot_pdf(linewidth=1, ax=fig_all, color='r', linestyle="-", label="Truncated Power Law")
        fit.lognormal.plot_pdf(linewidth=1, ax=fig_all, color='g', linestyle="--", label="Log-Normal")
        fit.stretched_exponential.plot_pdf(linewidth=1, ax=fig_all, color='purple', linestyle=":",
                                           label="Stretched Exponential")
        fig_all.set_ylabel(u"p(X)")
        fig_all.set_xlabel(xlabel)
        handles, labels = fig_all.get_legend_handles_labels()
        leg = fig_all.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.35, 0))
        leg.draw_frame(False)
        general_utils.save_plot(fig_all.get_figure(), os.path.join(dist_folder, "distribution_comparison_pdf_" + name))
        plt.clf()

        # Plotting cdf
        fig_all = fit.plot_cdf(color="b", linewidth=2, label="Empirical Data")

        # ax1in = inset_axes(fig_all, width="30%", height="30%", loc=3)
        # ax1in.hist(data, normed=True, color='b')
        # ax1in.set_xticks([])
        # ax1in.set_yticks([])

        fit.power_law.plot_cdf(linewidth=1, ax=fig_all, color='r', linestyle="--", label="Power Law")
        fit.truncated_power_law.plot_cdf(linewidth=1, ax=fig_all, color='r', linestyle="-", label="Truncated Power Law")
        fit.lognormal.plot_cdf(linewidth=1, ax=fig_all, color='g', linestyle="--", label="Log-Normal")
        fit.stretched_exponential.plot_cdf(linewidth=1, ax=fig_all, color='purple', linestyle=":",
                                           label="Stretched Exponential")
        fig_all.set_ylabel(u"p(X>x)")
        fig_all.set_xlabel(xlabel)
        handles, labels = fig_all.get_legend_handles_labels()
        leg = fig_all.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.35, 0))
        leg.draw_frame(False)
        general_utils.save_plot(fig_all.get_figure(), os.path.join(dist_folder, "distribution_comparison_cdf_" + name))
        plt.clf()

        # Compare the distributions
        all_comparisons = []
        for dist in ["lognormal", "truncated_power_law", "power_law", "stretched_exponential"]:
            for k in [k for k in fit.supported_distributions.keys() if k != dist]:
                result = OrderedDict({})
                result["name"] = name
                result["dist"] = dist
                R, p = fit.distribution_compare(dist, k, normalized_ratio=True)
                result["R"] = R
                result["p"] = p
                print R, p
                if R > 0:
                    print dist + " vs " + k + " : " + dist
                    result["best"] = dist
                else:
                    print dist + " vs " + k + ": " + k
                    result["best"] = k
                all_comparisons.append(result)

            fig_dist = fit.plot_pdf(color="b", linewidth=2, label="Empirical Data")

            # ax1in = inset_axes(fig_dist, width="30%", height="30%", loc=3)
            # ax1in.hist(data, normed=True, color='b')
            # ax1in.set_xticks([])
            # ax1in.set_yticks([])
            if dist == "lognormal":
                fit.lognormal.plot_pdf(linewidth=1, ax=fig_dist, color='r', linestyle="--",
                                       label="Log Normal")
            if dist == "truncated_power_law":
                fit.truncated_power_law.plot_pdf(linewidth=1, ax=fig_dist, color='r', linestyle="--",
                                                 label="Truncated Power Law")
            if dist == "power_law":
                fit.power_law.plot_pdf(linewidth=1, ax=fig_dist, color='g', linestyle="--", label="Power Law")
            if dist == "stretched_exponential":
                fit.stretched_exponential.plot_pdf(linewidth=1, ax=fig_dist, color='g', linestyle="--",
                                                   label="Stretched Exponential")

            fig_dist.set_ylabel(u"p(X)")
            fig_dist.set_xlabel(xlabel)
            handles, labels = fig_dist.get_legend_handles_labels()
            leg = fig_dist.legend(handles, labels, loc='lower center', bbox_to_anchor=(.6, 0))
            leg.draw_frame(False)
            general_utils.save_plot(fig_dist.get_figure(), os.path.join(dist_folder,"distribution_" + name + "_" + dist))
            plt.clf()

        df_distributions = pd.DataFrame(all_comparisons)
        save2csv(df_distributions, os.path.join(dist_folder, "distribution_comparison.csv"))

        # Store the fit distribution
        with open(os.path.join(dist_folder, "distribution_" + name + ".pickle"), 'wb') as handle:
            pickle.dump(fit, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return {"fig_all":fig_all, "fit":fit, "distributions_comparison": df_distributions}

    def compare_preferences_per_user(self):
        write_running_log("compare_preferences_per_user", self.logger)

        preference_folder = os.path.join(self.analysis_folder, "user_preferences")
        user_preference_file = os.path.join(preference_folder, "user_vs_group.csv")
        user_preference_merged_file = os.path.join(preference_folder, "user_vs_group_top_merged.csv")

        if not os.path.exists(preference_folder):
            os.mkdir(preference_folder)

        if not os.path.exists(user_preference_file):
            df_checkins_all = self.data_dic_all["checkins_all"]
            df_checkins_group = self.data_dic_all["checkin_group_all"]
            df_venues_all = self.data_dic_all["venues_all"]
            df_groups_all = self.data_dic_all["groups_all"]

            df_checkins_venues = df_checkins_all.merge(df_venues_all, on="venue_id")
            df_checkins_venues = df_checkins_venues.merge(df_checkins_group[["checkin_id", "group_id"]], on="checkin_id",
                                                          how="left")

            df_checkins_venues = df_checkins_venues.groupby("checkin_id").last().reset_index()

            user_checkins = df_checkins_venues[df_checkins_venues.group_id.isnull()]
            group_checkins = df_checkins_venues[~df_checkins_venues.group_id.isnull()]

            df_users_checkins = user_checkins[user_checkins.user_id.isin(df_groups_all.user_id.unique())].groupby("user_id")

            parallel_results = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=5)(
                delayed(compare_preferences)(user_id, user_checkins, group_checkins[
                    group_checkins.group_id.isin(df_groups_all.query("user_id == @user_id").group_id)])
                for user_id, user_checkins in df_users_checkins)

            df_kendelltau = deque()
            df_all_tops = []
            for r in parallel_results:
                res = OrderedDict({})
                res["user_id"] = r["user_id"]
                res["kendelltau"] = r["kendelltau"]
                res["kendelltau_pvalue"] = r["kendelltau_pvalue"]
                top_merged = r["top_merged"]
                top_merged["user_id"] = r["user_id"]
                df_all_tops.append(top_merged)
                df_kendelltau.append(res)
            df_kendelltau = pd.DataFrame(list(df_kendelltau))
            df_all_tops = pd.concat(df_all_tops)
            save2csv(df_kendelltau, user_preference_file)
            save2csv(df_all_tops, user_preference_merged_file)
        else:
            write_log("***** Using existing existing user vs group preference file. Erase " + user_preference_file +
                      " to recompute",
                      self.logger, True, False, print_color="green")
            df_kendelltau = pd.read_csv(user_preference_file)

        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        img = sns.kdeplot(df_kendelltau.kendelltau.dropna(), shade=True, ax=ax)
        # sns.plt.title("Average Distance to Group - Kernel Density Estimation")
        ax.set_xlabel("Kendall Tau", fontsize="x-large")
        ax.set_ylabel("Density", fontsize="x-large")
        ax.set_axis_bgcolor('white')
        ax.legend().set_visible(False)
        ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

        for label in ticklabels:
            label.set_color('black')
            label.set_fontsize('large')

        general_utils.save_plot(fig, os.path.join(preference_folder, "user_vs_group"))

        return df_kendelltau

    def compare_preferences_per_city(self):
        write_running_log("compare_preferences_per_city", self.logger)

        preference_folder = os.path.join(self.analysis_folder, "city_preferences")
        if not os.path.exists(preference_folder):
            os.mkdir(preference_folder)

        cities_preferences = []

        df_checkins_all = self.data_dic_all["checkins_all"]
        df_venues_all = self.data_dic_all["venues_all"]

        df_checkins_venues = df_checkins_all.merge(df_venues_all, on="venue_id")
        df_cities = df_checkins_venues.groupby(["fixed_city","fixed_geonameid"]).count()[["checkin_id"]].reset_index()
        df_cities = df_cities.sort_values("checkin_id", ascending=False).head(10)

        for i, v in df_cities.iterrows():
            fixed_city = v["fixed_city"]
            res = OrderedDict({})
            res["fixed_geonameid"] = v["fixed_geonameid"]

            df_user_inter_checkin = self.data_dic_all["user_inter_checkin"].query("fixed_city == @fixed_city")
            df_group_inter_checkin = self.data_dic_all["group_inter_checkin"].query("fixed_city == @fixed_city")

            top_user = df_user_inter_checkin
            top_user = top_user.groupby("second_category").count().sort_values("checkin_id", ascending=False)
            top_user = top_user[["checkin_id"]].reset_index()
            top_user["user_ranking"] = top_user.checkin_id.rank(ascending=False)
            top_user["user_ranking"] = top_user["user_ranking"].astype(int)

            top_group = df_group_inter_checkin
            top_group = top_group.groupby("second_category").count().sort_values("checkin_id", ascending=False)
            top_group = top_group[["checkin_id"]].reset_index()
            top_group["group_ranking"] = top_group.checkin_id.rank(ascending=False)
            top_group["group_ranking"] = top_group["group_ranking"].astype(int)

            top_user = top_user.merge(top_group[["second_category", "group_ranking"]], on="second_category")
            top_user = top_user[["second_category", "user_ranking", "group_ranking"]]
            save2csv(top_user, os.path.join(preference_folder, str(v["fixed_geonameid"]) + "_user_vs_group.csv"))

            top_group = top_group.merge(top_user[["second_category", "user_ranking"]], on="second_category")
            top_group = top_group[["second_category", "group_ranking", "user_ranking"]]
            save2csv(top_group, os.path.join(preference_folder, str(v["fixed_geonameid"]) + "_group_vs_user.csv"))

            fig = plt.figure()
            fig.gca().invert_yaxis()
            ax = fig.add_subplot(111)
            df_plot = top_user.query("user_ranking <= 50")[["user_ranking", "group_ranking", "second_category"]]
            df_plot = df_plot.rename(columns={"user_ranking": "User Ranking", "group_ranking": "Group Ranking"})
            img = parallel_coordinates(df_plot, 'second_category', colormap='Paired', ax=ax)
            ylim = ax.get_ylim()
            new_ylim = (ylim[0] - 1, 1)
            ax.set_ylim(new_ylim)
            ticks = ax.get_yticks()
            ticks.put(0, 1)
            ax.set_ylabel("Top")
            ax.set_yticks(ticks)
            ax.set_axis_bgcolor('white')
            ax.legend(loc='lower center', bbox_to_anchor=(1.1, .05), fontsize=4)

            general_utils.save_plot(fig, os.path.join(preference_folder, str(v["fixed_geonameid"]) + "_group_vs_user"))
            kendelltau = scipy.stats.kendalltau(top_user.user_ranking,
                                                       top_user.group_ranking)
            res["kendelltau"] = kendelltau[0]
            res["kendelltau_pvalue"] = kendelltau[1]

            cities_preferences.append(res)
            plt.clf()

        df_preferences = pd.DataFrame(cities_preferences)
        df_preferences = df_preferences.merge(df_cities, on="fixed_geonameid")

        save2csv(df_preferences, os.path.join(preference_folder, "all_cities.csv"))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        df_preferences["fixed_city"] = df_preferences["fixed_city"].apply(lambda x: unicode(x, "utf-8"))
        img = df_preferences.set_index("fixed_city")[["kendelltau"]].sort_values("kendelltau").plot.bar(ax=ax,
                                                                                                        figsize=(10, 5))
        ax.set_axis_bgcolor('white')
        ax.set_xlabel('City')
        ax.set_ylabel("Kendall's tau coefficient")
        ax.legend().set_visible(False)

        general_utils.save_plot(fig, os.path.join(preference_folder, "all_cities"))

        return cities_preferences

    def get_city_summary(self):
        write_running_log("get_city_summary", self.logger)

        df_users_all = self.data_dic_all["users_all"]
        df_checkins_all = self.data_dic_all["checkins_all"]
        df_checkin_group_all = self.data_dic_all["checkin_group_all"]
        df_checkins_with_all = self.data_dic_all["checkins_with_all"]
        df_venues_all = self.data_dic_all["venues_all"]

        df_checkins_merged = df_checkins_all.merge(df_venues_all, on="venue_id")
        df_checkins_merged = df_checkins_merged.merge(df_checkin_group_all[["checkin_id", "group_id"]], on="checkin_id",
                                                      how="left")

        df_checkins_merged = df_checkins_merged.groupby("checkin_id").last().reset_index()

        all_summary = df_checkins_merged.groupby("fixed_city").agg(lambda x: len(np.unique(x))).reset_index()

        all_summary_group = df_checkins_merged[~df_checkins_merged.group_id.isnull()].groupby("fixed_city").agg(
            lambda x: len(np.unique(x))).reset_index()

        all_summary = all_summary[["fixed_city", "checkin_id", "venue_id", "category_id"]]
        all_summary = all_summary.rename(columns={"fixed_city": "City"})
        all_summary = all_summary.rename(columns={"checkin_id": "Total Checkins"})
        all_summary = all_summary.rename(columns={"venue_id": "Total Venues"})
        all_summary = all_summary.rename(columns={"category_id": "Total Categories"})

        all_summary_group = all_summary_group[["fixed_city", "checkin_id", "venue_id", "category_id"]]
        all_summary_group = all_summary_group.rename(columns={"fixed_city": "City"})
        all_summary_group = all_summary_group.rename(columns={"checkin_id": "Group Checkins"})
        all_summary_group = all_summary_group.rename(columns={"venue_id": "Group Venues"})
        all_summary_group = all_summary_group.rename(columns={"category_id": "Group Categories"})

        df_summary = all_summary.merge(all_summary_group, on="City").sort_values("Total Checkins", ascending=False)

        save2csv(df_summary, os.path.join(self.analysis_folder, "city_summary.csv"))

        write_log("***** Summary file per city generated with the name city_summary.csv",
                  self.logger, True, False, print_color="green")

        return df_summary

    def analize_distance_to_group(self):
        write_running_log("analise_distance_to_group", self.logger)

        eps = self.settings.getfloat("settings", "eps")
        min_sample = self.settings.getint("settings", "min_sample")

        distance_to_group_folder = os.path.join(self.analysis_folder, "distance_to_group")

        if not os.path.exists(distance_to_group_folder):
            os.mkdir(distance_to_group_folder)

        df_checkins_all = self.data_dic_all["checkins_all"]
        df_checkins_group = self.data_dic_all["checkin_group_all"]
        df_venues_all = self.data_dic_all["venues_all"]
        df_groups_all = self.data_dic_all["groups_all"]

        df_checkins_venues = df_checkins_all.merge(df_venues_all, on="venue_id")
        df_checkins_venues = df_checkins_venues.merge(df_checkins_group[["checkin_id", "group_id"]], on="checkin_id",
                                                      how="left")
        df_cities = df_checkins_venues.groupby(["fixed_city", "fixed_geonameid"]).count()[["checkin_id"]].reset_index()
        df_cities = df_cities.sort_values("checkin_id", ascending=False).head(10)

        distances_per_city = {}
        for i,v in df_cities.iterrows():
            fixed_geonameid = v["fixed_geonameid"]
            distance_to_group_file = os.path.join(distance_to_group_folder, str(fixed_geonameid) +
                                                  "_average_distance_to_group.csv")

            if os.path.exists(distance_to_group_file):
                write_log("***** Using existing average distance to group file, erase " + distance_to_group_file +
                          " to re-compute", self.logger, True, False, print_color="green")
                df_distances = pd.read_csv(distance_to_group_file)
            else:
                city_checkins = df_checkins_venues.query("fixed_geonameid == @fixed_geonameid")

                df_groups_checkins = city_checkins[~city_checkins.group_id.isnull()].groupby("group_id")

                write_log("***** " + str(fixed_geonameid) + " Calculating group centroids", self.logger, True, False,
                          print_color="green")

                parallel_results = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=5)(
                    delayed(get_clusters)(group, eps, min_sample) for name, group in df_groups_checkins)

                df_group_clusters = deque()
                for r in parallel_results:
                    group_cluster = r["checkins"].drop("cluster_checkins", axis=1).merge(
                        r["checkins_centroids"], on="cluster")[["checkin_id", "group_id", "cluster", "cluster_checkins"
                                                                , "centroid_lat", "centroid_long"]]
                    df_group_clusters.append(group_cluster)

                del parallel_results

                df_group_clusters = pd.concat(df_group_clusters)
                save2csv(df_group_clusters,  os.path.join(distance_to_group_folder, str(fixed_geonameid)
                                                          + "_group_centroids.csv"))

                df_users_checkins = city_checkins[city_checkins.group_id.isnull()].groupby("user_id")

                write_log("***** " + str(fixed_geonameid) + " Calculating user centroids and distances to groups",
                          self.logger, True, False, print_color="green")

                parallel_results = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=5)(
                    delayed(avg_distance_to_group)(
                        df_group_clusters[df_group_clusters.group_id.isin(
                            df_groups_all.query("user_id == @user_id").group_id.values)],
                        user_checkins, user_id, eps, min_sample)
                    for user_id, user_checkins in df_users_checkins
                )

                # Process the results
                df_users_clusters = deque()
                df_distances = deque()
                for r in [r for r in parallel_results if not r is None]:
                    res = OrderedDict({})
                    user_clusters = r["user_cluster"]["checkins"].drop("cluster_checkins", axis=1).merge(
                        r["user_cluster"]["checkins_centroids"], on="cluster")[
                        ["checkin_id", "user_id", "cluster", "cluster_checkins", "centroid_lat", "centroid_long"]]
                    df_users_clusters.append(user_clusters)
                    res["user_id"] = r["user_id"]
                    res["distance_average"] = r["distance_average"]
                    res["distance_weighted_average"] = r["distance_weighted_average"]
                    df_distances.append(res)

                del parallel_results

                df_users_clusters = pd.concat(df_users_clusters)
                df_distances = pd.DataFrame(list(df_distances))

                save2csv(df_users_clusters,  os.path.join(distance_to_group_folder, str(fixed_geonameid)
                                                          + "_user_centroids.csv"))
                save2csv(df_distances, distance_to_group_file)

            distances_per_city[fixed_geonameid] = df_distances

            # Plot the distances
            distance_names = {"distance_average":"Average Distance to Groups (kms)",
                              "distance_weighted_average": "Distance to Groups (kms)"
                              }
            for distance in ["distance_average", "distance_weighted_average"]:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                img = df_distances[[distance]].plot.hist(ax=ax)
                ax.set_ylabel(distance_names[distance])
                ax.set_xlabel("Frequency")
                ax.set_axis_bgcolor('white')
                ax.legend().set_visible(False)

                save_plot(fig, distance_to_group_file.replace(".csv", "") + "_" + distance + "_histogram")
                plt.clf()

                fig = plt.figure()
                ax = fig.add_subplot(111)
                img = sns.kdeplot(df_distances[distance], shade=True, ax=ax)
                # sns.plt.title("Average Distance to Group - Kernel Density Estimation")
                ax.set_xlabel(distance_names[distance], fontsize="x-large")
                ax.set_ylabel("Density", fontsize="x-large")
                ax.set_axis_bgcolor('white')
                ax.legend().set_visible(False)
                ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

                for label in ticklabels:
                    label.set_color('black')
                    label.set_fontsize('large')

                save_plot(fig, distance_to_group_file.replace(".csv", "") + "_" + distance + "_kde")
                plt.clf()

        return {"distances_per_city": distances_per_city}