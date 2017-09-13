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
from sklearn.neighbors.kde import KernelDensity
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from collections import deque


def get_group_clusters_kde(group_id, checkins_train, group_test, possible_venues, kernel, bandwidth):
    # Assign all the clusters that DBSCAN didn't fit to the cluster -1
    all_possible_venues_cluster = deque()
    df_current_group_test = group_test.query("group_id == @group_id")
    for c in [c for c in df_current_group_test.group_cluster.unique() if c in checkins_train.group_cluster.unique()]:
        cluster_checkins = checkins_train.query("group_cluster == @c")
        if len(cluster_checkins) > 1:
            kde = KDE(bandwidth=bandwidth, kernel=kernel)
            kde.fit(cluster_checkins)
            df_kde_score = kde.score_sample(possible_venues)
            possible_venues = possible_venues[possible_venues.venue_id.isin(df_kde_score.query("kde_q >=3").venue_id)]
            all_possible_venues_cluster.append((c, possible_venues.venue_id.values))
        else:
            all_possible_venues_cluster.append((c, None))
    return {"group_id": group_id,
            "all_possible_venues_cluster": all_possible_venues_cluster}


def predict_group(group_id, checkins_train, checkins_test, possible_venues, df_recommended, kernel, bandwidth):
    # Assign all the clusters that DBSCAN didn't fit to the cluster -1
    checkins_test = checkins_test.query("group_id == @group_id")

    if len(checkins_test) > 0:
        all_df_recommended_kde = deque()
        for c in [c for c in checkins_test.group_cluster.unique() if c in checkins_train.group_cluster.unique()]:
            kde = KDE(bandwidth=bandwidth, kernel=kernel)
            kde.fit(df_checkins_train=checkins_train.query("group_cluster == @c"))
            df_kde_score = kde.score_sample(possible_venues)

            df_recommended_kde = df_recommended[df_recommended.venue_id.isin(df_kde_score.query("kde_q >=3").venue_id)]
            df_recommended_kde["rank"] = df_recommended_kde["rank"].rank()
            df_recommended_kde = df_recommended_kde[["group_id", "venue_id", "score", "rank"]]
            df_recommended_kde["group_cluster"] = c

            all_df_recommended_kde.append(df_recommended_kde)
        if len(all_df_recommended_kde) > 0:
            all_df_recommended_kde = pd.concat(all_df_recommended_kde)
    else:
        all_df_recommended_kde = None

    if all_df_recommended_kde is not None and len(all_df_recommended_kde) == 0:
        all_df_recommended_kde = None

    return {"group_id": group_id,
            "all_df_recommended_kde": all_df_recommended_kde,
            "df_recommended": df_recommended}


def combine_model_with_kde(model_name, group_model, group_train, group_test, dic_group_cluster_venues,
                  df_venues, cutoffs, kernel, bandwidth, combined_model_folder, logger):
    general_utils.write_running_log(model_name, logger)

    all_res_gl = deque()
    all_res_kde = deque()

    df_test_grouped = group_test.groupby("group_id")

    for group_id, test_checkins in df_test_grouped:
        df_train_for_group = group_train.query("group_id == @group_id")
        for c in [c for c in test_checkins.group_cluster.unique() if c in df_train_for_group.group_cluster.unique()]:
            sf_test_cluster = gl.SFrame(test_checkins.query("group_cluster == @c")[["group_id", "venue_id"]])

            # Recommend only items inside the KDE quartile
            # By default, GraphLab will not recommend items previously consumed

            possible_venues_in_kde = dic_group_cluster_venues[group_id][c]

            df_recommended_kde_only = group_model.recommend(users=[group_id],
                                                            items=possible_venues_in_kde,
                                                            k=max(cutoffs))

            # Recommend for all items
            df_recommended = group_model.recommend(users=[group_id], k=max(cutoffs))

            res_gl = gl.recommender.util.precision_recall_by_user(observed_user_items=sf_test_cluster,
                                                                  recommendations=df_recommended,
                                                                  cutoffs=cutoffs).to_dataframe()

            res_kde = gl.recommender.util.precision_recall_by_user(observed_user_items=sf_test_cluster,
                                                                   recommendations=df_recommended_kde_only,
                                                                   cutoffs=cutoffs).to_dataframe()

            res_gl["group_id"] = group_id
            res_gl["group_cluster"] = c
            res_kde["group_id"] = group_id
            res_kde["group_cluster"] = c

            all_res_gl.append(res_gl)
            all_res_kde.append(res_kde)

    all_res_gl = pd.concat(all_res_gl)
    all_res_kde = pd.concat(all_res_kde)

    all_res_gl["model_name"] = model_name
    all_res_kde["model_name"] = model_name

    # Store the evaluation results
    general_utils.save2csv(all_res_gl, os.path.join(combined_model_folder, "all_res_gl.csv"))
    general_utils.save2csv(all_res_kde, os.path.join(combined_model_folder, "all_res_kde.csv"))

    # Calculate the average
    sf_all_res_gl = gl.SFrame(all_res_gl).groupby('cutoff',
                                                  {'mean_precision': gl.aggregate.AVG('precision'),
                                                   'mean_recall': gl.aggregate.AVG('recall')}).topk('cutoff',
                                                                                                    reverse=True)

    print sf_all_res_gl
    general_utils.save2csv(sf_all_res_gl, os.path.join(combined_model_folder, "summary_gl_results.csv"))

    sf_all_res_kde = gl.SFrame(all_res_kde).groupby('cutoff',
                                               {'mean_precision': gl.aggregate.AVG('precision'),
                                                'mean_recall': gl.aggregate.AVG('recall')}).topk('cutoff',
                                                                                                 reverse=True)

    print sf_all_res_kde
    general_utils.save2csv(sf_all_res_kde, os.path.join(combined_model_folder, "summary_kde_results.csv"))

    return {"all_res_gl": all_res_gl,
            "all_res_kde": all_res_kde,
            "sf_all_res_kde": sf_all_res_kde,
            "sf_all_res_gl": sf_all_res_gl}


def combine_model(model_name, group_model, group_train, group_test, possible_venues,
                  df_venues, df_group_centroids, cutoffs,
                  kernel, bandwidth, combined_model_folder, logger):
    general_utils.write_running_log(model_name, logger)

    group_train = group_train.to_dataframe().merge(df_venues, on="venue_id")
    group_test = group_test.to_dataframe().merge(df_venues, on="venue_id")

    group_train = group_train.merge(df_group_centroids, on="checkin_id", how="left")
    group_test = group_test.merge(df_group_centroids, on="checkin_id", how="left")

    group_train = group_train[group_train.group_id.isin(group_test.group_id)]
    grouped_train = group_train.groupby("group_id")

    parallel_results = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=5)(
        delayed(predict_group)(
            group_id, checkins_train, group_test, possible_venues,
            group_model.recommend([group_id], k=1000).to_dataframe().merge(df_venues, on="venue_id"),
            kernel, bandwidth
        )
        for group_id, checkins_train in grouped_train)

    all_res_gl = deque()
    all_res_kde = deque()

    for r in parallel_results:
        group_id = r["group_id"]
        all_df_recommended_kde = r["all_df_recommended_kde"]
        df_recommended = r["df_recommended"]
        if all_df_recommended_kde is not None:
            for c in all_df_recommended_kde.group_cluster.unique():
                df_recommended_kde = all_df_recommended_kde.query("group_cluster == @c")
                res_gl = gl.recommender.util.precision_recall_by_user(observed_user_items=gl.SFrame(
                    group_test.query("group_id == @group_id and group_cluster == @c")[["group_id", "venue_id"]]),
                    recommendations=gl.SFrame(df_recommended[
                                                  ["group_id", "venue_id",
                                                   "score", "rank"]]),
                    cutoffs=cutoffs).to_dataframe()

                res_kde = gl.recommender.util.precision_recall_by_user(observed_user_items=gl.SFrame(
                    group_test.query("group_id == @group_id and group_cluster == @c")[["group_id", "venue_id"]]),
                    recommendations=gl.SFrame(df_recommended_kde[
                                                  ["group_id", "venue_id",
                                                   "score", "rank"]]),
                    cutoffs=cutoffs).to_dataframe()

                res_gl["group_id"] = group_id
                res_gl["group_cluster"] = c
                res_kde["group_id"] = group_id
                res_kde["group_cluster"] = c

                all_res_gl.append(res_gl)
                all_res_kde.append(res_kde)

    all_res_gl = pd.concat(all_res_gl)
    all_res_kde = pd.concat(all_res_kde)

    all_res_gl["model_name"] = model_name
    all_res_kde["model_name"] = model_name

    # Store the evaluation results
    general_utils.save2csv(all_res_gl, os.path.join(combined_model_folder, "all_res_gl.csv"))
    general_utils.save2csv(all_res_kde, os.path.join(combined_model_folder, "all_res_kde.csv"))

    # Calculate the average
    sf_all_res_gl = gl.SFrame(all_res_gl).groupby('cutoff',
                                                  {'mean_precision': gl.aggregate.AVG('precision'),
                                                   'mean_recall': gl.aggregate.AVG('recall')}).topk('cutoff',
                                                                                                    reverse=True)

    print sf_all_res_gl
    general_utils.save2csv(sf_all_res_gl, os.path.join(combined_model_folder, "summary_gl_results.csv"))

    sf_all_res_kde = gl.SFrame(all_res_kde).groupby('cutoff',
                                               {'mean_precision': gl.aggregate.AVG('precision'),
                                                'mean_recall': gl.aggregate.AVG('recall')}).topk('cutoff',
                                                                                                 reverse=True)

    print sf_all_res_kde
    general_utils.save2csv(sf_all_res_kde, os.path.join(combined_model_folder, "summary_kde_results.csv"))

    return {"all_res_gl": all_res_gl,
            "all_res_kde": all_res_kde,
            "sf_all_res_kde": sf_all_res_kde,
            "sf_all_res_gl": sf_all_res_gl}


class KDE:
    """
        A KDE estimation based on gaussian distribution. The latitude and longitude are
        projected to cartesian coordinates so we can use the KernelDensity module from scikit learn.
    """

    def __init__(self, kernel, bandwidth):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.kde = None

    def fit(self, df_checkins_train):
        Xtrain = df_checkins_train[["projected_x", "projected_y", "projected_z"]].values

        kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        kde.fit(Xtrain)
        self.kde = kde

    def score_sample(self, possible_venues):
        assert not self.kde is None, "Before scoring use the fit() method."
        # Compute Geo
        Xtest = possible_venues[["projected_x", "projected_y", "projected_z"]].values

        possible_venues["kde"] = self.kde.score_samples(Xtest)
        possible_venues["kde_q"] = pd.qcut(possible_venues["kde"], 4, labels=False)

        return possible_venues
