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
from utils.general_utils import *

gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', multiprocessing.cpu_count())


def erase_jobs():
    for i in gl.deploy.jobs.list().to_dataframe().index:
        try:
            del gl.deploy.jobs[i]
        except Exception as ex:
            print str(ex)
            pass

def save_model_parameters(parameters, experiments_folder):
    experiments_file = os.path.join(experiments_folder, "experiments.csv")
    if os.path.isfile(experiments_file):
        df_experiments = csv2pandas(experiments_file)
        df_experiments = df_experiments.append([parameters])
    else:
        df_experiments = pd.DataFrame([parameters])
    save2csv(df_experiments.drop_duplicates(), experiments_file)


def store_train_test(filepath, train_group, test_group, train_user, test_user):
    """
            This method creates a folder inside 'experiments_folder' with an identifier for the experiment.
            It also stores the predictions, the training dataset and testing dataset
    """
    save2csv(train_group, os.path.join(filepath, "train_group.csv"))
    save2csv(test_group, os.path.join(filepath, "test_group.csv"))
    save2csv(train_user, os.path.join(filepath, "train_user.csv"))
    save2csv(test_user, os.path.join(filepath, "test_user.csv"))


def load_train_test(filepath):
    """
            This method creates a folder inside 'experiments_folder' with an identifier for the experiment.
            It also stores the predictions, the training dataset and testing dataset
    """
    train_group = gl.SFrame(pd.read_csv(os.path.join(filepath, "train_group.csv")))
    test_group = gl.SFrame(pd.read_csv(os.path.join(filepath, "test_group.csv")))
    train_user = gl.SFrame(pd.read_csv(os.path.join(filepath, "train_user.csv")))
    test_user = gl.SFrame(pd.read_csv(os.path.join(filepath, "test_user.csv")))

    return {"train": train_group, "test": test_group, "train_user": train_user, "test_user": test_user}


class Splitter:
    """
        This class is in charge of separating the dataset in a training dataset and
        a testing dataset based on different methods.
    """

    def __init__(self, logger, datasets, settings, dataset):
        """
            The constructor joins the groups check-ins and the check-ins information.
            It also filters the dataset using the minimum and maximum check-in counts of the group
        """

        self.logger = logger
        self.settings = settings
        self.datasets = datasets
        self.dataset = dataset
        df_all_group_checkins = self.datasets["df_checkin_group"]
        self.df_all_group_checkins = df_all_group_checkins.sort_values(["group_id", "checkin_created_at"],
                                                                       ascending=True)

    def split_by_group(self, item_test_proportion, max_num_users):
        """
            This method applies the GraphLab Create method graphlab.recommender.util.random_split_by_user
            Check the documentation:
            https://turi.com/products/create/docs/generated/graphlab.recommender.util.random_split_by_user.html
        """
        sf_all_group_checkins = gl.SFrame(self.df_all_group_checkins)
        sf_train, sf_test = gl.recommender.util.random_split_by_user(sf_all_group_checkins,
                                                                     item_test_proportion=item_test_proportion,
                                                                     max_num_users=max_num_users,
                                                                     user_id="group_id",
                                                                     item_id="venue_id")

        df_all_user = self.datasets["df_checkins"]
        df_train_user = df_all_user[~df_all_user.checkin_id.isin(sf_test.to_dataframe().checkin_id)]
        sf_train_user = gl.SFrame(
            df_train_user[["user_id", "checkin_id", "venue_id", "checkin_created_at"]].drop_duplicates())
        df_test_user = df_all_user[df_all_user.checkin_id.isin(sf_test.to_dataframe().checkin_id)]
        sf_test_user = gl.SFrame(
            df_test_user[["user_id", "checkin_id", "venue_id", "checkin_created_at"]].drop_duplicates())
        return {"train": sf_train, "test": sf_test, "train_user": sf_train_user, "test_user": sf_test_user}

    def split_cluster(self, checkins, using_time=False):
        size = len(checkins.venue_id.unique())

        if using_time:
            checkins = checkins.sort_values("checkin_created_at", ascending=True)
            training = pd.DataFrame(checkins.venue_id.unique()[:int(size * .7)], columns=["venue_id"])
        else:
            training = pd.DataFrame(np.random.choice(checkins.venue_id.unique(), int(size * .7), replace=False),
                                    columns=["venue_id"])
        checkins.loc[checkins.venue_id.isin(training.venue_id), "training"] = True
        checkins.loc[~checkins.venue_id.isin(training.venue_id), "training"] = False
        return checkins[["venue_id", "training"]].drop_duplicates()

    def split_by_cluster(self, using_time=False):
        fixed_geonameid = self.settings.getint(self.dataset, "fixed_geonameid")
        distance_to_group_folder = self.settings.get(self.dataset, "distance_to_group_folder")

        df_checkins = self.datasets["df_checkins"]
        df_checkin_group = self.datasets["df_checkin_group"]
        df_venues = self.datasets["df_venues"]

        df_group_centroids = pd.read_csv(os.path.join(distance_to_group_folder,
                                                      str(fixed_geonameid) + "_group_centroids.csv"))

        df_user_centroids = pd.read_csv(os.path.join(distance_to_group_folder,
                                                     str(fixed_geonameid) + "_user_centroids.csv"))

        df_group_centroids = df_group_centroids.rename(
            columns={"cluster": "group_cluster", "cluster_checkins": "group_cluster_checkins",
                     "centroid_lat": "group_centroid_lat", "centroid_long": "group_centroid_long"})
        df_group_centroids = df_group_centroids.drop("group_id", axis=1)

        df_user_centroids = df_user_centroids.rename(
            columns={"cluster": "user_cluster", "cluster_checkins": "user_cluster_checkins",
                     "centroid_lat": "user_centroid_lat", "centroid_long": "user_centroid_long"})
        df_user_centroids = df_user_centroids.drop("user_id", axis=1)

        df_checkins_merged = df_checkins.merge(df_checkin_group[["group_id", "checkin_id"]], on="checkin_id",
                                               how="left")
        df_checkins_merged = df_checkins_merged.merge(df_venues, on="venue_id")

        df_checkins_merged = df_checkins_merged.query("fixed_geonameid == @fixed_geonameid")

        df_checkins_merged = df_checkins_merged.merge(df_group_centroids, on="checkin_id", how="left")
        df_checkins_merged = df_checkins_merged.merge(df_user_centroids, on="checkin_id", how="left")

        df_checkins_group_merged = df_checkins_merged[~df_checkins_merged.group_id.isnull()]

        # We use clusters above the median to make the split
        filter_size = df_checkins_group_merged[["group_cluster_checkins"]].median().values[0]

        df_splits = df_checkins_group_merged.query("group_cluster_checkins > @filter_size").groupby(["group_id",
                                                                                                     "group_cluster"])

        df_splitted = df_splits.apply(lambda checkins: self.split_cluster(checkins, using_time)).reset_index()
        df_splitted = df_splitted[["group_id", "group_cluster", "venue_id", "training"]]

        df_testing = df_splitted.query("training == False")
        df_testing["remove"] = True
        df_testing = df_testing[["group_id", "venue_id", "remove"]]

        df_checkins_merged = df_checkins_merged.merge(df_testing, on=["group_id", "venue_id"], how="left")
        checkins_training = df_checkins_merged[df_checkins_merged.remove.isnull()].checkin_id.values
        checkins_testing = df_checkins_merged[~df_checkins_merged.remove.isnull()].checkin_id.values

        # Remove the check-ins from the training set
        df_groups_train = self.df_all_group_checkins[self.df_all_group_checkins.checkin_id.isin(checkins_training)]
        df_groups_test = self.df_all_group_checkins[self.df_all_group_checkins.checkin_id.isin(checkins_testing)]

        sf_train = gl.SFrame(df_groups_train)
        sf_test = gl.SFrame(df_groups_test)

        df_all_user = self.datasets["df_checkins"]
        df_train_user = df_all_user[~df_all_user.checkin_id.isin(sf_test.to_dataframe().checkin_id)]
        sf_train_user = gl.SFrame(
            df_train_user[["user_id", "checkin_id", "venue_id", "checkin_created_at"]].drop_duplicates())
        df_test_user = df_all_user[df_all_user.checkin_id.isin(sf_test.to_dataframe().checkin_id)]
        sf_test_user = gl.SFrame(
            df_test_user[["user_id", "checkin_id", "venue_id", "checkin_created_at"]].drop_duplicates())
        return {"train": sf_train, "test": sf_test, "train_user": sf_train_user, "test_user": sf_test_user}


class Baselines:
    """
        This class is in charge of running the baseline methods
    """

    def __init__(self, logger, settings, datasets, dataset, folder_name, sf_train, sf_test, sf_train_user, sf_test_user,
                 df_venues_content, cutoffs):
        self.logger = logger
        self.settings = settings
        self.datasets = datasets
        self.dataset = dataset
        self.folder_name = folder_name
        self.sf_train = sf_train
        self.sf_test = sf_test
        self.sf_train_user = sf_train_user
        self.sf_test_user = sf_test_user
        if isinstance(df_venues_content["category_tree"].values[0], str):
            df_venues_content["category_tree"] = df_venues_content["category_tree"].apply(lambda x: x.split("|"))
        self.sf_venues_content = gl.SFrame(df_venues_content)
        self.cutoffs = cutoffs
        self.max_cutoff = max(self.cutoffs)

        # Add the venue to the
        # df_train_user = self.sf_train_user.to_dataframe()
        # df_train_user = df_train_user[["user_id", "checkin_id"]].drop_duplicates()
        # df_train_user = df_train_user.merge(df_checkins, on="checkin_id")
        # df_train_user = df_train_user[["user_id_x", "venue_id", "checkin_created_at", "checkin_id"]]
        # df_train_user = df_train_user.rename(columns={"user_id_x": "user_id"})
        #
        # df_test_user = self.sf_test_user.to_dataframe()
        # df_test_user = df_test_user[["user_id", "checkin_id"]].drop_duplicates()
        # df_test_user = df_test_user.merge(df_checkins, on="checkin_id")
        # df_test_user = df_test_user[["user_id_x", "venue_id", "checkin_created_at", "checkin_id"]]
        # df_test_user = df_test_user.rename(columns={"user_id_x": "user_id"})

    def popularity_recommender(self, model_folder, use_content):
        """
                Calls gl.popularity_recommender.create method and returns the model and the predictions
        """
        sf_venues_content = self.sf_venues_content
        if use_content == "cat":
            sf_venues_content = sf_venues_content[["venue_id", "category_tree"]]

        if use_content == "geo":
            sf_venues_content = sf_venues_content[["venue_id", "projected_x", "projected_y", "projected_z"]]

        # Group Model
        if use_content != "":
            group_model = gl.popularity_recommender.create(self.sf_train,
                                                           item_id="venue_id",
                                                           user_id="group_id",
                                                           item_data=sf_venues_content
                                                           )
        else:
            group_model = gl.popularity_recommender.create(self.sf_train,
                                                           item_id="venue_id",
                                                           user_id="group_id"
                                                           )
        group_predictions = group_model.recommend(k=self.max_cutoff).to_dataframe()

        # User Model
        if use_content != "":
            user_model = gl.item_similarity_recommender.create(self.sf_train_user,
                                                               item_id="venue_id",
                                                               user_id="user_id",
                                                               item_data=sf_venues_content
                                                               )
        else:
            user_model = gl.item_similarity_recommender.create(self.sf_train_user,
                                                               item_id="venue_id",
                                                               user_id="user_id"
                                                               )

        user_predictions = user_model.recommend(k=10).to_dataframe()

        group_model.save(os.path.join(model_folder, 'group_model'))
        user_model.save(os.path.join(model_folder, 'user_model'))

        save2csv(group_predictions, os.path.join(model_folder, "group_predictions.csv"))
        save2csv(user_predictions, os.path.join(model_folder, "user_predictions.csv"))

        return {"group_model": group_model, "group_predictions": group_predictions,
                "user_model": user_model, "user_predictions": user_predictions}

    def item_to_item_recommender(self, model_folder):
        """
                Calls gl.recommender.create method and returns the model and the predictions
        """
        parameters = {'similarity_type': ["jaccard", "cosine", "pearson"],
                      "item_id": "venue_id",
                      "user_id": "group_id"}

        split = gl.recommender.util.random_split_by_user(self.sf_train,
                                                         item_id="venue_id",
                                                         user_id="group_id",
                                                         item_test_proportion=.05)
        sf_train_val = split[0]
        sf_test_val = split[1]

        job = gl.grid_search.create((sf_train_val, sf_test_val),
                                    gl.item_similarity_recommender.create,
                                    parameters, return_model=True)

        best_params = job.get_best_params("validation_recall@5")
        erase_jobs()

        group_model = gl.item_similarity_recommender.create(self.sf_train,
                                                            item_id="venue_id",
                                                            user_id="group_id",
                                                            similarity_type=best_params["similarity_type"]
                                                            )

        group_predictions = group_model.recommend(k=self.max_cutoff).to_dataframe()

        # User Model
        split = gl.recommender.util.random_split_by_user(self.sf_train_user,
                                                         item_id="venue_id",
                                                         user_id="user_id",
                                                         item_test_proportion=.05)
        sf_train_val = split[0]
        sf_test_val = split[1]

        parameters["user_id"] = "user_id"
        job = gl.grid_search.create((sf_train_val, sf_test_val),
                                    gl.item_similarity_recommender.create,
                                    parameters, return_model=True)

        best_params = job.get_best_params("validation_recall@5")
        erase_jobs()

        user_model = gl.item_similarity_recommender.create(self.sf_train_user,
                                                           item_id="venue_id",
                                                           user_id="user_id",
                                                           similarity_type=best_params["similarity_type"]
                                                           )

        user_predictions = user_model.recommend(k=10).to_dataframe()

        group_model.save(os.path.join(model_folder, 'group_model'))
        user_model.save(os.path.join(model_folder, 'user_model'))

        save2csv(group_predictions, os.path.join(model_folder, "group_predictions.csv"))
        save2csv(user_predictions, os.path.join(model_folder, "user_predictions.csv"))

        return {"group_model": group_model, "group_predictions": group_predictions,
                "user_model": user_model, "user_predictions": user_predictions}

    def content_based_recommender_best_weight(self, using_key, sf_venues_content, sf_train_val, sf_test_val):
        df_param_results = []
        for w in np.arange(0.01, .02, 0.005):
            weights = {'category_tree': w,
                       'projected_x': 1 - w,
                       'projected_y': 1 - w,
                       'projected_z': 1 - w}

            group_model = gl.item_content_recommender.create(item_data=sf_venues_content,
                                                             observation_data=sf_train_val,
                                                             user_id=using_key,
                                                             item_id="venue_id",
                                                             max_item_neighborhood_size=1000,
                                                             weights=weights,
                                                             verbose=False)
            sf_recommendations = group_model.recommend(k=50)
            sf_evaluation = gl.recommender.util.precision_recall_by_user(
                observed_user_items=gl.SFrame(sf_test_val[[using_key, "venue_id"]]),
                recommendations=sf_recommendations,
                cutoffs=[5, 10, 20, 30, 40, 50])

            sf_evaluation_summary = sf_evaluation.groupby('cutoff',
                                                          {'mean_precision': gl.aggregate.AVG('precision'),
                                                           'mean_recall': gl.aggregate.AVG('recall')}).topk('cutoff',
                                                                                                            reverse=True)

            df_param_results.append(
                OrderedDict({"weight": w, "validation_recall@5": sf_evaluation_summary[0]["mean_recall"]}))
        df_param_results = pd.DataFrame(df_param_results)
        best_w = df_param_results.sort_values("validation_recall@5", ascending=False).values[0][1]
        return {'category_tree': best_w,
                       'projected_x': 1 - best_w,
                       'projected_y': 1 - best_w,
                       'projected_z': 1 - best_w}

    def content_based_recommender(self, model_folder, use_content):
        """
                Calls gl.recommender.item_content_recommender method and returns the model and the predictions
        """
        # TODO: Add category tree to the similarity
        sf_venues_content = self.sf_venues_content

        if use_content == "cat":
            sf_venues_content = sf_venues_content[["venue_id", "category_tree"]]

        if use_content == "geo":
            sf_venues_content = sf_venues_content[["venue_id", "projected_x", "projected_y", "projected_z"]]

        split = gl.recommender.util.random_split_by_user(self.sf_train,
                                                         item_id="venue_id",
                                                         user_id="group_id",
                                                         item_test_proportion=.05)
        sf_train_val = split[0]
        sf_test_val = split[1]

        if use_content == "catgeo":
            best_weight = self.content_based_recommender_best_weight("group_id", sf_venues_content, sf_train_val, sf_test_val )
        else:
            best_weight = "auto"

        group_model = gl.recommender.item_content_recommender.create(item_data=sf_venues_content,
                                                                     observation_data=self.sf_train,
                                                                     user_id="group_id",
                                                                     item_id="venue_id",
                                                                     max_item_neighborhood_size=1000,
                                                                     weights=best_weight)
        group_predictions = group_model.recommend(k=self.max_cutoff).to_dataframe()

        # User model
        split = gl.recommender.util.random_split_by_user(self.sf_train_user,
                                                         item_id="venue_id",
                                                         user_id="user_id",
                                                         item_test_proportion=.05)
        sf_train_val = split[0]
        sf_test_val = split[1]

        if use_content == "catgeo":
            best_weight = self.content_based_recommender_best_weight("user_id", sf_venues_content, sf_train_val,
                                                                    sf_test_val)
        else:
            best_weight = "auto"

        user_model = gl.recommender.item_content_recommender.create(item_data=sf_venues_content,
                                                                    observation_data=self.sf_train_user,
                                                                    user_id="user_id",
                                                                    item_id="venue_id",
                                                                    max_item_neighborhood_size=1000,
                                                                    weights=best_weight
                                                                    )
        user_predictions = user_model.recommend(k=10).to_dataframe()

        group_model.save(os.path.join(model_folder, 'group_model'))
        user_model.save(os.path.join(model_folder, 'user_model'))

        save2csv(group_predictions, os.path.join(model_folder, "group_predictions.csv"))
        save2csv(user_predictions, os.path.join(model_folder, "user_predictions.csv"))

        return {"group_model": group_model, "group_predictions": group_predictions,
                "user_model": user_model, "user_predictions": user_predictions}

    def implicit_matrix_factorization(self, model_folder):
        """
                Calls gl.ranking_factorization_recommender.create method and returns the model and the predictions
        """
        # Group Model
        parameters = {'num_factors': range(50, 91, 20),
                      'ials_confidence_scaling_factor': range(10, 21, 2),
                      'regularization': [.01, .001, .0001],
                      "item_id": "venue_id",
                      "user_id": "group_id",
                      "solver": "ials"}

        split = gl.recommender.util.random_split_by_user(self.sf_train,
                                                         item_id="venue_id",
                                                         user_id="group_id",
                                                         item_test_proportion=.05)
        sf_train_val = split[0]
        sf_test_val = split[1]

        job = gl.grid_search.create((sf_train_val, sf_test_val),
                                    gl.ranking_factorization_recommender.create,
                                    parameters, return_model=True)

        best_params = job.get_best_params("validation_recall@5")
        erase_jobs()

        group_model = gl.ranking_factorization_recommender.create(self.sf_train,
                                                                  item_id="venue_id",
                                                                  user_id="group_id",
                                                                  solver="ials",
                                                                  num_factors=best_params["num_factors"],
                                                                  regularization=best_params["regularization"],
                                                                  ials_confidence_scaling_factor=
                                                                  best_params["ials_confidence_scaling_factor"]
                                                                  )

        group_predictions = group_model.recommend(k=self.max_cutoff).to_dataframe()

        # User Model
        split = gl.recommender.util.random_split_by_user(self.sf_train_user,
                                                         item_id="venue_id",
                                                         user_id="user_id",
                                                         item_test_proportion=.05)
        sf_train_val = split[0]
        sf_test_val = split[1]

        parameters["user_id"] = "user_id"
        job = gl.grid_search.create((sf_train_val, sf_test_val),
                                    gl.ranking_factorization_recommender.create,
                                    parameters, return_model=True)

        best_params = job.get_best_params("validation_recall@5")
        erase_jobs()

        user_model = gl.ranking_factorization_recommender.create(self.sf_train_user,
                                                                 item_id="venue_id",
                                                                 user_id="user_id",
                                                                 solver="ials",
                                                                 num_factors=best_params["num_factors"],
                                                                 ials_confidence_scaling_factor=
                                                                 best_params["ials_confidence_scaling_factor"]
                                                                 )

        user_predictions = user_model.recommend(k=10).to_dataframe()

        group_model.save(os.path.join(model_folder, 'group_model'))
        user_model.save(os.path.join(model_folder, 'user_model'))

        save2csv(group_predictions, os.path.join(model_folder, "group_predictions.csv"))
        save2csv(user_predictions, os.path.join(model_folder, "user_predictions.csv"))

        return {"group_model": group_model, "group_predictions": group_predictions,
                "user_model": user_model, "user_predictions": user_predictions}

    def sgd_matrix_factorization(self, model_folder):
        """
                Calls gl.ranking_factorization_recommender.create method and returns the model and the predictions
        """
        # Group Model
        parameters = {'num_factors': range(50, 91, 20),
                      'regularization': [.01, .001, .0001],
                      "item_id": "venue_id",
                      "user_id": "group_id",
                      "solver": "adagrad"}

        split = gl.recommender.util.random_split_by_user(self.sf_train,
                                                         item_id="venue_id",
                                                         user_id="group_id",
                                                         item_test_proportion=.05)
        sf_train_val = split[0]
        sf_test_val = split[1]

        job = gl.grid_search.create((sf_train_val, sf_test_val),
                                    gl.ranking_factorization_recommender.create,
                                    parameters, return_model=True)

        best_params = job.get_best_params("validation_recall@5")
        erase_jobs()

        group_model = gl.ranking_factorization_recommender.create(self.sf_train,
                                                                  item_id="venue_id",
                                                                  user_id="group_id",
                                                                  regularization=best_params["regularization"],
                                                                  num_factors=best_params["num_factors"]
                                                                  )

        group_predictions = group_model.recommend(k=self.max_cutoff).to_dataframe()

        # User Model
        split = gl.recommender.util.random_split_by_user(self.sf_train_user,
                                                         item_id="venue_id",
                                                         user_id="user_id",
                                                         item_test_proportion=.05)
        sf_train_val = split[0]
        sf_test_val = split[1]

        parameters["user_id"] = "user_id"
        job = gl.grid_search.create((sf_train_val, sf_test_val),
                                    gl.ranking_factorization_recommender.create,
                                    parameters, return_model=True)

        best_params = job.get_best_params("validation_recall@5")
        erase_jobs()

        user_model = gl.ranking_factorization_recommender.create(self.sf_train_user,
                                                                 item_id="venue_id",
                                                                 user_id="user_id",
                                                                 regularization=best_params["regularization"],
                                                                 num_factors=best_params["num_factors"]
                                                                 )

        user_predictions = user_model.recommend(k=10).to_dataframe()

        group_model.save(os.path.join(model_folder, 'group_model'))
        user_model.save(os.path.join(model_folder, 'user_model'))

        save2csv(group_predictions, os.path.join(model_folder, "group_predictions.csv"))
        save2csv(user_predictions, os.path.join(model_folder, "user_predictions.csv"))

        return {"group_model": group_model, "group_predictions": group_predictions,
                "user_model": user_model, "user_predictions": user_predictions}

    def sgd_content_matrix_factorization(self, model_folder, use_content):
        """
                Calls gl.ranking_factorization_recommender.create method and returns the model and the predictions
        """
        sf_venues_content = self.sf_venues_content
        if use_content == "cat":
            sf_venues_content = sf_venues_content[["venue_id", "category_tree"]]

        if use_content == "geo":
            sf_venues_content = sf_venues_content[["venue_id", "projected_x", "projected_y", "projected_z"]]

        # Group Model
        parameters = {'num_factors': range(50, 91, 20),
                      'regularization': [.01, .001, .0001],
                      "item_id": "venue_id",
                      "user_id": "group_id",
                      "solver": "adagrad",
                      "item_data": sf_venues_content}

        split = gl.recommender.util.random_split_by_user(self.sf_train,
                                                         item_id="venue_id",
                                                         user_id="group_id",
                                                         item_test_proportion=.05)
        sf_train_val = split[0]
        sf_test_val = split[1]

        job = gl.grid_search.create((sf_train_val, sf_test_val),
                                    gl.ranking_factorization_recommender.create,
                                    parameters, return_model=True)

        best_params = job.get_best_params("validation_recall@5")
        erase_jobs()

        group_model = gl.ranking_factorization_recommender.create(self.sf_train,
                                                                  item_id="venue_id",
                                                                  user_id="group_id",
                                                                  regularization=best_params["regularization"],
                                                                  num_factors=best_params["num_factors"],
                                                                  item_data=sf_venues_content
                                                                  )

        group_predictions = group_model.recommend(k=self.max_cutoff).to_dataframe()

        # User Model
        split = gl.recommender.util.random_split_by_user(self.sf_train_user,
                                                         item_id="venue_id",
                                                         user_id="user_id",
                                                         item_test_proportion=.05)
        sf_train_val = split[0]
        sf_test_val = split[1]

        parameters["user_id"] = "user_id"
        job = gl.grid_search.create((sf_train_val, sf_test_val),
                                    gl.ranking_factorization_recommender.create,
                                    parameters, return_model=True)

        best_params = job.get_best_params("validation_recall@5")
        erase_jobs()

        user_model = gl.ranking_factorization_recommender.create(self.sf_train_user,
                                                                 item_id="venue_id",
                                                                 user_id="user_id",
                                                                 regularization=best_params["regularization"],
                                                                 num_factors=best_params["num_factors"],
                                                                 item_data=sf_venues_content
                                                                 )

        user_predictions = user_model.recommend(k=10).to_dataframe()

        group_model.save(os.path.join(model_folder, 'group_model'))
        user_model.save(os.path.join(model_folder, 'user_model'))

        save2csv(group_predictions, os.path.join(model_folder, "group_predictions.csv"))
        save2csv(user_predictions, os.path.join(model_folder, "user_predictions.csv"))

        return {"group_model": group_model, "group_predictions": group_predictions,
                "user_model": user_model, "user_predictions": user_predictions}

    def evaluate_and_store(self, model_folder, parameters, model=None, recommendations=None, combination_method=""):
        """
                This method creates a folder inside 'experiments_folder' with an identifier for the experiment.
                It also stores the predictions, the training dataset and testing dataset
        """
        if model is None and recommendations is None:
            raise Exception("Either pass a model or pass the recommendations")

        save_model_parameters(parameters, self.folder_name)

        if recommendations is None:
            sf_recommendations = model.recommend(k=self.max_cutoff)
        else:
            sf_recommendations = gl.SFrame(data=recommendations) if not isinstance(recommendations, gl.SFrame) \
                else recommendations

            assert recommendations.column_names() != ['group_id', 'venue_id', 'score', 'rank'], \
                "The recommendations SFrame columns must be ['group_id', 'venue_id', 'score', 'rank']"

        if combination_method == "":
            write_running_log("group recommendation " + parameters["baseline"], self.logger)
            recommendations_file = os.path.join(model_folder, "group_recommendations.csv")
            precision_recall_user_file = os.path.join(model_folder, "precision_recall_by_user.csv")
            precision_recall_file = os.path.join(model_folder, "precision_recall.csv")
        else:
            write_running_log("combination: " + combination_method, self.logger)
            recommendations_file = os.path.join(model_folder, "recommendations_" + combination_method + ".csv")
            precision_recall_user_file = os.path.join(model_folder,
                                                      "precision_recall_by_user_" + combination_method + ".csv")
            precision_recall_file = os.path.join(model_folder, "precision_recall_" + combination_method + ".csv")

        save2csv(sf_recommendations, recommendations_file)

        sf_evaluation = gl.recommender.util.precision_recall_by_user(observed_user_items=self.sf_test,
                                                                     recommendations=sf_recommendations,
                                                                     cutoffs=self.cutoffs)

        save2csv(sf_evaluation, precision_recall_user_file)

        sf_evaluation_summary = sf_evaluation.groupby('cutoff',
                                                      {'mean_precision': gl.aggregate.AVG('precision'),
                                                       'mean_recall': gl.aggregate.AVG('recall')}).topk('cutoff',
                                                                                                        reverse=True)

        print sf_evaluation_summary
        save2csv(sf_evaluation_summary, precision_recall_file)

        return {"model_folder": model_folder,
                "recommendations": sf_recommendations,
                "precision_recall_by_user": sf_evaluation,
                "precision_recall": sf_evaluation}

    def combine_by_least_misery(self, user_predictions):
        df_group_users = self.datasets["df_all_groups"][["group_id", "user_id"]].drop_duplicates()
        df_group_user_recs = user_predictions.query("rank >= 7 and rank <= 10").merge(df_group_users,
                                                                                      on="user_id").sort_values(
            ["group_id"])
        combination = df_group_user_recs.groupby(["group_id", "venue_id"]).mean().drop(["user_id", "score"],
                                                                                       axis=1).reset_index()
        combination["rank"] = combination.groupby("group_id")["rank"].rank(ascending=False, method="first")
        combination = combination.sort_values(["group_id", "rank"], ascending=True)
        combination["score"] = combination["rank"]
        return gl.SFrame(combination)

    def combine_by_average_no_misery(self, user_predictions):
        df_group_users = self.datasets["df_all_groups"][["group_id", "user_id"]].drop_duplicates()
        df_group_user_recs = user_predictions.query("rank <= 5").merge(df_group_users, on="user_id").sort_values(
            ["group_id"])
        combination = df_group_user_recs.groupby(["group_id", "venue_id"]).mean().drop(["user_id", "score"],
                                                                                       axis=1).reset_index()
        combination["rank"] = combination.groupby("group_id")["rank"].rank(ascending=False, method="first")
        combination = combination.sort_values(["group_id", "rank"], ascending=True)
        combination["score"] = combination["rank"]
        return gl.SFrame(combination)

    def combine_by_average(self, user_predictions):
        df_group_users = self.datasets["df_all_groups"][["group_id", "user_id"]].drop_duplicates()
        df_group_user_recs = user_predictions.merge(df_group_users, on="user_id").sort_values(["group_id"])
        combination = df_group_user_recs.groupby(["group_id", "venue_id"]).mean().drop(["user_id", "score"],
                                                                                       axis=1).reset_index()
        combination["rank"] = combination.groupby("group_id")["rank"].rank(ascending=False, method="first")
        combination = combination.sort_values(["group_id", "rank"], ascending=True)
        combination["score"] = combination["rank"]
        return gl.SFrame(combination)

    def load_model(self, model_folder):
        group_model = gl.load_model(os.path.join(model_folder, "group_model"))
        user_model = gl.load_model(os.path.join(model_folder, "user_model"))

        group_predictions = pd.read_csv(os.path.join(model_folder, "group_predictions.csv"))
        user_predictions = pd.read_csv(os.path.join(model_folder, "user_predictions.csv"))

        return {"group_model": group_model, "group_predictions": group_predictions,
                "user_model": user_model, "user_predictions": user_predictions}
