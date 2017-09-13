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
from experiment_utils import *
from utils.general_utils import *
import shutil
from termcolor import colored
from models import projected_kde
from joblib import Parallel, delayed
import multiprocessing

matplotlib.style.use('ggplot')
plt.style.use("ggplot")
markers = [".", "*", ">", "o", "v", "^", "<", "s", "p", "h", "H", "D", "d", "|", "_"]
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', multiprocessing.cpu_count())


def main(logger, settings):
    write_running_log("main", logger)
    if settings.get("settings", "mapbox_access_token") == "":
        write_log("No access token specified for MapBox. The maps will not be generated", logger, True, False,
                  print_color="yellow")
    else:
        write_log("MapBox access token specified. Maps will be generated if the token is valid", logger, True, False,
                  print_color="green")

    for ds in [d for d in settings.sections() if d != "settings"]:
        write_log("Validating " + ds.upper(), logger, True, False, print_color="green")
        # Avoid overwriting existing baselines
        experiments_folder = settings.get(ds, "experiments_folder")
        if os.path.exists(experiments_folder):
            write_log("The experiments folder " + experiments_folder + " for " + ds.upper() + " already exists. "
                                                                                              "Existing models will be used. Erase them to recompute",
                      logger, True, False, print_color="yellow")
        else:
            os.mkdir(experiments_folder)

        run_splits = settings.get(ds, "run_splits")
        for split in run_splits.split(","):
            assert split in ["split_by_group", "split_by_cluster", "split_by_cluster_time"], "Invalid split method"

        run_baselines = settings.get(ds, "run_baselines")
        for baseline in run_baselines.split(","):
            assert baseline in ["item_to_item",
                                "content_based_recommender_geo",
                                "content_based_recommender_cat",
                                "content_based_recommender_catgeo",
                                "implicit_matrix_factorization",
                                "sgd_matrix_factorization",
                                "sgd_content_matrix_factorization_geo",
                                "sgd_content_matrix_factorization_cat",
                                "sgd_content_matrix_factorization_catgeo",
                                "popularity_recommender",
                                "popularity_recommender_geo",
                                "popularity_recommender_cat",
                                "popularity_recommender_catgeo"], "Invalid baseline method"

        run_combinations = settings.get(ds, "run_combinations")
        for combination in run_combinations.split(","):
            assert combination in ["average", "average_no_misery", "least_misery"], "Invalid combination method"

    write_log("Everything seems fine for now.", logger, True, False, print_color="green")

    # Everything seems fine! Lets have some fiesta
    for ds in [d for d in settings.sections() if d != "settings"]:
        write_log("*****Running for: " + ds.upper(), logger, True, False, print_color="green")
        experiments_folder = settings.get(ds, "experiments_folder")
        item_test_proportion = settings.getfloat(ds, "item_test_proportion")
        max_test_users_percent = settings.getfloat(ds, "max_test_users_percent")
        run_splits = settings.get(ds, "run_splits")
        run_baselines = settings.get(ds, "run_baselines")
        run_experiment_models = settings.get(ds, "run_experiment_models")
        run_combinations = settings.get(ds, "run_combinations")
        evaluation_cutoffs = [int(c) for c in settings.get(ds, "evaluation_cutoffs").split(",")]
        dataset = DataSet(settings, ds, logger)
        datasets = dataset.load_clean_dataset_city()
        fixed_geonameid = settings.getint(ds, "fixed_geonameid")
        distance_to_group_folder = settings.get(ds, "distance_to_group_folder")
        kernel = settings.get(ds, "kernel")
        bandwidth = settings.getfloat(ds, "bandwidth")

        df_group_centroids = pd.read_csv(os.path.join(distance_to_group_folder,
                                                      str(fixed_geonameid) + "_group_centroids.csv"))

        df_group_centroids = df_group_centroids.rename(
            columns={"cluster": "group_cluster", "cluster_checkins": "group_cluster_checkins",
                     "centroid_lat": "group_centroid_lat", "centroid_long": "group_centroid_long"})

        df_group_centroids = df_group_centroids.drop("group_id", axis=1)

        # Log the dataset sizes
        print_and_log_df_size(datasets, logger)

        # Run the baselines using the different splits
        baseline_results = {}
        splitter = Splitter(logger, datasets, settings, ds)
        for split in run_splits.split(","):
            write_running_log("split: " + split, logger)
            dic_splits = {}

            folder_name = os.path.join(experiments_folder, split)
            # Check if the datasets exist, otherwise load the files
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
                if split == "split_by_group":
                    dic_splits = splitter.split_by_group(item_test_proportion,
                                                         max_test_users_percent *
                                                         len(datasets["df_checkin_group"].group_id.unique()))
                if split == "split_by_cluster":
                    dic_splits = splitter.split_by_cluster(False)

                if split == "split_by_cluster_time":
                    dic_splits = splitter.split_by_cluster(True)

                write_running_log("Storing the datasets", logger)
                store_train_test(folder_name, dic_splits["train"], dic_splits["test"],
                                 dic_splits["train_user"], dic_splits["test_user"])

            else:
                write_running_log("Loading existing datasets", logger)
                dic_splits = load_train_test(folder_name)

            print_and_log_df_size(dic_splits, logger)

            baseline_executor = Baselines(logger, settings, datasets, ds, folder_name, dic_splits["train"],
                                          dic_splits["test"], dic_splits["train_user"], dic_splits["test_user"],
                                          datasets["df_venues_content"], evaluation_cutoffs)

            # This will run the baselines. We train using all the group training and test in the all the group testing
            for baseline in run_baselines.split(","):
                # Let's name the baseline so we know what parameters where passed
                write_running_log("baseline: " + baseline, logger)

                model_folder = os.path.join(folder_name, baseline)

                if not os.path.exists(model_folder):
                    os.mkdir(model_folder)
                    if baseline == "item_to_item":
                        baseline_results[baseline] = baseline_executor.item_to_item_recommender(model_folder)
                    if baseline == "content_based_recommender_geo":
                        baseline_results[baseline] = baseline_executor.content_based_recommender(model_folder, "geo")
                    if baseline == "content_based_recommender_cat":
                        baseline_results[baseline] = baseline_executor.content_based_recommender(model_folder, "cat")
                    if baseline == "content_based_recommender_catgeo":
                        baseline_results[baseline] = baseline_executor.content_based_recommender(model_folder, "catgeo")
                    if baseline == "implicit_matrix_factorization":
                        baseline_results[baseline] = baseline_executor.implicit_matrix_factorization(model_folder)
                    if baseline == "sgd_matrix_factorization":
                        baseline_results[baseline] = baseline_executor.sgd_matrix_factorization(model_folder)
                    if baseline == "sgd_content_matrix_factorization_geo":
                        baseline_results[baseline] = baseline_executor.sgd_content_matrix_factorization(model_folder,
                                                                                                        "geo")
                    if baseline == "sgd_content_matrix_factorization_cat":
                        baseline_results[baseline] = baseline_executor.sgd_content_matrix_factorization(model_folder,
                                                                                                        "cat")
                    if baseline == "sgd_content_matrix_factorization_catgeo":
                        baseline_results[baseline] = baseline_executor.sgd_content_matrix_factorization(model_folder,
                                                                                                        "catgeo")
                    if baseline == "popularity_recommender":
                        baseline_results[baseline] = baseline_executor.popularity_recommender(model_folder,"")
                    if baseline == "popularity_recommender_geo":
                        baseline_results[baseline] = baseline_executor.popularity_recommender(model_folder,
                                                                                              "geo")
                    if baseline == "popularity_recommender_cat":
                        baseline_results[baseline] = baseline_executor.popularity_recommender(model_folder,
                                                                                              "cat")
                    if baseline == "popularity_recommender_catgeo":
                        baseline_results[baseline] = baseline_executor.popularity_recommender(model_folder, "catgeo")
                else:
                    # Load the models
                    baseline_results[baseline] = baseline_executor.load_model(model_folder)

                write_running_log("Storing and evaluating " + baseline, logger)
                # Store the model
                parameters = OrderedDict({"baseline": baseline,
                                          "dataset": ds,
                                          "item_test_proportion": item_test_proportion,
                                          "max_test_users_percent": max_test_users_percent,
                                          "split": split})

                group_results = baseline_executor.evaluate_and_store(model_folder,
                                                                     parameters,
                                                                     model=baseline_results[baseline]["group_model"])

                # Combine the predictions
                for combination in run_combinations.split(","):
                    if combination == "average":
                        combined_predictions = baseline_executor.combine_by_average(
                            baseline_results[baseline]["user_predictions"])
                    if combination == "average_no_misery":
                        combined_predictions = baseline_executor.combine_by_average_no_misery(
                            baseline_results[baseline]["user_predictions"])
                    if combination == "least_misery":
                        combined_predictions = baseline_executor.combine_by_least_misery(
                            baseline_results[baseline]["user_predictions"])

                    combined_results = baseline_executor.evaluate_and_store(model_folder,
                                                                            parameters=parameters,
                                                                            recommendations=combined_predictions,
                                                                            combination_method=combination)
            # This will run the evaluation per group and per cluster
            for experiment_model in run_experiment_models.split(","):
                # Let's name the baseline so we know what parameters where passed
                write_running_log("experiment model: " + experiment_model, logger)
                experiments_model_folder = os.path.join(folder_name, experiment_model)

                if not os.path.exists(experiments_model_folder):
                    os.mkdir(experiments_model_folder)

                if experiment_model == "projected_kde":
                    # For this experiment we load the existing baselines and multiply the geographical score
                    # with the recommendation

                    # First let's find the kde for each group and cluster
                    df_venues = datasets["df_venues"]

                    df_venues_ids = np.unique(np.append(datasets["df_checkins"].venue_id.unique(),
                                                        datasets["df_checkin_group"].venue_id.unique()))

                    possible_venues = df_venues[df_venues.venue_id.isin(df_venues_ids)]

                    write_running_log("getting group clusters KDEs", logger)

                    group_train = dic_splits["train"].to_dataframe().merge(df_venues, on="venue_id")
                    group_test = dic_splits["test"].to_dataframe().merge(df_venues, on="venue_id")

                    group_train = group_train.merge(df_group_centroids, on="checkin_id", how="left")
                    group_test = group_test.merge(df_group_centroids, on="checkin_id", how="left")

                    group_train = group_train[group_train.group_id.isin(group_test.group_id)]
                    grouped_train = group_train.groupby("group_id")

                    parallel_results = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=5)(
                        delayed(projected_kde.get_group_clusters_kde)(group_id,
                                                                      checkins_train,
                                                                      group_test.query("group_id == @group_id"),
                                                                      possible_venues,
                                                                      kernel, bandwidth
                                                                      )for group_id, checkins_train in grouped_train)
                    dic_group_cluster_venues = {}
                    for group_result in parallel_results:
                        dic_cluster_venues = {}
                        for group_cluster_result in group_result["all_possible_venues_cluster"]:
                            dic_cluster_venues[group_cluster_result[0]] = group_cluster_result[1]
                        dic_group_cluster_venues[group_result["group_id"]] = dic_cluster_venues

                    for baseline in run_baselines.split(","):
                        # Let's name the baseline so we know what parameters where passed
                        write_running_log("combining projected_kde with baseline: " + baseline, logger)
                        model_name = baseline + "_projected_kde"
                        model_folder = os.path.join(folder_name, baseline)
                        combined_model_folder = os.path.join(folder_name, model_name)

                        if not os.path.exists(combined_model_folder):
                            os.mkdir(combined_model_folder)

                        if os.path.exists(model_folder):
                            baseline_model = baseline_executor.load_model(model_folder)
                            group_model = baseline_model["group_model"]
                            # We take the venues possible venues from the

                            # combined_model_results = projected_kde.combine_model(model_name,
                            #                                                      group_model,
                            #                                                      dic_splits["train"],
                            #                                                      dic_splits["test"],
                            #                                                      possible_venues,
                            #                                                      df_venues,
                            #                                                      df_group_centroids,
                            #                                                      evaluation_cutoffs,
                            #                                                      kernel,
                            #                                                      bandwidth,
                            #                                                      combined_model_folder,
                            #                                                      logger)

                            combined_model_results = projected_kde.combine_model_with_kde(model_name,
                                                                                          group_model,
                                                                                          group_train,
                                                                                          group_test,
                                                                                          dic_group_cluster_venues,
                                                                                          df_venues,
                                                                                          evaluation_cutoffs,
                                                                                          kernel,
                                                                                          bandwidth,
                                                                                          combined_model_folder,
                                                                                          logger)
                        else:
                            write_log("No baseline model" + baseline + ". Skipping", logger,
                                      True, False, print_color="red")

    write_log("Main process completed", logger, True, False)


if __name__ == '__main__':
    usage = """
        python run_baselines.py
        The following parameters are mandatory:
            -c --config     The path to the configuration file for running the experiments
    """
    parser = argparse.ArgumentParser(description="Run the experiments.")
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    if not os.path.isfile(args.config):
        print usage
        exit("The configuration file does not exists")
    else:
        # Load the settings
        settings = ConfigParser.ConfigParser()
        settings._interpolation = configparser.ExtendedInterpolation()
        settings.read(args.config)

        # Set-up the logging
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            filename=args.config.split("/")[-1].replace(".config", ".log"))
        logger = logging.getLogger(__name__)
        write_log('Ready to start the experiments', logger, True, False)

        main(logger, settings)
