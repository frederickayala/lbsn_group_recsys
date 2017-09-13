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
from analysis_utils import *
from utils.general_utils import *
import shutil
from termcolor import colored

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

    # Create the analysis folder if it doesn't exist
    analysis_folder = settings.get("settings", "analysis_folder")
    if not os.path.exists(analysis_folder):
        os.mkdir(analysis_folder)

    # Validations
    for ds in [d for d in settings.sections() if d != "settings"]:
        write_log("Validating " + ds.upper(), logger, True, False, print_color="green")
        dataset_folder = settings.get(ds, "dataset_folder")
        assert os.path.exists(dataset_folder)
        if not "global" in ds:
            assert settings.get(ds,"centroid") != ""
            assert settings.getint(ds, "offset_in_miles") > 0

    write_log("Everything seems fine for now", logger, True, False, print_color="green")

    # Everything seems fine! Lets have some fiesta
    data_dic = {}
    for ds in [d for d in settings.sections() if d != "settings"]:
        write_log("*****Running for: " + ds.upper(), logger, True, False, print_color="green")
        dataset = DataSet(settings, ds, logger)
        datasets = dataset.load_global_dataset()

        # Log the dataset sizes
        print_and_log_df_size(datasets, logger)
        data_dic[ds] = datasets

    analysis_utils = Analysis(data_dic, settings, logger)
    analysis_utils.get_summary()

    # Analyze the inter time and distance for individuals and groups
    dic_inter_checkins = analysis_utils.analyze_inter_checkins()

    # Print a city summary after cleaning the dataset
    analysis_utils.get_city_summary()

    # Analyze the inter time and distance distributions for individuals and groups
    df_user_inter_checkin = dic_inter_checkins["user_inter_checkin_clean"]
    df_group_inter_checkin = dic_inter_checkins["group_inter_checkin_clean"]
    df_venues_categories_all = dic_inter_checkins["venues_categories_all"]
    distribution_folders = [f for f in os.listdir(analysis_utils.analysis_folder) if "distribution" in f and "." not in f]
    if len(distribution_folders) == 0:
        # For individuals
        user_checkins = df_user_inter_checkin.groupby("user_id").count()["checkin_id"].dropna().values
        user_inter_distance = df_user_inter_checkin.inter_distance.dropna().values
        user_categories = df_user_inter_checkin.groupby("second_category").count()["checkin_id"].dropna().values
        user_venues = df_user_inter_checkin.groupby("venue_id").count()["checkin_id"].dropna().values

        dist_user_checkins = analysis_utils.analyze_distribution(user_checkins, "user_checkins", "Users' check-ins count",
                                                                 1, max(user_checkins))
        dist_user_inter_distance = analysis_utils.analyze_distribution(user_inter_distance, "user_inter_distance",
                                                                       "Users's distance between check-ins (kms.)", 1, 50)
        dist_user_categories = analysis_utils.analyze_distribution(user_categories, "user_categories",
                                                                   "Users' categories check-ins", 1, max(user_categories))
        dist_user_venues = analysis_utils.analyze_distribution(user_venues, "user_venues", "Users' venues check-ins", 1,
                                                               max(user_venues))

        # For groups
        group_checkins = df_group_inter_checkin.groupby("group_id").count()["checkin_id"].dropna().values
        group_inter_distance = df_group_inter_checkin.inter_distance.dropna().values
        group_categories = df_group_inter_checkin.groupby("second_category").count()["checkin_id"].dropna().values
        group_venues = df_group_inter_checkin.groupby("venue_id").count()["checkin_id"].dropna().values

        dist_group_checkins = analysis_utils.analyze_distribution(group_checkins, "group_checkins",
                                                                  "Groups' check-ins count", 1, max(group_checkins))
        dist_group_inter_distance = analysis_utils.analyze_distribution(group_inter_distance, "group_inter_distance",
                                                                        "Groups's distance between check-ins (kms.)", 1,
                                                                        max(group_inter_distance))
        dist_group_categories = analysis_utils.analyze_distribution(group_categories, "group_categories",
                                                                    "Groups' categories check-ins", 1,
                                                                    max(group_categories))
        dist_group_venues = analysis_utils.analyze_distribution(group_venues, "group_venues",
                                                                "Groups' venues check-ins", 1, max(group_venues))

        for dist in [dist_group_checkins, dist_group_inter_distance, dist_group_categories, dist_group_venues,
                     dist_user_checkins, dist_user_inter_distance, dist_user_categories, dist_user_venues]:
            name = dist["distributions_comparison"].name.unique()[0]
            best = get_best_fit(dist["distributions_comparison"], "power_law", 0, 0)
            write_log("%s follows %s, R:%f p: %f" % (name, best[0], best[1], best[2]), logger, True, False)
    else:
        write_log("*****Showing existing distributions. Erase the folders if you want to re-compute them",
                  logger, True, False, print_color="green")
        for dist in distribution_folders:
            distributions_comparison = pd.read_csv(os.path.join(
                os.path.join(analysis_utils.analysis_folder,dist),"distribution_comparison.csv"))
            name = distributions_comparison.name.unique()[0]
            best = get_best_fit(distributions_comparison, "power_law", 0, 0)
            write_log("%s follows %s, R:%f p: %f" % (name, best[0], best[1], best[2]), logger, True, False)

    # Compare top categories per city
    pref_city_res = analysis_utils.compare_preferences_per_city()

    # Compare top user vs group
    pref_city_res = analysis_utils.compare_preferences_per_user()

    # Analyze distances to groups
    distances_per_city = analysis_utils.analize_distance_to_group()

    write_log("Main process completed", logger, True, False)

if __name__ == '__main__':
    usage = """
        python run_analysis.py
        The following parameters are mandatory:
            -c --config     The path to the configuration file for running the analysis
    """
    parser = argparse.ArgumentParser(description="Run the baselines.")
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
        write_log('Ready to start the analysis', logger, True, False)

        main(logger, settings)