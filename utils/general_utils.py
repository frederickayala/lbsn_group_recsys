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
import rtree
from rtree import index
from collections import deque

gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', multiprocessing.cpu_count())

# Taken from http://stackoverflow.com/questions/10473852/convert-latitude-and-longitude-to-point-in-3d-space
def llh_to_ecef(venue_id, lat, lon, alt):
    # see http://www.mathworks.de/help/toolbox/aeroblks/llatoecefposition.html

    rad = np.float64(6378137.0)        # Radius of the Earth (in meters)
    f = np.float64(1.0/298.257223563)  # Flattening factor WGS84 Model
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    ff = (1.0-f)**2
    c = 1/np.sqrt(cos_lat**2 + ff * sin_lat**2)
    s = c * ff

    x = (rad * c + alt)*cos_lat * np.cos(lon)
    y = (rad * c + alt)*cos_lat * np.sin(lon)
    z = (rad * s + alt)*sin_lat

    return pd.Series(OrderedDict({"venue_id": venue_id, "projected_x":x, "projected_y":y, "projected_z":z}))


def write_log(message, logger, print_too, is_error, print_color="magenta"):
    logger.info(message) if not is_error else logger.error(message)
    if print_too:
        if not is_error:
            print colored(message, print_color)
        else:
            print colored(message, "red")


def print_and_log_df_size(df, logger):
    for l in df.keys():
        if not df[l] is None:
            write_log(" Total rows in " + l + ": " + str(len(df[l])), logger, True, False)


def write_running_log(method, logger):
    write_log("********** Executing " + method, logger, True, False, print_color="blue")


def save2csv(df,filename):
    df = df.to_dataframe() if not isinstance(df, pd.DataFrame) else df
    df.to_csv(filename,index=False,encoding="utf8",quoting=csv.QUOTE_ALL,quotechar="\"")


def csv2pandas(filename):
    return pd.read_csv(filename,encoding="utf8",quoting=csv.QUOTE_ALL,quotechar="\"")


def save_plot(fig,save_to):
    if os.path.isfile(save_to):
        os.remove(save_to)
    fig.savefig(save_to + ".png",format="png", bbox_inches="tight",dpi=300)
    fig.savefig(save_to + ".tif",format="tif", bbox_inches="tight",dpi=300)
    fig.savefig(save_to + ".pdf",format="pdf", bbox_inches="tight",dpi=300)


# Code from
# http://stackoverflow.com/questions/1648917/given-a-latitude-and-longitude-and-distance-i-want-to-find-a-bounding-box
class BoundingBox(object):
    def __init__(self, *args, **kwargs):
        self.lat_min = None
        self.lon_min = None
        self.lat_max = None
        self.lon_max = None


def get_bounding_box(latitude_in_degrees, longitude_in_degrees, half_side_in_miles):
    assert half_side_in_miles > 0
    assert -90.0 <= latitude_in_degrees <= 90.0
    assert -180.0 <= longitude_in_degrees <= 180.0

    half_side_in_km = half_side_in_miles * 1.609344
    lat = math.radians(latitude_in_degrees)
    lon = math.radians(longitude_in_degrees)

    radius  = 6371
    # Radius of the parallel at given latitude
    parallel_radius = radius*math.cos(lat)

    lat_min = lat - half_side_in_km/radius
    lat_max = lat + half_side_in_km/radius
    lon_min = lon - half_side_in_km/parallel_radius
    lon_max = lon + half_side_in_km/parallel_radius
    rad2deg = math.degrees

    box = BoundingBox()
    box.lat_min = rad2deg(lat_min)
    box.lon_min = rad2deg(lon_min)
    box.lat_max = rad2deg(lat_max)
    box.lon_max = rad2deg(lon_max)

    return (box)


class Plotting:
    """
            The methods in this class help to visualize information in maps and in charts.
    """
    def __init__(self,mapbox_token=""):
        self.mapbox_token = mapbox_token

    def mapping_static(self, filepath, min_lat, min_long, max_lat, max_long):
        service = Static(
            access_token=self.mapbox_token)
        square = {
            "type": "Feature",
            "properties": {
                "stroke": "#0000ff",
                "stroke-width": 2,
                "stroke-opacity": .4,
                "fill": "#0000ff",
                "fill-opacity": 0.2
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [min_long, min_lat],
                    [min_long, max_lat],
                    [max_long, max_lat],
                    [max_long, min_lat],
                    [min_long, min_lat]
                ]]}}
        response = service.image('mapbox.streets', features=[square])

        with open(filepath,'wb') as output:
            output.write(response.content)


class DataSet:
    """
        Class for methods to load and store the datasets
    """

    def __init__(self, settings, dataset, logger):
        self.settings = settings
        self.dataset = dataset
        self.logger = logger
        self.datasets = {}

    def fix_city_country(self, venue, df_cities_to_index, idx):
        res = OrderedDict({})
        res["venue_id"] = venue.venue_id
        nearest_city = df_cities_to_index.loc[list(idx.nearest((venue.venue_lat, venue.venue_long)))[0]]
        res["fixed_geonameid"] = nearest_city.name
        res["fixed_city"] = nearest_city["name"]
        res["fixed_country"] = nearest_city["country_code"]
        return pd.Series(res)

    def fix_cities(self, df_venues, save_to):
        write_running_log("fix_cities", self.logger)
        # Fix the city field for the venues
        df_cities = pd.read_csv(self.settings.get("settings", "geonames_cities_file"), sep="\t",
                                names=["geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
                                       "feature_class", "feature_code", "country_code", "cc2", "admin1_code",
                                       "admin2_code", "admin3_code", "admin4_code", "population", "elevation", "dem",
                                       "timezone", "modification_date"])

        # We take the capital and main cities
        geonames_filter_types = self.settings.get("settings", "geonames_filter_types").split(",")

        df_cities_to_index = df_cities[df_cities.feature_code.isin(geonames_filter_types)]
        df_cities_to_index = df_cities_to_index.sort_values("population", ascending=False)
        df_cities_to_index = df_cities_to_index.set_index("geonameid")

        idx = index.Index()
        for i, v in df_cities_to_index.iterrows():
            idx.insert(i, (v.latitude, v.longitude))

        df_venues_fixed_city = df_venues.apply(lambda x: self.fix_city_country(x, df_cities_to_index, idx), axis=1)
        save2csv(df_venues_fixed_city, save_to)
        return df_venues.merge(df_venues_fixed_city, on="venue_id")

    def load_global_dataset(self):
        """
            This method loads the datasets and takes the latest version of the user and the venue
        """
        dataset_folder = self.settings.get(self.dataset, "dataset_folder")
        # Load the files
        df_users = pd.read_csv(os.path.join(dataset_folder, "df_users.csv"))
        df_users = df_users.drop_duplicates()

        df_checkins = pd.read_csv(os.path.join(dataset_folder, "df_checkins.csv"))
        df_checkins = df_checkins.drop_duplicates()

        df_checkin_group = pd.read_csv(os.path.join(dataset_folder, "df_checkin_group.csv"))
        df_checkin_group = df_checkin_group.drop_duplicates()

        df_checkins_with = pd.read_csv(os.path.join(dataset_folder, "df_checkins_with.csv"))
        df_checkins_with = df_checkins_with.drop_duplicates()

        df_venues = pd.read_csv(os.path.join(dataset_folder, "df_venues.csv"))
        df_venues = df_venues.drop_duplicates()

        df_venues_categories = pd.read_csv(os.path.join(dataset_folder, "df_venues_categories.csv"))
        df_venues_categories = df_venues_categories.drop_duplicates()

        df_group_user_checkins = None

        # Use the latest user information
        df_users = df_users.groupby("user_id").last().reset_index()

        # Use the latest check-ins information
        df_checkins = df_checkins.groupby("checkin_id").last().reset_index()

        # Use the latest venues information
        df_venues = df_venues.sort_values(["venue_id", "venue_checkinsCount"], ascending=True).groupby(
            "venue_id").last().reset_index()

        # Remove irrelevant categories
        write_log("***** Removing venues of irrelevant categories", self.logger, True, False, print_color="green")
        categories_file = self.settings.get("settings",  "categories_file")
        remove_categories = self.settings.get("settings", "remove_categories").split(",")

        df_checkins_merged = df_checkins.merge(df_venues_categories, on="venue_id")
        df_checkins_merged = df_checkins_merged.groupby("checkin_id").last().reset_index()
        df_categories = df_checkins_merged.groupby("category_name")[["checkin_id"]].count().reset_index()

        df_4sq_categories = pd.read_csv(self.settings.get("settings", "categories_file"))
        df_4sq_categories = df_4sq_categories.rename(columns={"child": "category_name"})
        df_categories = df_categories.merge(df_4sq_categories, on="category_name")
        df_categories = df_categories.sort_values(["checkin_id"], ascending=False)

        df_remove = []
        for remove in remove_categories:
            df_remove.append(df_categories[df_categories.full.apply(lambda x: remove in x)])

        df_remove = pd.concat(df_remove)
        remove_categories = df_remove.category_name.unique()
        checkins_to_remove = df_checkins_merged[df_checkins_merged.category_name.isin(remove_categories)].checkin_id.values

        write_log("***** Total checkins to be removed: " + str(len(checkins_to_remove)), self.logger, True, False,
                  print_color="green")

        df_checkins = df_checkins[~df_checkins.checkin_id.isin(checkins_to_remove)]
        df_checkin_group = df_checkin_group[~df_checkin_group.checkin_id.isin(checkins_to_remove)]
        df_checkins_with = df_checkins_with[~df_checkins_with.checkin_id.isin(checkins_to_remove)]
        df_venues = df_venues[df_venues.venue_id.isin(df_checkins.venue_id)]
        df_venues_categories = df_venues_categories[df_venues_categories.venue_id.isin(df_checkins.venue_id)]

        # Fix the cities
        df_venues_fixed_city_file = os.path.join(self.settings.get("settings", "analysis_folder"),
                                                 "df_venues_fixed_city.csv")

        if os.path.exists(df_venues_fixed_city_file):
            write_log("***** Using existing fixed cities for venues, erase df_venues_fixed_city.csv to re-compute",
                      self.logger, True, False, print_color="green")
            df_venues_fixed_city = pd.read_csv(df_venues_fixed_city_file)
            df_venues = df_venues.merge(df_venues_fixed_city, on="venue_id")
        else:
            df_venues = self.fix_cities(df_venues, df_venues_fixed_city_file)

        # Remove the users that move around too much
        df_checkins_venue = df_checkins.merge(df_venues, on="venue_id")
        df_std = df_checkins_venue.groupby("user_id")[["venue_lat", "venue_long"]].aggregate(np.std).dropna().reset_index()

        df_std["lat_nth"] = pd.qcut(df_std.venue_lat, 8, labels=False)
        df_std["long_nth"] = pd.qcut(df_std.venue_long, 8, labels=False)

        remove_nthile_from = self.settings.getint("settings", "remove_nthile_from")

        df_checkins_venue_filtered = df_checkins_venue[
            df_checkins_venue.user_id.isin(df_std.query("lat_nth < @remove_nthile_from and "
                                                        "long_nth < @remove_nthile_from").user_id)]

        venues_clean = df_checkins_venue_filtered.venue_id.unique()
        venues_all = df_checkins_venue.venue_id.unique()

        main_city = df_checkins_venue.groupby("venue_city").count().sort_values("checkin_id", ascending=False)
        main_city = main_city.head(1).index.values[0]

        # Remove the check-ins and venues that that are not relevant
        checkins_all = df_checkins.checkin_id.unique()
        df_checkins = df_checkins[df_checkins.venue_id.isin(venues_clean)]
        checkins_clean = df_checkins.checkin_id.unique()
        df_checkin_group = df_checkin_group[df_checkin_group.checkin_id.isin(checkins_clean)]
        df_checkins_with = df_checkins_with[df_checkins_with.checkin_id.isin(checkins_clean)]
        df_venues = df_venues[df_venues.venue_id.isin(venues_clean)]
        df_venues_categories = df_venues_categories[
            df_venues_categories.venue_id.isin(venues_clean)]

        if "projected_x" not in df_venues.columns:
            # Complementing the venues with the 3D cartesian coordinates
            write_running_log("lat_long_to_cartesian", self.logger)
            df_projected_venues = df_venues.apply(lambda x: llh_to_ecef(x.venue_id, x.venue_lat, x.venue_long, 0),
                                                  axis=1)
            df_venues = df_venues.merge(df_projected_venues, on="venue_id")

        if "main_category" not in df_venues.columns:
            write_running_log("fixing categories", self.logger)
            df_category_tree = pd.read_csv(self.settings.get("settings", "categories_file"))
            df_category_tree = df_category_tree.rename(columns={"child": "category_name"})
            df_category_tree["category_tree"] = df_category_tree.full.apply(
                lambda x: [c for c in x.split("|") if c != "root"])

            df_venues_categories = df_venues_categories.merge(df_category_tree[["category_name", "category_tree"]],
                                                              on="category_name")

            df_venues = df_venues.merge(df_venues_categories, on="venue_id").groupby("venue_id").last().reset_index()
            df_venues["category_tree_size"] = df_venues.category_tree.apply(lambda x: len(x))
            df_venues = df_venues[df_venues.category_tree_size > 1]
            df_venues["main_category"] = df_venues["category_tree"].apply(lambda x: x[0])
            df_venues["second_category"] = df_venues["category_tree"].apply(lambda x: x[1])
            df_venues["third_category"] = df_venues["category_tree"].apply(lambda x: x[2] if len(x) > 2 else None)
            df_venues["fourth_category"] = df_venues["category_tree"].apply(lambda x: x[3] if len(x) > 3 else None)
            df_venues["category_tree"] = df_venues["category_tree"].apply(lambda x: "|".join(x))
        else:
            assert "main_category" in df_venues.columns and "second_category" in df_venues.columns and \
                   "third_category" in df_venues.columns and "fourth_category" in df_venues.columns, \
                "Check the categories in the venues file"

        venues_content = ["venue_id", "category_tree", "projected_x", "projected_y", "projected_z"]
        df_venues_content = df_venues[venues_content].drop_duplicates()

        # Create the content for the venues
        # venues_content = ["venue_cc", "venue_city", "venue_id", "venue_tipCount", "venue_usersCount"]
        # df_venues.venue_cc = df_venues.venue_cc.astype(str)
        # df_venues.venue_city = df_venues.venue_city.astype(str)
        # df_venues_categories.loc[:, "value"] = 1
        # df_venues_pivot = df_venues_categories.pivot(index="venue_id", columns="category_id",
        #                                              values="value").reset_index()
        # df_venues_content = df_venues[venues_content].merge(df_venues_pivot, on="venue_id")
        #
        # df_venues_content = df_venues_content[df_venues_content.venue_id.isin(df_venues.venue_id)]

        # Print a summary of the dataset
        write_log("The global dataset " + self.dataset, self.logger, True, False)

        write_log("Total number of venues:" + str(len(df_venues.venue_id)), self.logger, True, False)
        write_log("Total number of checkins:" + str(len(df_checkins.checkin_id)), self.logger, True, False)

        self.datasets = {"df_users": df_users,
                "df_checkins": df_checkins,
                "df_checkin_group": df_checkin_group,
                "df_checkins_with": df_checkins_with,
                "df_venues": df_venues,
                "df_venues_categories": df_venues_categories,
                "df_venues_content": df_venues_content,
                "df_group_user_checkins": df_group_user_checkins}

        return self.datasets

    def load_clean_dataset(self):
        """
            This method loads the datasets and takes the latest version of the user and the venue
        """
        dataset_folder = self.settings.get(self.dataset, "dataset_folder")
        # Load the files
        df_users = pd.read_csv(os.path.join(dataset_folder, "users_clean.csv"))
        df_users = df_users.drop_duplicates()

        df_checkins = pd.read_csv(os.path.join(dataset_folder, "checkins_clean.csv"))
        df_checkins = df_checkins.drop_duplicates()

        df_checkin_group = pd.read_csv(os.path.join(dataset_folder, "checkin_group_clean.csv"))
        df_checkin_group = df_checkin_group.drop_duplicates()

        df_checkins_with = pd.read_csv(os.path.join(dataset_folder, "checkins_with_clean.csv"))
        df_checkins_with = df_checkins_with.drop_duplicates()

        df_venues = pd.read_csv(os.path.join(dataset_folder, "venues_clean.csv"))
        df_venues = df_venues.drop_duplicates()

        df_venues_categories = pd.read_csv(os.path.join(dataset_folder, "venues_categories_clean.csv"))
        df_venues_categories = df_venues_categories.drop_duplicates()

        df_group_user_checkins = None

        # Use the latest user information
        df_users = df_users.groupby("user_id").last().reset_index()

        # Use the latest check-ins information
        df_checkins = df_checkins.groupby("checkin_id").last().reset_index()

        # Use the latest venues information
        df_venues = df_venues.sort_values(["venue_id", "venue_checkinsCount"], ascending=True).groupby(
            "venue_id").last().reset_index()

        # Create the content for the venues
        venues_content = ["venue_cc", "venue_city", "venue_id", "venue_tipCount", "venue_usersCount"]
        df_venues.venue_cc = df_venues.venue_cc.astype(str)
        df_venues.venue_city = df_venues.venue_city.astype(str)
        df_venues_categories.loc[:, "value"] = 1
        df_venues_pivot = df_venues_categories.pivot(index="venue_id", columns="category_id",
                                                     values="value").reset_index()
        df_venues_content = df_venues[venues_content].merge(df_venues_pivot, on="venue_id")

        df_venues_content = df_venues_content[df_venues_content.venue_id.isin(df_venues.venue_id)]

        # Print a summary of the dataset
        write_log("The dataset " + self.dataset, self.logger, True, False)

        write_log("Total number of venues:" + str(len(df_venues.venue_id)), self.logger, True, False)
        write_log("Total number of checkins:" + str(len(df_checkins.checkin_id)), self.logger, True, False)

        self.datasets = {"df_users": df_users,
                "df_checkins": df_checkins,
                "df_checkin_group": df_checkin_group,
                "df_checkins_with": df_checkins_with,
                "df_venues": df_venues,
                "df_venues_categories": df_venues_categories,
                "df_venues_content": df_venues_content,
                "df_group_user_checkins": df_group_user_checkins}

        return self.datasets

    def load_clean_dataset_city(self):
        """
            This method loads the datasets and takes the latest version of the user and the venue
        """
        dataset_folder = self.settings.get(self.dataset, "dataset_folder")
        fixed_geonameid = self.settings.getint(self.dataset, "fixed_geonameid")
        # Load the files
        df_users = pd.read_csv(os.path.join(dataset_folder, "users_clean.csv"))
        df_users = df_users.drop_duplicates()

        df_checkins = pd.read_csv(os.path.join(dataset_folder, "checkins_clean.csv"))
        df_checkins = df_checkins.drop_duplicates()

        df_checkin_group = pd.read_csv(os.path.join(dataset_folder, "checkin_group_clean.csv"))
        df_checkin_group = df_checkin_group.drop_duplicates()

        df_checkins_with = pd.read_csv(os.path.join(dataset_folder, "checkins_with_clean.csv"))
        df_checkins_with = df_checkins_with.drop_duplicates()

        df_venues = pd.read_csv(os.path.join(dataset_folder, "venues_clean.csv"))
        df_venues = df_venues.drop_duplicates()

        df_venues_categories = pd.read_csv(os.path.join(dataset_folder, "venues_categories_clean.csv"))
        df_venues_categories = df_venues_categories.drop_duplicates()

        df_group_user_checkins = None

        # Use the latest user information
        df_users = df_users.groupby("user_id").last().reset_index()

        # Use the latest check-ins information
        df_checkins = df_checkins.groupby("checkin_id").last().reset_index()

        # Use the latest venues information
        df_venues = df_venues.sort_values(["venue_id", "venue_checkinsCount"], ascending=True).groupby(
            "venue_id").last().reset_index()

        df_checkins_venue = df_checkins.merge(df_venues, on="venue_id")

        df_checkins_venue_filtered = df_checkins_venue.query(
            "fixed_geonameid == @fixed_geonameid")

        venues_clean = df_checkins_venue_filtered.venue_id.unique()
        venues_all = df_checkins_venue.venue_id.unique()

        main_city = df_checkins_venue_filtered.groupby("venue_city").count().sort_values("checkin_id", ascending=False)
        main_city = main_city.head(1).index.values[0]

        # Remove the check-ins and venues that that are not relevant
        checkins_all = df_checkins.checkin_id.unique()
        df_checkins = df_checkins[df_checkins.venue_id.isin(venues_clean)]
        checkins_clean = df_checkins.checkin_id.unique()
        df_checkin_group = df_checkin_group[df_checkin_group.checkin_id.isin(checkins_clean)]
        df_checkins_with = df_checkins_with[df_checkins_with.checkin_id.isin(checkins_clean)]
        df_venues = df_venues[df_venues.venue_id.isin(venues_clean)]
        df_venues_categories = df_venues_categories[
            df_venues_categories.venue_id.isin(venues_clean)]

        if "projected_x" not in df_venues.columns:
            # Complementing the venues with the 3D cartesian coordinates
            write_running_log("lat_long_to_cartesian", self.logger)
            df_projected_venues = df_venues.apply(lambda x: llh_to_ecef(x.venue_id, x.venue_lat, x.venue_long, 0), axis=1)
            df_venues = df_venues.merge(df_projected_venues, on="venue_id")

        if "main_category" not in df_venues.columns:
            # Complementing the venues with the 3D cartesian coordinates
            df_category_tree = pd.read_csv(self.settings.get("settings","categories_file"))
            df_category_tree = df_category_tree.rename(columns={"child": "category_name"})
            df_category_tree["category_tree"] = df_category_tree.full.apply(
                lambda x: [c for c in x.split("|") if c != "root"])

            df_venues_categories = df_venues_categories.merge(df_category_tree[["category_name","category_tree"]],
                                                              on="category_name")

            df_venues = df_venues.merge(df_venues_categories, on="venue_id").groupby("venue_id").last().reset_index()
            df_venues["category_tree_size"] = df_venues.category_tree.apply(lambda x: len(x))
            df_venues = df_venues[df_venues.category_tree_size > 1]
            df_venues["main_category"] = df_venues["category_tree"].apply(lambda x: x[0])
            df_venues["second_category"] = df_venues["category_tree"].apply(lambda x: x[1])
            df_venues["third_category"] = df_venues["category_tree"].apply(lambda x: x[2] if len(x) > 2 else None)
            df_venues["fourth_category"] = df_venues["category_tree"].apply(lambda x: x[3] if len(x) > 3 else None)
            df_venues["category_tree"] = df_venues["category_tree"].apply(lambda x: "|".join(x))
        else:
            assert "main_category" in df_venues.columns and "second_category" in df_venues.columns and \
                   "third_category" in df_venues.columns and "fourth_category" in df_venues.columns, \
                "Check the categories in the venues file"

        # Create the content for the venues

        venues_content = ["venue_id", "category_tree", "projected_x", "projected_y", "projected_z"]
        df_venues_content = df_venues[venues_content].drop_duplicates()
        # df_venues.venue_cc = df_venues.venue_cc.astype(str)
        # df_venues.venue_city = df_venues.venue_city.astype(str)
        # df_venues_categories.loc[:, "value"] = 1
        # df_venues_pivot = df_venues_categories.pivot(index="venue_id", columns="category_id",
        #                                              values="value").reset_index()
        # df_venues_content = df_venues.merge(df_venues_categories, on="venue_id").groupby("venue_id").last().reset_index()
        # df_venues_content = df_venues_content[venues_content]
        # df_venues_content['category_id'] = pd.Categorical(df_venues_content['category_id']).codes
        # df_venues_content = df_venues_content[df_venues_content.venue_id.isin(df_venues.venue_id)]

        # Print a summary of the dataset
        write_log("The main city is " + main_city, self.logger, True, False)

        write_log("Only " + str(len(venues_clean)) + " ("
                  + str(round(100 * float(len(venues_clean)) / len(venues_all), 0)) + "%)"
                  + " venues out of " + str(len(venues_all)) + " are inside " + main_city, self.logger, True, False)

        write_log("Only " + str(len(checkins_clean)) + " ("
                  + str(round(100 * float(len(checkins_clean)) / len(checkins_all), 0)) + "%)"
                  + " checkins out of " + str(len(checkins_all)) + " are inside " + main_city, self.logger, True, False)

        # Get the group user relations
        write_running_log("groups_user_relations", self.logger)
        df_all_groups = deque()
        for group in df_checkin_group.group_id.unique():
            l_res = deque()
            for user in group.split(","):
                res = OrderedDict({})
                res["group_id"] = group
                res["user_id"] = int(user)
                l_res.append(res)
            df_all_groups.append(pd.DataFrame(list(l_res)))
        df_all_groups = pd.concat(df_all_groups)

        self.datasets = {"df_users": df_users,
                "df_checkins": df_checkins,
                "df_checkin_group": df_checkin_group,
                "df_checkins_with": df_checkins_with,
                "df_venues": df_venues,
                "df_venues_categories": df_venues_categories,
                "df_venues_content": df_venues_content,
                "df_group_user_checkins": df_group_user_checkins,
                "df_all_groups": df_all_groups}

        return self.datasets
