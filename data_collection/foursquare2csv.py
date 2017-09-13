# coding: utf-8

import pandas as pd
import io
import json
from collections import OrderedDict
import csv
import os
import sys, traceback

def save2csv(df,filename):
    df.to_csv(filename,index=False,encoding="utf8",quoting=csv.QUOTE_ALL,quotechar="\"")

def get_checkin_info(t):
    ci = OrderedDict({})
    ci["checkin_id"] = t["id"]
    ci["checkin_created_at"] = t["createdAt"]
    ci["user_id"] = t["user"]["id"]
    ci["venue_id"] = t["venue"]["id"]
    ci["checkin_created_via"] = t["source"]["name"]
    ci["checkin_likes"] = t["likes"]["count"]
    ci["checkin_timeZoneOffset"] = t["timeZoneOffset"]
    ci["checkin_beenHere"] = t["venue"]["beenHere"]["marked"]
    return ci

def main(checkins_file,output_dir):
    print("Loading checkins_file")
    users = []
    checkins = []
    checkins_with = []
    venues = []
    venues_categories = []
    with io.open(checkins_file, 'r', encoding='utf-8') as f_checkins: 
        for c in f_checkins:
            try:
                t = json.loads(c)

                #Users
                u = OrderedDict({})
                u["user_id"] = t["user"]["id"]
                u["user_screen_name"] = t["user"]["canonicalPath"][1:]
                u["user_gender"] = t["user"].get("gender",None)
                u["user_firstname"] = t["user"].get("firstName",None)
                users.append(u)

                #Venues
                v = OrderedDict({})
                v["venue_id"] = t["venue"]["id"]
                v["venue_name"] = t["venue"]["name"]
                v["venue_tipCount"] = t["venue"]["stats"].get("tipCount",None)
                v["venue_checkinsCount"] = t["venue"]["stats"].get("checkinsCount",None)
                v["venue_usersCount"] = t["venue"]["stats"].get("usersCount",None)
                v["venue_lat"] = t["venue"]["location"].get("lat",None)
                v["venue_long"] = t["venue"]["location"].get("lng",None)
                v["venue_city"] = t["venue"]["location"].get("city",None)
                v["venue_cc"] = t["venue"]["location"].get("cc",None)
                v["venue_country"] = t["venue"]["location"].get("country",None)
                v["venue_state"] = t["venue"]["location"].get("state",None)
                v["venue_address"] = t["venue"]["location"].get("address",None)
                v["venue_verified"] = t["venue"].get("verified",None)
                
                venues.append(v)

                #Venue Categories
                for vc in t["venue"]["categories"]:
                    d_vc = OrderedDict({})
                    d_vc["venue_id"] = t["venue"]["id"]
                    d_vc["category_id"] =  vc["id"]
                    d_vc["category_pluralName"] =  vc["pluralName"]
                    d_vc["category_primary"] =  vc["primary"]
                    d_vc["category_name"] =  vc["name"]
                    d_vc["category_shortName"] =  vc["shortName"]
                    
                    venues_categories.append(d_vc)

                #Checkins
                checkins.append(get_checkin_info(t))
                
                #Checked in with others            
                if "entities" in t.keys():
                    friends = [u for u in t["entities"] if u["type"] == 'user']
                    if len(friends) > 0:
                        for friend in friends:
                            ciw = OrderedDict({})
                            ciw["user_id"] = t["user"]["id"]
                            ciw["checkin_id"] = t["id"]
                            ciw["venue_id"] = t["venue"]["id"]
                            ciw["checkin_created_at"] = t["createdAt"]
                            ciw["with"] = friend["id"]
                            checkins_with.append(ciw)
            except Exception as ex:
                print str(ex)
                traceback.print_exc(file=sys.stdout)
                pass

    #df_users
    print("Processing users (df_users)")
    df_users = pd.DataFrame(users).drop_duplicates()
    del users
    save2csv(df_users,os.path.join(output_dir,"df_users.csv"))

    #df_checkins
    print("Processing checkins (df_checkins)")
    df_checkins = pd.DataFrame(checkins).drop_duplicates()
    del checkins
    save2csv(df_checkins,os.path.join(output_dir,"df_checkins.csv"))

    #df_checkins_with
    print("Processing checkins with (df_checkins_with, df_group_checkin)")
    df_checkins_with = pd.DataFrame(checkins_with).drop_duplicates()
    del checkins_with

    #Get the groups
    df_checkins_with['with'] = df_checkins_with['with'].astype(str)
    df_checkins_with['user_id'] = df_checkins_with['user_id'].astype(str)
    df_groups = df_checkins_with.groupby(["checkin_id"])[['with']].aggregate(lambda x: ','.join(sorted([y for y in x if y != 'nan'])))
    df_groups = df_groups.reset_index()
    df_checkin_group = df_checkins[["checkin_id","user_id","venue_id","checkin_created_at"]].drop_duplicates().merge(df_groups,on="checkin_id")
    df_checkin_group.user_id = df_checkin_group.user_id.astype(str)
    df_checkin_group["group_id"] = df_checkin_group["user_id"] + "," + df_checkin_group["with"]
    df_checkin_group["group_id"] = df_checkin_group["group_id"].apply(lambda x: x if x[0] != "," else "")
    df_checkin_group = df_checkin_group[["checkin_id","group_id","venue_id","checkin_created_at"]].sort_values("group_id")

    df_checkins_with = df_checkins_with.merge(df_checkin_group[["checkin_id","group_id"]],on="checkin_id")

    save2csv(df_checkin_group,os.path.join(output_dir,"df_checkin_group.csv"))
    save2csv(df_checkins_with,os.path.join(output_dir,"df_checkins_with.csv"))

    # df_users_groups
    print("Processing groups (df_users_groups)")
    

    # df_venues
    print("Processing venues (df_venues)")
    df_venues = pd.DataFrame(venues).drop_duplicates()
    del venues
    save2csv(df_venues,os.path.join(output_dir,"df_venues.csv"))

    # df_venues_categories
    print("Processing venues categories (df_venues_categories)")
    df_venues_categories = pd.DataFrame(venues_categories).drop_duplicates()
    del venues_categories
    save2csv(df_venues_categories,os.path.join(output_dir,"df_venues_categories.csv"))

    # df_group_user_checkins
    if "global" not in output_dir:
        print("Creating a DataFrame that associates the users checkins to the groups (df_group_user_checkins)")
        all_group_user = []
        for group in df_checkin_group.group_id.unique():
            for user in group.split(","):
                all_group_user.append(OrderedDict({"group_id": group, "user_id": user}))

        df_group_user = pd.DataFrame(all_group_user).drop_duplicates()

        df_user_checkins = df_checkins.merge(df_group_user, on="user_id")[["user_id", "checkin_id", 'venue_id', 'checkin_created_at', "group_id"]]
        df_user_with = df_checkins_with[["with", "checkin_id", "venue_id", 'checkin_created_at']].rename(columns={"with": "user_id"})
        df_user_with = df_user_with.merge(df_group_user, on="user_id")[["user_id", "checkin_id", 'venue_id', 'checkin_created_at', "group_id"]]
        df_group_user_checkins = pd.concat([df_user_checkins, df_user_with]).drop_duplicates()
        save2csv(df_group_user_checkins, os.path.join(output_dir, "df_group_user_checkins.csv"))

    print("Done!")
if __name__ == "__main__":
    usage = """
        usage: 
            python foursquare2csv.py checkins_file output_dir
        """
    if len(sys.argv) < 3:
        print usage
        sys.exit(1)
    else:
        if not os.path.isfile(sys.argv[1]):
            print "The parameter checkins_file must exist"
            sys.exit(1)
        if not os.path.isdir(sys.argv[2]):
            print "The parameter output_dir must be an existing folder"
            sys.exit(1)
        print main(sys.argv[1],sys.argv[2])