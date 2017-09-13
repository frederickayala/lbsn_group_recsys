# coding: utf-8

import sys, traceback
import pandas as pd
import json
import networkx as nx
import tweepy
from tweepy import parsers
from bs4 import BeautifulSoup
import requests
import urlparse
import oauth2 as oauth
import urllib2 as urllib
import csv
import time
import io
import logging
import ConfigParser
import configparser

_debug = 0
http_handler  = urllib.HTTPHandler(debuglevel=_debug)
https_handler = urllib.HTTPSHandler(debuglevel=_debug)

def twitterreq(url, http_method, parameters, settings):
    api_key = settings.get("TwitterKey","api_key")
    api_secret = settings.get("TwitterKey","api_secret")
    access_token_key = settings.get("TwitterKey","access_token_key")
    access_token_secret = settings.get("TwitterKey","access_token_secret")
    
    oauth_token = oauth.Token(key=access_token_key, secret=access_token_secret)
    oauth_consumer = oauth.Consumer(key=api_key, secret=api_secret)
    
    signature_method_hmac_sha1 = oauth.SignatureMethod_HMAC_SHA1()

    req = oauth.Request.from_consumer_and_token(oauth_consumer,
                                             token=oauth_token,
                                             http_method=http_method,
                                             http_url=url, 
                                             parameters=parameters)

    req.sign_request(signature_method_hmac_sha1, oauth_consumer, oauth_token)

    headers = req.to_header()

    if http_method == "POST":
        encoded_post_data = req.to_postdata()
    else:
        encoded_post_data = None
        url = req.to_url()

    opener = urllib.OpenerDirector()    
    opener.add_handler(http_handler)
    opener.add_handler(https_handler)

    response = opener.open(url, encoded_post_data)
    return response

def parse_status(status):
    d = {}
    d['text'] = status['text']
    d['id'] = status['id']
    d['favorite_count'] = status['favorite_count']
    d['retweet_count'] = status['retweet_count']
    d['coordinates'] = status['coordinates']
    d['entities'] = status['entities']
    d['user_id'] = status['user']["id_str"]
    d['user_screen_name'] = status['user']["screen_name"]
    d['created_at'] = status['created_at']
    d['source'] = status['source']
    return d

#get_user_foursquare_checkins('1446563863','Nidaonurr',[])
def get_user_foursquare_checkins(user_id,screen_name,crawled_users,current_level,how_deep,settings,logger):
    user_statuses = []
    to_search = 'swarmapp.com'
    search_url = "https://api.twitter.com/1.1/statuses/user_timeline.json?"
    parameters = {}
    continueFlag = True
    search = "q=" + to_search
    search = search.encode('utf8')
    logger.info("get_user_foursquare_checkins('" + "','".join([user_id,screen_name,str(current_level),str(how_deep)]) +"'")
    try:
        query_url = search_url + "count=200&exclude_replies=true&include_rts=false&user_id=" + user_id +"&screen_name=" + screen_name + "&" + search
        response = twitterreq(query_url, "GET", parameters,settings)
        json_response = json.load(response)
        crawled_users.add(user_id)
        time.sleep(3)
        for status in json_response:
            if type(status) is dict:
                status_clean = parse_status(status)
                if "foursquare" in status_clean["source"].lower():
                    user_statuses.append(status_clean)
                    if  " w/ " in status_clean["text"]:
                        for user_mentioned in status_clean["entities"]["user_mentions"]:                            
                            if user_mentioned["id_str"] not in crawled_users and current_level < how_deep:
                                user_mentioned_homeline = get_user_foursquare_checkins(user_mentioned["id_str"],user_mentioned["screen_name"],crawled_users,current_level+1,how_deep,settings,logger)[0]
                                crawled_users.add(user_mentioned["id_str"])
                                for friend_status in user_mentioned_homeline:
                                    user_statuses.append(friend_status)
    except Exception as error:
        traceback.print_exc(file=sys.stdout)
        txt_error = "Something went wrong. Query:" + query_url
        logger.error(txt_error)
    return (user_statuses,crawled_users)

def main(settings):
    #Read settings file
    log_file = settings.get("Settings","log_file")
    tweets_file = settings.get("Settings","tweets_file")
    how_deep = settings.getint("Settings","depth_level")
    geocode = settings.get("Settings","geocode")
    #Remove the defaults Jupyter logging handler
    root = logging.getLogger()
    root.handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO,filename=log_file)
    logger = logging.getLogger(__name__)
    logger.info('Ready to start getting data from Twitter')

    all_users = set()

    to_search = 'swarmapp.com "w/"'
    if len(geocode) > 0:
     to_search += '&geocode=' + geocode
    search_url = "https://api.twitter.com/1.1/search/tweets.json?"
    parameters = {}
    max_id = ""
    continueFlag = True
    search = "q=" + to_search

    while continueFlag:
        try:
            if max_id != "":
                response = twitterreq(search_url + "&max_id=" + max_id + "&" + search + "&count=100&include_entities=1", "GET", parameters, settings)
            else:
                response = twitterreq(search_url + search + "&count=100&include_entities=1", "GET", parameters, settings)
            json_response = json.load(response)
            #Handle the statuses
            statuses = json_response['statuses']
            print len(statuses)
            for status in statuses:
                status_clean = parse_status(status)
                if status_clean["user_id"] not in all_users:
                    new_set = set()
                    user_colletion = get_user_foursquare_checkins(status_clean["user_id"],status_clean["user_screen_name"],new_set,0,how_deep,settings,logger)
                    tweets = user_colletion[0]
                    for uc in user_colletion[1]:
                        all_users.add(uc)                
        
                    with io.open(tweets_file, 'a', encoding='utf-8') as f:
                        for t in [unicode(json.dumps(t,ensure_ascii=False)) for t in tweets]:
                            f.write(t)
                            f.write(unicode("\n"))

            #Handle the Metadata
            metadata = json_response['search_metadata']
            if metadata.has_key('next_results'):
                if urlparse.parse_qs(metadata["next_results"]).has_key("?max_id"):
                    max_id = urlparse.parse_qs(metadata["next_results"])["?max_id"][0]
                if urlparse.parse_qs(metadata["next_results"]).has_key("max_id"):
                    max_id = urlparse.parse_qs(metadata["next_results"])["max_id"][0]
            else:
                continueFlag = False            
        except Exception as error:
            traceback.print_exc(file=sys.stdout)
            query_url = search_url + "&max_id=" + max_id + "&" + search + "&count=100&include_entities=1"
            txt_error = "Something went wrong. Query:" + query_url + " Error:" + str(error)
            logger.error(txt_error)

if __name__ == "__main__":
    usage = """
        usage: 
            python twitter_public_checkins.py settings_file
        """
    if len(sys.argv) < 2:
        print usage
        sys.exit(1)
    else:
        settings = ConfigParser.ConfigParser()
        settings._interpolation = configparser.ExtendedInterpolation()
        settings.read(sys.argv[1])
        print main(settings)