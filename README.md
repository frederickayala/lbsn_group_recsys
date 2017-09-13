You may find the paper here: http://dl.acm.org/citation.cfm?id=3091485

If you use any part of the code or dataset, kindly cite our work as:

Bibtex:
<pre>
@inproceedings{Ayala-Gomez:2017:WGR:3091478.3091485,
 author = {Ayala-G\'{o}mez, Frederick and Dar\'{o}czy, B\'{a}lint and Mathioudakis, Michael and Bencz\'{u}r, Andr\'{a}s and Gionis, Aristides},
 title = {Where Could We Go?: Recommendations for Groups in Location-Based Social Networks},
 booktitle = {Proceedings of the 2017 ACM on Web Science Conference},
 series = {WebSci '17},
 year = {2017},
 isbn = {978-1-4503-4896-6},
 location = {Troy, New York, USA},
 pages = {93--102},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3091478.3091485},
 doi = {10.1145/3091478.3091485},
 acmid = {3091485},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {group recommendation, location-based social networks, recommender systems},
}
</pre>

ACM Ref:
<pre>
Frederick Ayala-Gómez, Bálint Daróczy, Michael Mathioudakis, András Benczúr, and Aristides Gionis. 2017. Where Could We Go?: Recommendations for Groups in Location-Based Social Networks. In Proceedings of the 2017 ACM on Web Science Conference (WebSci '17). ACM, New York, NY, USA, 93-102. DOI: https://doi.org/10.1145/3091478.3091485
</pre>

# Getting Started
- Install python and virtualenv
- virtualenv venv
- source venv/bin/activate
- pip install -r requirements.txt

## Additional third party software to be installed
- Install Turi's GraphLab Create: You need to ask for an academic license

# Data Collection
### Getting data from Twitter
- source venv/bin/activate
- cd data_collection
- Use the template in data_collection/config/template.config as a reference and update it with your data
- Resolve the checkins using Foursaquare API

### Parsing the JSON check-ins file
- source venv/bin/activate
- cd data_collection
- python foursquare2csv.py checkins_file output_dir
    - A script that parses the checkins to pandas Dataframes:
        - df_checkin_group.csv: checkin_id, group_id 
        - df_checkins.csv: checkin_beenHere, checkin_created_at, checkin_created_via, checkin_id, checkin_likes, checkin_timeZoneOffset, user_id, venue_id 
        - df_checkins_with.csv: checkin_id, user_id, with, group_id 
        - df_users.csv: user_firstname, user_gender, user_id, user_screen_name 
        - df_venues_categories.csv: category_id, category_name, category_pluralName, category_primary, category_shortName, venue_id 
        - df_venues.csv: venue_address, venue_cc, venue_checkinsCount, venue_city, venue_country, venue_id, venue_lat, venue_long, venue_name, venue_state, venue_tipCount, venue_usersCount, venue_verified
    
# Data Analysis
### Statistics and charts
- cd data_analysis/
- python run_analysis.py configuration_file
    - Where:
    - configuration_file: Your configuration for running the analysis

# Experiments
### Run Baselines
- python run_baselines.py configuration_file
    - Where:
    - configuration_file: Your configuration for running the baselines

# Dataset
- For privacy reasons, we cannot share a public link to the dataset. Please contact the main author for further information.   