import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import pickle

PROPUBLICA_API_KEY = "9A9tOoalbjOsfsLz4LO2aSVlaAuQNiYdgzhJszji"

"""
# Step 1, use the TrackingCongressInTheAgeOfTrump538.csv -> get all the bills
# Step 2, download the summaries of all the bills
# Step 3, Take those texts, train it with 0/1 trump agrees

Perfect Table Schema:

Bill Table:
- Bill ID
- Bill Title
- Bill Summary

Congressman Table:
- Congressman/Senator
- Chamber
- State
- Political Party..?
- Iteration (115 vs 116 Congress)

Vote History:
- Bill ID
- Congressman
- Vote
"""

REQUESTS_CACHE = None
BILL_TABLE_DATA = {
    'id': [],
    'title': [],
    'summary': [],
    'summary_short': [],
    'trump_support': []
}

CONGRESS_TABLE_DATA = {
    'id': [],
    'name': [],
    'chamber': [],
    'state': [],
    'party': [],
    'iteration': [],
    'trump_agree_rate': [],
}

VOTE_TABLE_DATA = {
    'bill': [],
    'person': [],
    'vote': []
}

VOTE_URLS = []

def cache_get_json(url, *args, **kwargs):
    global REQUESTS_CACHE
    if REQUESTS_CACHE is None:
        if os.path.isfile('./json_$.cache'):
            with open('./json_$.cache', 'rb') as handle:
                REQUESTS_CACHE = pickle.load(handle)
        else:
            REQUESTS_CACHE = {}

    if url not in REQUESTS_CACHE:
        REQUESTS_CACHE[url] = requests.get(url, *args, **kwargs).json()
        with open('./json_$.cache', 'wb') as handle:
            pickle.dump(REQUESTS_CACHE, handle)

    return REQUESTS_CACHE[url]

def download_bill_summaries():
    voting_record = pd.read_csv('data/TrackingCongressInTheAgeOfTrump538.csv')
    bill_ids = voting_record['bill_id'].unique()

    for bill_id in bill_ids:
        bill_id_orig = bill_id = bill_id.lower()

        components = bill_id.split('-')
        bill_id = '-'.join(components[:-1])
        congress = components[-1]

        try:
            results = cache_get_json(
                f"https://api.propublica.org/congress/v1/{congress}/bills/{bill_id}.json",
                headers={
                    'X-API-Key': PROPUBLICA_API_KEY
                }
            )['results'][0]
        except KeyError:
            results = cache_get_json(
                f"https://api.propublica.org/congress/v1/{congress}/bills/{bill_id}.json",
                headers={
                    'X-API-Key': PROPUBLICA_API_KEY
                }
            )
            continue

        bill_title = results['title']
        bill_summary = results['summary']
        bill_summary_short = results['summary_short']

        
        try:
            VOTE_URLS.append((bill_id, results['votes'][0]['api_url']))
        except IndexError:
            continue

        print("Adding:", bill_id, bill_title)

        trump_stance = voting_record[voting_record['bill_id'] == bill_id_orig]['trump_position'].unique()[0]
        trump_agree = int(trump_stance.lower() == 'support')

        BILL_TABLE_DATA['id'].append(bill_id_orig)
        BILL_TABLE_DATA['title'].append(bill_title)
        BILL_TABLE_DATA['summary'].append(bill_summary.replace('(This measure has not been amended since it was introduced. The summary of that version is repeated here.) ', ''))
        BILL_TABLE_DATA['summary_short'].append(bill_summary_short.replace('(This measure has not been amended since it was introduced. The summary of that version is repeated here.) ', ''))
        BILL_TABLE_DATA['trump_support'].append(trump_agree)

    dataframe = pd.DataFrame(BILL_TABLE_DATA)
    with open('data/bill_table.csv', 'w', newline='', encoding='utf-8') as handle:
        dataframe.to_csv(handle)

def download_votes_table():
    for bill_id, vote_url in VOTE_URLS:
        print("Downloading voting record for:", bill_id, end=' ... ')
        results = cache_get_json(
            vote_url,
            headers={
                'X-API-Key': PROPUBLICA_API_KEY
            }
        )['results']['votes']['vote']['positions']
        
        for vote_record in results:
            VOTE_TABLE_DATA['bill'].append(bill_id)
            VOTE_TABLE_DATA['person'].append(vote_record['member_id'])
            VOTE_TABLE_DATA['vote'].append(int(vote_record['vote_position'].lower().strip() == 'yes'))

        print('Counted', len(results), 'votes')

    dataframe = pd.DataFrame(VOTE_TABLE_DATA)
    with open('data/vote_table.csv', 'w', newline='', encoding='utf-8') as handle:
        dataframe.to_csv(handle)

def download_congressmen_table():
    agree_rate = pd.read_csv('data/CongressionalAgree538.csv')

    for chamber in ('house', 'senate'):
        for congress in (115, 116):
            print(f"Downloading {chamber} ({congress}) ... ", end='')
            payload = f"https://api.propublica.org/congress/v1/{congress}/{chamber}/members.json"
            results = cache_get_json(
                payload,
                headers={
                    'X-API-Key': PROPUBLICA_API_KEY
                }
            )['results'][0]['members']

            for member in results:
                member_name = f'{member["first_name"]} {member["last_name"]}'
                member_id = member['id']

                agree_history = float(agree_rate[agree_rate['bioguide'] == member_id]['agree_pct'].mean())

                CONGRESS_TABLE_DATA['id'].append(member_id)
                if os.path.isfile(f"ideolog/base/static/base/{member['state'].lower()}-{member['last_name']}.jpg"):
                    os.system(
                        f"Powershell -Command Invoke-WebRequest -UseBasicParsing -Uri \"https://raw.githubusercontent.com/unitedstates/images/gh-pages/congress/225x275/{member_id}.jpg\" -O \"ideolog/base/static/base/{member['state'].lower()}-{member['last_name']}.jpg\""
                    )
                CONGRESS_TABLE_DATA['name'].append(member_name)
                CONGRESS_TABLE_DATA['party'].append(member['party'])
                CONGRESS_TABLE_DATA['state'].append(member['state'])
                CONGRESS_TABLE_DATA['chamber'].append(chamber)
                CONGRESS_TABLE_DATA['iteration'].append(congress)
                CONGRESS_TABLE_DATA['trump_agree_rate'].append(agree_history)

            print("Pulled", len(results), "members")

    dataframe = pd.DataFrame(CONGRESS_TABLE_DATA)
    with open('data/house_table.csv', 'w', newline='', encoding='utf-8') as handle:
        dataframe.to_csv(handle)

def main():
    print("Downloading Bill Summaries: ")
    download_bill_summaries()
    print("Downloading Votes: ")
    download_votes_table()
    print("Downloading congressman: ")
    download_congressmen_table()

if __name__ == "__main__":
    main()