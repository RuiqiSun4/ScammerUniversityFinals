import requests
import pandas as pd

# NOAA API token 
API_TOKEN = "SqNUYjvNsiaOVPiyMxgvIWKLZmWMzels"

# basic header
headers = {
    "token": API_TOKEN
}

url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/datasets"
response = requests.get(url, headers=headers)
datasets = response.json()
for d in datasets['results']:
    print(d['id'], d['name'])

    