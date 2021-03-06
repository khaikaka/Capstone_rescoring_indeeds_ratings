import time
from selenium.webdriver import Firefox
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from collections import OrderedDict
from helper import *
from upload_data_to_S3 import *
import boto3




tech_companies_lst = ['Apple', 'Google', 'Microsoft', 'Amazon', 'Intel', 'IBM', 'Netflix', 'Oracle',
                     'Cisco', 'Adobe', 'Twitter', 'Workday', 'Dell', 'Airbnb', 'Tesla', 'Uber',
                      'Salesforce', 'Kaiser Permanente', 'Indeed', 'Redfin', 'Zillow', 'Expedia', 'T-mobile',
                     'Tableau', 'eBay','Allstate', 'KPMG', 'Nordstrom', 'Boeing', 'Starbuck','University of Washington',
                     'NOKIA','JLL', 'Adobe']

for cp in tech_companies_lst:
    main_scraping_function(cp)
