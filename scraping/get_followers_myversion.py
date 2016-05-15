###############################################################
# Julien Pascal
#
# Code to gather information on Twitter friends and followers
##############################################################

import tweepy
import time
import os
import sys
import json
import argparse
import pandas as pd

#################################
#ADJUST THE PATHS TO YOUR SETTING
#locate in the correct directory:
path = '/home/julien/Social-Networks/scraping'
path_data = '/home/julien/Social-Networks/data'
os.chdir(path) 

LINKS_DIR = 'Links'
MAX_FRIENDS = 200
FRIENDS_OF_FRIENDS_LIMIT = 200

if not os.path.exists(LINKS_DIR):
    os.makedir(LINKS_DIR)

enc = lambda x: x.encode('ascii', errors='ignore')

###########################
# ENTER YOUR API KEYS HERE
#
# The consumer keys can be found on your application's Details
# page located at https://dev.twitter.com/apps (under "OAuth settings")
CONSUMER_KEY = 'xxxx'
CONSUMER_SECRET = 'xxxx'

API_KEY = 'xxxx'
API_SECRET = 'xxxx'

# The access tokens can be found on your applications's Details
# page located at https://dev.twitter.com/apps (located
# under "Your access token")
ACCESS_TOKEN = 'xxxx'
ACCESS_TOKEN_SECRET = 'xxxx'


'''
# == OAuth Authentication ==
#
# This mode of authentication is the new preferred way
# of authenticating with Twitter.
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

status = "Testing!"
api.update_status(status=status)
'''

# Replace the API_KEY and API_SECRET with your application's key and secret.
auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)
 
api = tweepy.API(auth, wait_on_rate_limit=True,
                   wait_on_rate_limit_notify=True)

#Check for the API
if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)

list_user = [] #initialization

#Load the list of the participants in the thread #COP21MIX between May27th and May 30th:
file_name = path_data + '/' + 'List_COP21MIW.csv' #load the csv file
df = pd.read_csv(file_name)

Twitter_accounts = [] #Put the names into a list
for i in range(0,len(df)):
	Twitter_accounts.append(df.ix[i,'@name'])

#Twitter_accounts = ['MIW_SaudiArabia','StrandedMIW', 'NGO_MIW', 'CitiesMIW'] #have to fill up the list

Twitter_ids = [] #Have to find the correspondence between name and id
#Store the fist 5000 followers and friends:
for i in range(0,len(Twitter_accounts)):
	userfname =  path + '/' + LINKS_DIR + '/' + Twitter_accounts[i] + '.json'
	if not os.path.exists(userfname): #Check whether or not the fle already exists
		while True: 
			try:
				user = api.get_user(Twitter_accounts[i]) #get a lot of information
				print('Success getting followers')
				break #If success, get out of the while loop
			except tweepy.RateLimitError:
				print('Error getting followers:')
				print(type(error)) #
				print('Sleep for fifteen minutes')
				time.sleep(15 * 60 + 2)
		while True: 
			try:
				friendsid = api.friends_ids(Twitter_accounts[i]) #get the 5000 first friends
				print('Success getting friends')
				break
			except tweepy.TweepError as error:
				print('Error getting friends:')
				print(type(error)) #
				print('Sleep for fifteen minutes')
				time.sleep(15 * 60 + 2)

		d = {'name': user.name,
		'screen_name': user.screen_name,
		'id': user.id,
		'friends_count': user.friends_count,
		'followers_count': user.followers_count,
		'followers_ids': user.followers_ids(),
		'friends_ids': friendsid,
		'description': user.description} #id of followers
		list_user.append(d)
		userfname = path + '/' + LINKS_DIR + '/' + Twitter_accounts[i] + '.json'
		print(user.screen_name)
		with open(userfname, 'w') as f:
			json.dump(d, f) #s

