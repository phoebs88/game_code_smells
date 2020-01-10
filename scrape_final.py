import pandas as pd
f=open("repos.txt","r") #file containing game repositories links
fileData = f.readlines()
for repolink in fileData:







#-------------------------------------SCRAPE ISSUES
'''GO TO links
EXPAND ALL
COLLECT IN CSV FORMAT 
ISSUE ID, REPONAME, Title, body, state, comments, '''


from github import Github
from pprint import pprint
import os

guser = os.getenv('GUSER')
gpass = os.getenv('GPASS')

gpat = os.getenv('PAT')


GTHUB = Github(guser,gpass)

GTHUBKEY = Github(gpat)

f=open("repos.txt","r") #file containing game repositories links
fileData = f.readlines()
for repolink in fileData:
REPOS = [ _ for _ in GTHUB.get_user().get_repos()]

print(REPOS)

print('API status: {}'.format(GTHUB.get_last_api_status_message().body))

print('Total public repos: {}'.format(GTHUB.get_user().public_repos))
print('Total private repos: {}'.format(GTHUB.get_user().total_private_repos))