import json
import csv
import re
import time
from random import randrange
from random import randint

from itertools import chain

import requests.exceptions
import click
from github import Github
from github.GithubException import GithubException

REQUEST_WAIT = 60
CSV_OUTPUT = 'energy-matches.csv'
CSV_HEADER = [
    'user',
    'repo',
    'url',
    'ref',
    'text',
    'state',
    'title',
    'contribution_type'
]
"""------------------------------------------
commit
user
repo
url
ref
title
diff of the commit from previous one---to know what has been changed
"see from pygithub what are the important fields need to be considered for commits, see some blogs also"

issues
user
repo
ref
title




We also have to collect metadata of all the repos like loc, number of stars


"""






#71a2347fff92ec8658f756c4d7457c0e324037db
#akhila token









# authenticate github api
#with open('./config.json') as config_file:
    #config = json.load(config_file)

#guser = os.getenv('phoebs88')
#gpass = os.getenv('')
#GTHUB = Github(guser,gpass)


# regex to match commit messages
PATTERN_ENERGY = "(.*(energy).*)|(.*(battery).*)|(.*(power).*)|(.*(when).*)|(.*(is).*)"
regexEnergy = re.compile(PATTERN_ENERGY, re.IGNORECASE)

_github_api = None
def get_github_api():
    print("Init Github API client.1")
    # authenticate github api
    global _github_api
    print("Init Github API client.2")
    if _github_api is None:
        print("Init Github API client.3")
        #with open('./config.json') as config_file:
            #config = json.load(config_file)
        _github_api = Github(
            login_or_token='ebb5ecc8b8dd3b50df6ca32d034e27aa1fe1e5bf'
        )
        print("Init Github API client.4")
    return _github_api


def get_repo(organization, project):
    print("Get commits of a repository.1")
    return get_github_api().get_user(organization).get_repo(project)



def analyze_repo(user, project, retry=120):
    print("Find pattern a given repository.1")

    result = []
    print("Find pattern a given repository.2")
    try:
        print("Find pattern a given repository.3")
        repo = get_repo(user, project)
        print("Find pattern a given repository.4")
        commits = repo.get_commits()
        print("Find pattern a given repository.5")
        for commit in commits:
            print("Find pattern a given repository.6")
            commit_message = commit.commit.message
            #get comments of commits
            cmnts= commit.get_comments()
            
            #match = regexEnergy.search(commit_message)
            #if match:
            print('----------------')
            print("Repo {}/{}".format(user, repo))
            print(commit_message)
            print(commit.html_url)
            result.append({
                'user': user,
                'repo': project,
                'url': commit.html_url,
                'ref': commit.sha,
                'text':commit_message,
                #'match': match.group(0),
                #'created_at':
                'state': 'none',
                'title':commit,
                'contribution_type': 'commit_message'
                })
            for cmnt in cmnts:
                result.append({
                'user': user,
                'repo': project,
                'url': commitcomment.html_url,
                'ref': commitcomment.sha,
                'text':commitcomment.body,
                #'match': match.group(0),
               # 'created_at': commitcomment.created_at;
                ''
                'state': 'none',
                'title':'no_title',
                'contribution_type': 'commit_comment'
                })
        pull_requests = repo.get_pulls(state='all')
        issues_all = repo.get_issues(state='all') # somehow also collects PRs
        issues = (issue for issue in issues_all if issue.pull_request is None)
        for subject in chain(issues, pull_requests):
            content = "\n".join([
                subject.title,
                str(subject.body),
            ]+[str(comment.body) for comment in subject.get_comments()])
            #match = regexEnergy.search(content)
            #if match:
            result.append({
                'user': user,
                'repo': project,
                'url': subject.html_url,
                'ref': subject.number,
                'text': subject.body,
                'state': subject.state,
                'title':subject.title,
                'contribution_type': type(subject).__name__
                })
            """cmnts=subject.get_comments()
            for cmnt in cmnts:
                result.append({
                'user': user,
                'repo': project,
                'url': cmnt.html_url,
                'ref': 'none',
                'text':cmnt.body,
                #'match': match.group(0),
               # 'created_at': commitcomment.created_at;
                ''
                'state': 'none',
                'title':'no_title',
                'contribution_type': type(subject).__name__+'_comments'
                })"""
    except requests.exceptions.HTTPError as error:
        print(
            "Error in repo {}/{}: 403 forbidden -- {}"
            "".format(user, repo, error.message)
        )
        if retry:
            print("Will retry in 1 minute.")
            time.sleep(REQUEST_WAIT*randrange(0.7, 1.4, 0.1))
            return analyze_repo(user, repo, retry-1)
    return result


@click.argument('input_path', default="make_java.csv",
                type=click.Path(dir_okay=False))
@click.argument('output_path', default="result.csv",
                type=click.Path(dir_okay=False))
@click.option('--history', type=click.Path(dir_okay=False))
@click.command()



def main(input_path, output_path, history):
    print("Process github repos.")
    history_list = None
    print("Process github repos.1")
    with open(output_path, 'a') as f:
        writer = csv.DictWriter(
        f, fieldnames= CSV_HEADER)
        writer.writeheader()
        f.close()
    if history:
        print("Process github repos2")
        with open(history, 'r') as history_file:
            print("Process github repos.3")
            history_list = history_file.read().splitlines()
            print("Process github repos4")

    with open(input_path, 'r') as input_file:
        print("Process github repos5")
        csv_reader = csv.DictReader(input_file)
        print("Process github repos6")
        print(csv_reader)
        for app in csv_reader:
            print("Process github repos.7")
            print(app)
            try:
                app_uri = "{}/{}".format(app['user'], app['project_name'])
                print("Process github repos.8")
                """click.secho(
                    'Processing {}'.format(app_uri),
                    fg='blue'
                )"""
                print("Process github repos.9")
                if history_list and (app_uri in history_list):
                    """click.secho(
                        'History: {} already processed'.format(app_uri),
                        fg='green'
                    )"""
                    continue
                print("Process github repos.10")
                results = analyze_repo(app['user'], app['project_name'])
                print("Process github repos.11")
                with open(output_path, 'a') as output_file:
                    print("Process github repos.12")
                    csv_writer = csv.DictWriter(
                        output_file,
                        fieldnames=CSV_HEADER
                    )
                    print("Process github repos.13")
                    csv_writer.writerows(results)
                    print("Process github repos.14")
                if history:
                    with open(history, 'a') as history_file:
                        history_file.write(app_uri+"\n")

            except GithubException as exception:
                print(exception)
                print(exception.args)
                print("Skipping repo {}/{}: {}.".format(
                    app['user'],
                    app['project_name'],
                    exception.data['message']
                ))
                if exception.status == 403:
                    #sleep 1--2minutes
                    time.sleep(randint(60, 120))
            #pretend this is not a bot
            time.sleep(randint(1, 10))


def exit_gracefully(start_time):
    exit_time = time.time()
    duration = exit_time - start_time
    #click.secho(
        #"Script exited in {:.2f} minutes.".format(duration/60),
        #fg='green'
    #)


if __name__ == '__main__':
    START_TIME = time.time()
    print("asg")
    try:
        print("asxfs")
        main()
        print("hvjhj")
    except KeyboardInterrupt:
        pass
    finally:
        exit_gracefully(START_TIME)
