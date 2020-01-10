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
CSV_HEADER = [
    'repo',
    'user',
    'project_name'
]
@click.argument('input_path', default="popular_game_list_desktop.csv",
                type=click.Path(dir_okay=False))
@click.argument('output_path', default="popular_desktop_csv_100.csv",
                type=click.Path(dir_okay=False))
@click.command()
def main(input_path, output_path):
        with open(output_path, 'a') as f:
                writer = csv.DictWriter(f, fieldnames= CSV_HEADER)
                writer.writeheader()
                f.close()
        with open(input_path, 'r') as input_file:
                print("Process github repos5")
                csv_reader = csv.reader(input_file)
                print("Process github repos6")
                print(csv_reader)
                for app in csv_reader:
                        print("app i am")
                        print(app)
                        with open(output_path, 'a') as output_file:
                            print("Process github repos.12")
                            csv_writer = csv.DictWriter(
                                output_file,
                                fieldnames=CSV_HEADER
                            )
                            #x = re.split("//", app)
                            x=app[0].split('/')
                            print(x)
                            results=[]
                            results.append({
                		          'repo': app,
                		          'user': x[3],
                		          'project_name': x[4]})
                            print("Process github repos.13")
                            csv_writer.writerows(results)
def exit_gracefully(start_time):
    exit_time = time.time()
    duration = exit_time - start_time
    #click.secho(
     #   "Script exited in {:.2f} minutes.".format(duration/60),
      #  fg='green'
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

