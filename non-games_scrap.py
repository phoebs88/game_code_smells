from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from time import sleep

options = Options()
options.add_argument("--disable-notifications")
#options.add_argument("download.default_directory=C:\\Users\\minni\\Desktop\\javascript_games")
driver = webdriver.Chrome(r"C:\Users\Vartika Agrahari\Downloads\chromedriver_win32\chromedriver.exe",chrome_options=options)

f=open("JavaGamesLink.txt","r") #file containing game repositories links
fileData = f.readlines()
for repolink in fileData:
    driver.get(repolink+'/archive/master.zip')
    sleep(30)
f.close()
driver.quit()
