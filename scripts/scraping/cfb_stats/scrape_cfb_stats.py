import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count
import time
import logging
import os
import pandas as pd

from scripts.scraping.cfb_stats.helper.scrape_cfb_stats_helper import handle_parse_page

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class ScrapeCFBStats:
    def __init__(self, data_dir, save_dir, positional_stats):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.positional_stats = positional_stats
        self.base_url = 'https://cfbstats.com/2024/team/index.html'
        self.fallback_url = 'https://6squ2cs06j.execute-api.us-east-1.amazonaws.com/v1/web_content_scraper'
        self.failed_requests = Manager().Queue()
        self.fallback_success = Manager().Value('i', 0)
        self.fallback_failure = Manager().Value('i', 0)
        

    def download_page(self, url=None, retries=2, wait=3):
        if not url:
            url = self.base_url

        for attempt in range(retries):
            try:
                page = requests.get(url, timeout=10)
                if page.status_code == 200:
                    return page
                elif page.status_code == 404:
                    logging.error(f"404 error for URL: {url}")
                    return None
                else:
                    logging.warning(f"HTTP error {page.status_code} for URL: {url}. Attempt {attempt + 1}/{retries}")
                    time.sleep(wait)
                wait *= 2
            except requests.RequestException as e:
                logging.warning(f"Request exception for URL: {url}. Attempt {attempt + 1}/{retries}. Error: {str(e)}")
                time.sleep(wait)
                wait *= 2

        # If all retries fail, use the fallback method
        logging.info(f"All retries failed for URL: {url}. Using fallback method.")
        return self.fallback_download(url)

    def fallback_download(self, url):
        try:
            response = requests.post(self.fallback_url, json={"url": url})
            if response.status_code == 200:
                content = response.json().get('body')
                if content:
                    self.fallback_success.value += 1
                    return type('Response', (), {'content': content.encode('utf-8'), 'status_code': 200})()
            logging.error(f"Fallback request failed for URL: {url}. Status code: {response.status_code}")
            self.fallback_failure.value += 1
        except requests.RequestException as e:
            logging.error(f"Fallback request exception for URL: {url}. Error: {str(e)}")
            self.fallback_failure.value += 1
        return None

    def process_failed_requests(self):
        failed_count = self.failed_requests.qsize()
        for _ in tqdm(range(failed_count), desc="Processing failed requests"):
            url = self.failed_requests.get()
            self.fallback_download(url)

    def scrape_team_site(self, args):
        url, season, team_name = args
        url = url.replace('index.html', '')
        
        team_data = {}
        
        for stat in self.positional_stats:
            stat_name = stat['name']

            stat_url = url + stat_name + '/index.html'
            page = self.download_page(stat_url)

            if not page:
                print(f"Failed to download page for {team_name} - {stat_name}")
                continue

            stat_data_df = handle_parse_page(page.content, stat_name)
            
            # Add team_name column to the DataFrame
            stat_data_df['team_name'] = team_name
            
            team_data[stat_name] = stat_data_df

        return team_name, team_data

    def save_data(self, season, stat_name, new_data):
        save_path = f'{self.save_dir}/cfb_stats_{season}_{stat_name}.csv'
        
        if os.path.exists(save_path):
            try:
                existing_df = pd.read_csv(save_path)
                if not existing_df.empty:
                    existing_df = pd.concat([existing_df, new_data], ignore_index=True)
                else:
                    existing_df = new_data
            except pd.errors.EmptyDataError:
                existing_df = new_data
            except Exception as e:
                logging.error(f"Error reading existing CSV for {stat_name}: {str(e)}")
                existing_df = new_data
        else:
            existing_df = new_data

        existing_df.to_csv(save_path, index=False)

    def scrape_season(self, season):
        df = pd.read_csv(f'{self.data_dir}/cfb_roster_{season}.csv')
        team_data = df[['team_page_url', 'team_name']].drop_duplicates()

        print(f"Number of rows in DataFrame: {len(df)}")
        print(f"Number of unique teams: {len(team_data)}")

        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(self.scrape_team_site, [(row['team_page_url'], season, row['team_name']) for _, row in team_data.iterrows()]), 
                        total=len(team_data), 
                        desc=f"Scraping Season {season}"))
        
        # Aggregate results and save
        for stat_name in self.positional_stats:
            
            all_team_data = pd.concat([team_data[stat_name['name']] for _, team_data in results if stat_name['name'] in team_data], ignore_index=True)
            self.save_data(season, stat_name['name'], all_team_data)

        print(f"Saved data for {len(results)} teams")

    def run_scrape(self):
        seasons = [file.split('_')[-1].split('.')[0] for file in os.listdir(self.data_dir) if 'cfb_roster' in file]

        #clean save directory
        for file in os.listdir(self.save_dir):
            if 'cfb_stats' in file:
                os.remove(os.path.join(self.save_dir, file))
        
        for season in seasons:
            print(f"Scraping Season: {season}")
            self.scrape_season(season)

        self.process_failed_requests()
        print(f"Fallback successes: {self.fallback_success.value}")
        print(f"Fallback failures: {self.fallback_failure.value}")
