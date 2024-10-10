import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from multiprocessing import Pool, Manager
import time
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class ScrapeCFBRoster:
    def __init__(self, save_dir):
        self.base_url = 'https://cfbstats.com/2024/team/index.html'
        self.save_dir = save_dir
        self.fallback_url = 'https://6squ2cs06j.execute-api.us-east-1.amazonaws.com/v1/web_content_scraper'
        self.fallback_success = Manager().Value('i', 0)
        self.fallback_failure = Manager().Value('i', 0)

    def download_page(self, url=None, retries=3, wait=5):
        if not url:
            url = self.base_url
    
        for attempt in range(retries):
            try:
                page = requests.get(url, timeout=10)
                if page.status_code == 200:
                    return page
                elif page.status_code == 404:
                    logging.error(f"404 Error for URL: {url}")
                    return None
                else:
                    time.sleep(wait)
                wait *= 2
            except requests.RequestException:
                time.sleep(wait)
                wait *= 2
        
        # If all retries fail, use fallback method
        return self.fallback_download(url)

    def fallback_download(self, url):
        try:
            response = requests.post(self.fallback_url, json={"url": url})
            if response.status_code == 200:
                content = response.json().get('body')
                if content:
                    self.fallback_success.value += 1
                    return type('Response', (), {'content': content, 'status_code': 200})()
            self.fallback_failure.value += 1
        except requests.RequestException:
            self.fallback_failure.value += 1
        return None

    def get_seasons(self, page):
        bs = BeautifulSoup(page.content, 'html.parser')
        seasons_div = bs.find('div', id='seasons')
        seasons = seasons_div.find_all('li')
        season_data = {}
        for season in seasons:
            season_name = season.text
            season_data[season_name] = f"https://cfbstats.com/{season_name}/team/index.html" if season_name != '2024' else self.base_url
        return season_data

    def get_teams(self, season_url):
        page = self.download_page(season_url)
        if not page:
            return []
        bs = BeautifulSoup(page.content, 'html.parser')
        content = bs.find('div', id='content')
        conferences = content.find_all('div', class_='conferences')

        teams = []
        for conference in conferences:
            conference_divs = conference.find_all('div', class_='conference')
            for conference_div in conference_divs:
                team_divs = conference_div.find_all('li')
                for team_div in team_divs:
                    team_name = team_div.text.strip().lower().replace(' ', '_').replace('(', '').replace(')', '')
                    team_url = team_div.find('a').get('href')
                    teams.append({
                        "team_name": team_name,
                        "page_url": f"https://cfbstats.com{team_url}"
                    })
        return teams
    
    def process_team(self, args):
        team, season = args
        return self.handle_get_team_stats_and_roster(team, season)
     
    def handle_get_team_stats_and_roster(self, team, season):
        page_url = team['page_url']
        roster_url = page_url.replace('index.html', 'roster.html')

        page = self.download_page(roster_url)
        if not page:
            logging.error(f"Failed to download roster for {team['team_name']}. Season: {season}")
            return None
        
        bs = BeautifulSoup(page.content, 'html.parser')
        roster_table = bs.find('table', class_='team-roster')

        if not roster_table:
            logging.error(f"No roster table found for {team['team_name']}. Season: {season}")
            return None

        players = []
        for row in roster_table.find_all('tr'):
            player = {}
            columns = row.find_all('td')
            if not columns:
                continue

            player['number'] = columns[0].text  
            player['name'] = columns[1].text
            player['position'] = columns[2].text
            player['year'] = columns[3].text
            player['height'] = columns[4].text
            player['weight'] = columns[5].text
            player['hometown'] = columns[6].text
            last_school = columns[7].text
            player['team_page_url'] = team['page_url']

            player['last_school'] = [part.strip() for part in last_school.split('/')]

            players.append(player)

        team['players'] = players
        return team
        
    def run_scrape(self):
        page = self.download_page()
        if page:
            seasons = self.get_seasons(page)
    
            # Reverse seasons list
            seasons = dict(reversed(list(seasons.items())))
    
            if seasons:
                all_seasons_teams = {}
                for season, url in tqdm(seasons.items(), desc="Seasons"):
                    all_seasons_teams[season] = self.get_teams(url)

                for season in all_seasons_teams:
                    season_data = []
                    print(f"Processing season: {season}")
    
                    teams = [(team, season) for team in all_seasons_teams[season]]
    
                    with Pool() as pool:
                        results = pool.map(self.process_team, teams)
    
                    for result in results:
                        if result:
                            season_data.append(result)
                        else:
                            print(f"Failed to process team: {result['team_name']}")
    
                    with open(f"{self.save_dir}/cfb_roster_{season}.json", 'w') as f:
                        json.dump(season_data, f, indent=4)
    
            print(f"\nScraping Summary:")
            print(f"Successfully processed by fallback method: {self.fallback_success.value}")
            print(f"Failed to process by fallback method: {self.fallback_failure.value}")