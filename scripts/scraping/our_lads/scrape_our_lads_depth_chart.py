import json
import os
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

class ScrapeOurLadsDepthChart:
    def __init__(self, data_dir, save_dir):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.data = None

    def load_data(self):
        try:
            with open(os.path.join(self.data_dir, 'our_lads_teams.json'), 'r') as f:
                self.data = json.load(f)
            if not self.data:
                print('Error loading data: Empty file')
                return None
        except FileNotFoundError:
            print(f'Error loading data: File not found at {self.data_dir}')
            return None
        except json.JSONDecodeError:
            print('Error loading data: Invalid JSON format')
            return None
        
    def download_page(self, url):
        try:
            page = requests.get(url)
            page.raise_for_status()
            return page.content
        except requests.RequestException as e:
            print(f'Error downloading page {url}: {str(e)}')
            return None
        
    def parse_page(self, page):
        try:
            soup = BeautifulSoup(page, 'html.parser')
            content = soup.find('div', class_='page-content-inner', id='page-content-wrapper')
            if not content:
                raise ValueError("Content div not found")

            info_section = content.find('div', class_='blog-page dc-content-hdr')
            if not info_section:
                raise ValueError("Info section not found")

            data1 = info_section.find('div', id='ctl00_phContent_dData1')
            data2 = info_section.find('div', id='ctl00_phContent_dData2')
            if not data1 or not data2:
                raise ValueError("Data sections not found")

            info = data1.find('ul')
            conference = info.find('li', id='ctl00_phContent_liCONF').text
            city = info.find('li', id='ctl00_phContent_liCITY').text
            stadium = info.find('li', id='ctl00_phContent_liSTAD').text.split(':')
            stadium_name = stadium[1].strip() if len(stadium) > 1 else ""
            stadium_capacity = stadium[2].strip().replace('(', '').replace(')', '') if len(stadium) > 2 else ""

            info = data2.find('ul')
            head_coach = info.find('li', id='ctl00_phContent_liHC').text.split(':')[1].strip()
            offensive_schemes = info.find('li', id='ctl00_phContent_liOFF').text.split(':')[1].strip()
            defensive_schemes = info.find('li', id='ctl00_phContent_liDEF').text.split(':')[1].strip()

            depth_chart = content.find('div', class_='row', id='dc-wrapper')
            if not depth_chart:
                raise ValueError("Depth chart not found")

            table = depth_chart.find('table', class_='table table-bordered')
            if not table:
                raise ValueError("Depth chart table not found")

            rows = table.find('tbody').find_all('tr')

            current_scheme = ''
            players = []
            for row in rows:
                if row.get('class') == ['row-dc-pos']:
                    current_scheme = row.find('td').text
                    continue
                if row.get('class') == ['row-dc-pos-mobile']:
                    continue

                tds = row.find_all('td')
                index = 0
                player = {}
                current_position = ''
                for td in tds:
                    if index == 0:
                        current_position = td.text
                    elif index == 1:
                        player['number'] = td.text
                    elif index == 2:
                        a_tag = td.find('a')
                        if a_tag and a_tag.text.strip():
                            player['name'] = a_tag.text
                            player['player_link'] = a_tag.get('href')
                            name_parts = player['name'].rsplit(' ', 1)
                            player['player_name'] = name_parts[0]
                            player['player_class'] = name_parts[1].strip() if len(name_parts) > 1 else ""
                    
                    index += 1
                    if index == 3:
                        player['scheme'] = current_scheme
                        player['position'] = current_position
                        if player.get('player_name') != None: 
                            players.append(player)
                            
                        player = {}
                        index = 1

            return {
                'conference': conference,
                'city': city,
                'stadium_name': stadium_name,
                'stadium_capacity': stadium_capacity,
                'head_coach': head_coach,
                'offensive_schemes': offensive_schemes,
                'defensive_schemes': defensive_schemes,
                'players': players
            }

        except Exception as e:
            print(f"Error parsing page: {str(e)}")
            return None

    def process_team(self, team):
        url = team['depth_chart_url']
        page = self.download_page(url)
        if not page:
            return None
        data = self.parse_page(page)
        if data:
            team_name = team['team_name']
            #remove spaces on the end if any
            team_name = team_name.strip()

            #replace spaces with underscores
            team_name = team_name.replace(' ', '_')

            #set to lowercase
            team_name = team_name.lower()

            data['team_name'] = team_name

            #move team_name to the front of the list
            data = {**{'team_name': team_name}, **data}
            return data
        return None

    def scrape_depth_charts(self):
        if not self.data:
            self.load_data()
            if not self.data:
                return None
        
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(self.process_team, self.data), total=len(self.data)))
        
        valid_results = [result for result in results if result is not None]
        
        with open(os.path.join(self.save_dir, 'our_lads_depth_charts.json'), 'w') as f:
            json.dump(valid_results, f, indent=2)
        
        print(f"Scraped {len(valid_results)} out of {len(self.data)} teams successfully.")