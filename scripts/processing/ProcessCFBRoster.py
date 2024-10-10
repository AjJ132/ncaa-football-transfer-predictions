import json
import os
import pandas as pd
from tqdm import tqdm

class ProcessCFBRoster:
    def __init__(self, data_dir, save_dir):
        self.data_dir = data_dir
        self.save_dir = save_dir

    def process_season(self, season_data, season):
        flat_data = []
        for team in season_data:
            for player in team['players']:
                flat_data.append({
                    'team_name': team['team_name'],
                    'number': player['number'],
                    'name': player['name'],
                    'position': player['position'],
                    'year': player['year'],
                    'height': player['height'],
                    'weight': player['weight'],
                    'hometown': player['hometown'],
                    'last_schools': ', '.join(player['last_school'][::-1]),
                    'team_page_url': team['page_url']
                })

        df = pd.DataFrame(flat_data)
        df = df.sort_values(['team_name']).reset_index(drop=True)
        df.to_csv(f'{self.save_dir}/cfb_roster_{season}.csv', index=False)

    def process_roster(self):
        print("Processing Rosters")
        
        for file in tqdm(os.listdir(self.data_dir), desc="Processing seasons"):
            if file.startswith('cfb_roster_') and file.endswith('.json'):
                season = file.split('_')[-1].split('.')[0]
                with open(os.path.join(self.data_dir, file), 'r') as f:
                    season_data = json.load(f)
                self.process_season(season_data, season)

        print("Processing Complete")

