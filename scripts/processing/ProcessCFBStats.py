import json
import os
import pandas as pd
import numpy as np

class ProcessCFBStats:
    def __init__(self, data_dir, save_dir, processed_dir, roster_dir):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.processed_dir = processed_dir
        self.roster_dir = roster_dir
        self.seasons = set()
        self.stat_types = set()
        self.team_records = []

    def combine_positional_stats(self):
        # Get list of all files in data_dir
        files = os.listdir(self.data_dir)

        # Initialize empty dictionary to store DataFrames for each stat type
        combined_dfs = {}

        for file in files:
            # Extract information from filename
            parts = file.split('_')
            if len(parts) != 4 or not parts[2].isdigit() or not parts[3].endswith('.csv'):
                continue  # Skip files that don't match the expected format

            season = parts[2]
            stat_type = parts[3].replace('.csv', '')

            self.seasons.add(season)
            self.stat_types.add(stat_type)

            # Read file into DataFrame
            df = pd.read_csv(os.path.join(self.data_dir, file))

            # Add season column
            df['season'] = season

            # Combine DataFrames by stat type
            if stat_type in combined_dfs:
                combined_dfs[stat_type] = pd.concat([combined_dfs[stat_type], df], ignore_index=True)
            else:
                combined_dfs[stat_type] = df

        # Save each DataFrame to a separate file
        for stat_type, df in combined_dfs.items():
            output_path = os.path.join(self.save_dir, f'cfb_stats_{stat_type}.csv')
            df.to_csv(output_path, index=False)
            print(f"Saved {stat_type} stats to {output_path}")


    def generate_player_ids(self, df):
        # Generate unique player IDs by combining player name and team
        df['player_id'] = df['name'].str.lower() + '_' + df['team_name'].str.lower()
        df['player_id'] = df['player_id'].str.replace(' ', '_')
        return df
    
    
    def identify_transfers(self, dfs):
        # Check the type of dfs
        print(f"Type of dfs: {type(dfs)}")
        
        # Ensure dfs is a list of DataFrames
        if not isinstance(dfs, list) or not all(isinstance(df, pd.DataFrame) for df in dfs):
            raise TypeError("dfs must be a list of pandas DataFrames")
        
        # Create a combined DataFrame with only 'name', 'season', and 'team_name'
        combined_df = pd.concat([df[['name', 'season', 'team_name']].copy() for df in dfs], ignore_index=True)
        
        # Sort by 'name' and 'season'
        combined_df = combined_df.sort_values(by=['name', 'season'])
        
        # Set all transfer status to False initially
        combined_df['transfer'] = False
        
        # Group by player name
        grouped = combined_df.groupby('name')
        
        print(f"Number of rows: {len(combined_df)}")
        
        # Iterate over each group
        for name, group in grouped:
            # Check if the player has multiple teams
            if len(group['team_name'].unique()) > 1:
                # Ensure group is ordered by season
                group = group.sort_values(by='season')
                
                # Combine all rows where season and team_name are the same
                group = group.groupby(['season', 'team_name']).agg({'name': 'first'}).reset_index()
                
                # Loop over each row in group
                season = 0
                team_name = ''
                for index, row in group.iterrows():
                    # Set initial values
                    if team_name == '' or season == 0:
                        season = row['season']
                        team_name = row['team_name']
                        continue
                    
                    # Check if the player transferred
                    if row['team_name'] != team_name:
                        # Set transfer status to True for the previous year
                        combined_df.loc[(combined_df['name'] == name) & (combined_df['season'] == season), 'transfer'] = True
                    
                    # Update season and team_name
                    season = row['season']
                    team_name = row['team_name']

        # TEMP: save merged df to csv
        # combined_df.to_csv('combined_df.csv', index=False)
        
        # Merge the transfer information back into the original DataFrames
        updated_dfs = []
        for df in dfs:
            # Merge only the 'transfer' column from combined_df
            updated_df = pd.merge(df, combined_df[['name', 'season', 'team_name', 'transfer']], 
                                on=['name', 'season', 'team_name'], 
                                how='left')
            updated_df['transfer'] = updated_df['transfer'].fillna(False)
            updated_df['transfer'] = updated_df['transfer'].astype(bool)
            updated_dfs.append(updated_df)
        
        # Return the updated DataFrames
        return updated_dfs
    
    def prepare_quarterbacks_ml_data(self):
        # Load the data from the save directory
        passing_data = pd.read_csv(os.path.join(self.save_dir, 'cfb_stats_passing.csv'))
        rushing_data = pd.read_csv(os.path.join(self.save_dir, 'cfb_stats_rushing.csv'))
        offense_data = pd.read_csv(os.path.join(self.save_dir, 'cfb_stats_total.csv'))

        # Rename columns
        passing_data = passing_data.rename(columns=lambda x: 'passing_' + x if x not in ['name', 'yr', 'pos', 'g', 'team_name', 'season', 'player_id', 'transfer'] else x)
        rushing_data = rushing_data.rename(columns=lambda x: 'rushing_' + x if x not in ['name', 'yr', 'pos', 'g', 'team_name', 'season', 'player_id', 'transfer'] else x)
        offense_data = offense_data.rename(columns=lambda x: 'offense_' + x if x not in ['name', 'yr', 'pos', 'g', 'team_name', 'season', 'player_id', 'transfer'] else x)

        # Join on name, team_name, season
        qb_data = pd.merge(passing_data, rushing_data, on=['name', 'team_name', 'pos', 'season', 'player_id', 'transfer', 'yr', 'g'], how='outer')
        qb_data = pd.merge(qb_data, offense_data, on=['name', 'team_name', 'pos', 'season', 'player_id', 'transfer', 'yr', 'g'], how='outer')

        # Filter out where position is not QB
        qb_data = qb_data[qb_data['pos'] == 'QB']

        # Sort by name and season
        qb_data = qb_data.sort_values(['name', 'season'])

        # Reorder columns
        columns_order = ['transfer', 'name', 'season', 'pos', 'team_name'] + [col for col in qb_data.columns if col not in ['transfer', 'name', 'season', 'pos', 'team_name']]
        qb_data = qb_data[columns_order]

        # Load team records
        team_records_df = pd.read_csv(os.path.join(self.processed_dir, 'cfb_team_records.csv'))

        # Merge QB data with team records
        qb_data = pd.merge(qb_data, team_records_df[['team_name', 'season', 'team_wins', 'team_losses']], 
                        on=['team_name', 'season'], how='left')

        # Rename columns to season_wins and season_losses
        qb_data = qb_data.rename(columns={'team_wins': 'season_wins', 'team_losses': 'season_losses'})


        # Convert years to numeric
        year_mapping = {'FR': 1, 'SO': 2, 'JR': 3, 'SR': 4}
        qb_data['yr'] = qb_data['yr'].map(year_mapping)
        
        # Handle any remaining non-numeric values
        qb_data['yr'] = pd.to_numeric(qb_data['yr'], errors='coerce')
        
        # Fill NaN values with a default value (e.g., 0 or the mean of the column)
        qb_data['yr'] = qb_data['yr'].fillna(0)
        
        # Now convert to integer
        qb_data['yr'] = qb_data['yr'].astype(int)

        # Save to ml_ready directory
        qb_data.to_csv(os.path.join(self.processed_dir, 'cfb_qb_stats.csv'), index=False)

        return qb_data

    def generate_team_loss_win_records(self,):
        #load all files from roster directory with cfb_roster in the name
        files = [f for f in os.listdir(self.roster_dir) if 'cfb_roster' in f]

        #iterate over each file
        for file in files:
            #load the file
            with open(os.path.join(self.roster_dir, file), 'r') as f:
                data = json.load(f)

            #get season from file name 'cfb_roster_{season}.json'
            season = file.split('_')[2].replace('.json', '')

            #iterate over each team
            for team in data:
                #get the team name
                team_name = team['team_name']

                #get schedule
                schedule = team['schedule']

                #loop over each item in schedule, if result contians W or L add to team_records
                #if result doesnt contain W or L, then skip

                team_losses = 0
                team_wins = 0
                
                for game in schedule:
                    #check if the result contains W or L
                    if 'W' in game['result'] or 'L' in game['result']:
                        if 'L' in game['result']:
                            team_losses += 1
                        else:
                            team_wins += 1

                #append to team_records
                self.team_records.append({
                    'team_name': team_name,
                    'season': season,
                    'team_wins': team_wins,
                    'team_losses': team_losses
                })

        #save team_records to a csv
        team_records_df = pd.DataFrame(self.team_records)
        team_records_df.to_csv(os.path.join(self.processed_dir, 'cfb_team_records.csv'), index=False)

    def process_stats(self):
        # Combine positional stats into separate DataFrames and save them
        self.combine_positional_stats()

        print(f"Processed seasons: {sorted(self.seasons)}")
        print(f"Processed stat types: {sorted(self.stat_types)}")

        # Convert stat_types to a list for indexing
        stat_types_list = list(self.stat_types)

        # Load each positional stat type and process
        for stat_type in stat_types_list:
            file_path = os.path.join(self.save_dir, f'cfb_stats_{stat_type}.csv')
            df = pd.read_csv(file_path)
            df = self.generate_player_ids(df)

            # Save updated DataFrame
            df.to_csv(file_path, index=False)
            print(f"Generated player IDs and identified transfers for {stat_type} stats")

        #TEMP: load all dfs into transfer function
        dfs = [pd.read_csv(os.path.join(self.save_dir, f'cfb_stats_{stat_type}.csv')) for stat_type in stat_types_list]

        # Identify transfers
        dfs = self.identify_transfers(dfs)

        # save updated dfs
        for i, df in enumerate(dfs):
            #merge duplicate rows
            df = df.drop_duplicates(subset=['name', 'team_name', 'season'], keep='first')

            #group by name
            df = df.groupby('name').apply(lambda x: x.sort_values('season')).reset_index(drop=True)

            file_path = os.path.join(self.save_dir, f'cfb_stats_{stat_types_list[i]}.csv')
            df.to_csv(file_path, index=False)
            print(f"Identified transfers for {stat_types_list[i]} stats and saved to {file_path}")

        # Generate team loss and win records
        self.generate_team_loss_win_records()

        # Prepare data for quarterbacks
        qb_data = self.prepare_quarterbacks_ml_data()

        print("Processing complete.")

