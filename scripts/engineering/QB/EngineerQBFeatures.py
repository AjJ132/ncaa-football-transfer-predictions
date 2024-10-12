import pandas as pd
import numpy as np

class EngineerQBFeatures:

    def __init__(self, data_path, save_dir):
        self.data_path = data_path
        self.save_dir = save_dir

    def safe_pct_change(self, series):
        pct_change = series.pct_change(fill_method=None)  # Explicitly set fill_method to None
        # Handle inf: If old value was 0, and new value is not, set to 1 (100% increase)
        # If both old and new are 0, set to 0 (no change)
        pct_change = pct_change.mask(np.isinf(pct_change) & (series != 0), 1)
        pct_change = pct_change.mask(np.isinf(pct_change) & (series == 0), 0)
        return pct_change

    def calculate_g_diff(self, grouped):
        result = []
        for name, group in grouped:
            group = group.sort_values('season')
            group['g_diff'] = group['g'].diff()
            result.append(group)
        for i in range(len(result)):
            result[i]['g_diff'] = result[i]['g_diff'].fillna(0)
        return pd.concat(result, ignore_index=True)

    def calculate_pass_rating_perc(self, qb_data):
        qb_data['pass_rating_perc'] = qb_data.groupby(['season', 'team_name'])['passing_rating'].rank(pct=True)
        qb_data['pass_rating_perc'] = qb_data['pass_rating_perc'].apply(lambda x: round(x, 2))
        return qb_data
    
    def calculate_other_qbs(self, qb_data):
        qb_data['other_qbs'] = qb_data.groupby(['season', 'team_name'])['name'].transform(lambda x: x.nunique() - 1)
        return qb_data

    def calculate_playing_time_metrics(self, qb_data):
        # Assuming 'g' represents games played and we have a 'passing_att' column
        qb_data['attempts_per_game'] = qb_data['passing_att'] / qb_data['g']
        qb_data['playing_time_ratio'] = qb_data.groupby(['season', 'team_name'])['passing_att'].transform(lambda x: x / x.sum())
        return qb_data

    def calculate_yoy_performance_change(self, qb_data):
        for stat in ['passing_rating', 'passing_att', 'passing_yards', 'rushing_yards', 'passing_td', 'offense_plays', 'offense_total_yards', 'offense_yards/play', 'offense_yards/g']:
            qb_data[f'{stat}_yoy_change'] = qb_data.groupby('name').apply(lambda x: self.safe_pct_change(x[stat])).reset_index(level=0, drop=True)
            
            # Fill NaN values with the median change for that season (using median instead of mean to be more robust to outliers)
            season_median_change = qb_data.groupby('season')[f'{stat}_yoy_change'].transform('median')
            qb_data[f'{stat}_yoy_change'] = qb_data[f'{stat}_yoy_change'].fillna(season_median_change)
            
            # If still NaN (e.g., for the first season in the dataset), fill with -999
            qb_data[f'{stat}_yoy_change'] = qb_data[f'{stat}_yoy_change'].fillna(-999)
        
        return qb_data

    def calculate_team_performance(self, qb_data):
        qb_data['team_win_pct'] = qb_data['season_wins'] / (qb_data['season_wins'] + qb_data['season_losses'])
        qb_data['team_win_pct_change'] = qb_data.groupby('team_name')['team_win_pct'].transform(self.safe_pct_change)
        
        # Fill NaN values with the median change for that season
        season_median_change = qb_data.groupby('season')['team_win_pct_change'].transform('median')
        qb_data['team_win_pct_change'] = qb_data['team_win_pct_change'].fillna(season_median_change)
        
        # If still NaN (e.g., for the first season in the dataset), fill with -999
        qb_data['team_win_pct_change'] = qb_data['team_win_pct_change'].fillna(-999)
        
        return qb_data

    def estimate_depth_chart_position(self, qb_data):
        # Estimate depth chart position based on attempts
        qb_data['depth_chart_position'] = qb_data.groupby(['season', 'team_name'])['passing_att'].rank(ascending=False, method='min')
        return qb_data
    
    def engineer_qb_features(self):
        qb_data = pd.read_csv(self.data_path)
        grouped = qb_data.groupby('name')

        qb_data = self.calculate_g_diff(grouped)
        qb_data = self.calculate_pass_rating_perc(qb_data)
        qb_data = self.calculate_other_qbs(qb_data)
        qb_data = self.calculate_playing_time_metrics(qb_data)
        qb_data = self.calculate_team_performance(qb_data)
        qb_data = self.estimate_depth_chart_position(qb_data)
        qb_data = self.calculate_yoy_performance_change(qb_data)

        return qb_data
    
    def prepare_and_export_time_series_data(self, qb_data):
        columns_to_keep = [
            'name', 'team_name', 'season', 'g', 'g_diff', 'passing_rating', 'pass_rating_perc', 'other_qbs',
            'attempts_per_game', 'playing_time_ratio', 'team_win_pct', 'team_win_pct_change',
            'depth_chart_position', 'passing_rating_yoy_change', 'passing_att_yoy_change',
            'passing_yards_yoy_change', 'passing_td_yoy_change', 'transfer', 'offense_plays', 'offense_total_yards',
            'offense_yards/play', 'offense_yards/g', 'offense_plays_yoy_change', 'offense_total_yards_yoy_change', 'offense_yards/play_yoy_change', 'offense_yards/g_yoy_change'
        ]

        qb_data = qb_data[columns_to_keep]
        qb_data = qb_data.sort_values(['name', 'season'])
        qb_data['next_team'] = qb_data.groupby('name')['team_name'].shift(-1)
        qb_data['transferred_next_year'] = ((qb_data['team_name'] != qb_data['next_team']) & 
                                            (qb_data['next_team'].notna())).astype(int)
        qb_data = qb_data.drop('next_team', axis=1)
        
        last_seasons = qb_data.groupby('name')['season'].transform('max')
        # Ensure 'transfer' is of the same dtype as 'transferred_next_year'
        qb_data['transfer'] = qb_data['transfer'].astype(int)
        qb_data.loc[qb_data['season'] == last_seasons, 'transferred_next_year'] = qb_data.loc[qb_data['season'] == last_seasons, 'transfer']

        # Ensure all float columns are finite
        float_columns = qb_data.select_dtypes(include=['float64']).columns
        for col in float_columns:
            qb_data[col] = pd.to_numeric(qb_data[col], errors='coerce')
            qb_data[col] = qb_data[col].replace([np.inf, -np.inf], np.nan)
            qb_data[col] = qb_data[col].fillna(qb_data[col].median())

        # Round float columns to two decimal places
        for col in float_columns:
            qb_data[col] = qb_data[col].round(2)

        return qb_data

    def run(self):
        qb_data = self.engineer_qb_features()
        qb_data.to_csv(f"{self.save_dir}/qb_features.csv", index=False)

        qb_data = self.prepare_and_export_time_series_data(qb_data)
        qb_data.to_csv(f"{self.save_dir}/qb_features_time_series.csv", index=False)

        print(f"Engineered features saved to {self.save_dir}/qb_features.csv")
        return qb_data