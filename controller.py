import os
import json

from scripts.scraping.our_lads.scrape_our_lads_teams import ScrapeOurLadsTeams
from scripts.scraping.our_lads.scrape_our_lads_depth_chart import ScrapeOurLadsDepthChart
from scripts.scraping.cfb_stats.scrape_cfb_roster import ScrapeCFBRoster
from scripts.scraping.cfb_stats.scrape_cfb_stats import ScrapeCFBStats

from scripts.processing.ProcessCFBRoster import ProcessCFBRoster
from scripts.processing.ProcessCFBStats import ProcessCFBStats

from scripts.engineering.QB.EngineerQBFeatures import EngineerQBFeatures

from scripts.machine_learning.QB.QBEnsembleLearning import QBEnsembleLearning
from scripts.machine_learning.QB.QBTimeSeriesApproach import QBTimeSeriesApproach


def main():
    dirs = [
        'data',
        'data/temp',
        'data/raw',
        'data/raw/stats',
        'data/raw/roster',
        'data/processed',
        'data/processed/rosters',
        'data/processed/stats',
        'data/ml_ready',
        'data/predictions',
        'data/visualizations'

    ]

    # ensure directories exist
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    #Begin scraping depth charts
    # teams_scraper = ScrapeOurLadsTeams(
    #     save_dir='data/raw/our_lads_teams.json'
    # )

    # # teams_scraper.run_scrape()

    # depth_chart_scraper = ScrapeOurLadsDepthChart(
    #     data_dir='data/raw/',
    #     save_dir='data/processed/'
    # )

    # # depth_chart_scraper.scrape_depth_charts()

    # scrape_cfb_roster = ScrapeCFBRoster(
    #     save_dir='data/raw/roster'
    # )

    # scrape_cfb_roster.run_scrape()

    # process_cfb_roster = ProcessCFBRoster(
    #     data_dir='data/raw/roster',
    #     save_dir='data/processed/rosters'
    # )

    # # process_cfb_roster.process_roster()

    #load positions & stats
    # positional_stats = {}
    # with open('positional_stats/positional_stats.json', 'r') as f:
    #     positional_stats = json.load(f)

    # scrape_cfb_stats = ScrapeCFBStats(
    #     data_dir='data/processed/rosters',
    #     save_dir='data/raw/stats',
    #     positional_stats=positional_stats
    # )


    # scrape_cfb_stats.run_scrape()

    # exit()

    # process_cfb_stats = ProcessCFBStats(
    #     data_dir='data/raw/stats',
    #     save_dir='data/processed/stats',
    #     processed_dir='data/processed',
    #     roster_dir='data/raw/roster'
    # )

    # process_cfb_stats.process_stats()

    # exit()

    engineer_qb_features = EngineerQBFeatures(
        data_path='data/processed/cfb_qb_stats.csv',
        save_dir='data/ml_ready'
    )

    # engineer_qb_features.run()

    # qb_model_training = QBEnsembleLearning(
    #     data_path='data/ml_ready/qb_features.csv',
    #     predictions_path='data/predictions/qb_transfer_predictions.csv',
    #     visualizations_path='data/visualizations/',
    #     stats_path='data/visualizations/model_stats.json'
    # )

    # qb_model_training.run()

    qb_time_series_approach = QBTimeSeriesApproach(
        data_path='data/ml_ready/qb_features_time_series.csv',
        output_path='data/predictions/'
    )

    qb_time_series_approach.run()



if __name__ == "__main__":
    main()