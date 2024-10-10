import requests
from bs4 import BeautifulSoup
import json

class ScrapeOurLadsTeams:
    def __init__ (self, save_dir):
        self.base_url = 'https://www.ourlads.com/ncaa-football-depth-charts/'
        self.save_dir = save_dir



    def download_page(self,):
        #travel to the base url and download the page
        page = requests.get(self.base_url)

        #check if the page downloaded successfully
        if page.status_code == 200:
            return page
        else:
            return None
        
    def parse_page(self, page):
        soup = BeautifulSoup(page.content, 'html.parser')
        main_content = soup.find('main')
        
        if not main_content:
            print("Main content not found")
            return None
        
        rows = main_content.find_all('div', class_='row')

        if not rows:
            print("Rows not found")
            return None
        
        teams = []
        for row in rows:

            
            #find all the divs with class = 'nfl-dc-mm-team'
            team_name_divs = row.find_all('div', class_='nfl-dc-mm-team')
            team_link_divs = row.find_all('div', class_='nfl-dc-mm-team-links')

            if not team_name_divs or not team_link_divs:
                continue

            #loop through the divs and send in the pairs of divs to the parse_team_div function
            for team_name_div, team_link_div in zip(team_name_divs, team_link_divs):

                team = self.parse_team_div(team_name_div, team_link_div)
                if team:
                    #append the team to the list of teams
                    teams.append(team)
                else:
                    continue

        
        # return 
        return teams

    def parse_team_div(self, team_name_div, team_link_div):
        #look for the team name and the depth chart link
        
        #team name will be raw text in div with class = 'nfl-dc-mm-team-name'
        team_name = team_name_div.find('div', class_='nfl-dc-mm-team-name')

        if team_name is None or team_name == '':
            print('No team name found')
            return None
        
        team_name = team_name.text

        #team depth chart link will be <a> tag with text = 'Depth Chart'
        depth_chart_link = team_link_div.find('a', text='Depth Chart')

        if depth_chart_link is None:
            print('Error finding depth chart link')
            return None
        
        #get the href attribute from the <a> tag
        depth_chart_url = depth_chart_link.get('href')

        return {
            'team_name': team_name,
            'depth_chart_url': self.base_url + depth_chart_url
        }
    
    def run_scrape(self,):
        page = self.download_page()

        if page:
            data = self.parse_page(page)

            if data:
                with open(self.save_dir, 'w') as f:
                    json.dump(data, f)
            else:
                print('Error parsing page')
        else:
            print('Error downloading page')