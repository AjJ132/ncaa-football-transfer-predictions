import pandas as pd
from bs4 import BeautifulSoup

def handle_parse_page(page_content, stat_name):
    if stat_name == 'passing':
        return parse_passing_stats(page_content)
    elif stat_name == 'rushing':
        return parse_rushing_stats(page_content)
    elif stat_name == 'receiving':
        return parse_receiving_stats(page_content)
    elif stat_name == 'total':
        return parse_total_offense_stats(page_content)
    else:
        print(f"Stat name {stat_name} not found")
        return pd.DataFrame()

def parse_stats(page_content):
    bs = BeautifulSoup(page_content, 'html.parser')
    
    table = bs.find('table', class_='leaders')
    if not table:
        print("Table not found")
        return pd.DataFrame()

    # Get headers from the first row
    headers = [th.text.strip().lower().replace(' ', '_') for th in table.find_all('th')]
    
    # Parse data rows
    data = []
    for row in table.find_all('tr')[1:]:  # Skip the header row
        cols = row.find_all('td')
        if len(cols) == len(headers):
            player = {}
            for i, col in enumerate(cols):
                player[headers[i]] = col.text.strip()
            data.append(player)

    # Drop first column (rank)
    for player in data:
        player.pop('', None)
    
    # Remove 'Total' and 'Opponents' rows
    data = [player for player in data if player.get('name', '').strip().lower() not in ['total', 'opponents', 'team']]
    
    return pd.DataFrame(data)

def parse_rushing_stats(page_content):
    return parse_stats(page_content)

def parse_passing_stats(page_content):
    return parse_stats(page_content)

def parse_receiving_stats(page_content):
    return parse_stats(page_content)

def parse_total_offense_stats(page_content):
    return parse_stats(page_content)