import requests


api_key = "206c5b9549466e63683cb8aa04ea528b"

# NFL team abbreviations to full names mapping
nfl_teams = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders"
}

def get_qb_passing_yards_odds(matchup, qb_name):
    # Convert matchup abbreviation to full team name
    full_team_name = nfl_teams.get(matchup.upper())
    if not full_team_name:
        print(f"Invalid team abbreviation: {matchup}")
        return None

    # Get events
    events_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events?apiKey={api_key}"
    events_response = requests.get(events_url)
    
    if events_response.status_code != 200:
        print(f"Error fetching events: {events_response.status_code}")
        return None
    
    events = events_response.json()
    
    # Find the event ID for the given matchup
    event_id = None
    for event in events:
        if full_team_name in event['home_team'] or full_team_name in event['away_team']:
            event_id = event['id']
            break
    
    if not event_id:
        print(f"No event found for matchup: {matchup}")
        return None
    print(event_id)
    # Get odds for player passing yards using the specific endpoint
    odds_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds?apiKey={api_key}&regions=us&markets=player_pass_yds"
    odds_response = requests.get(odds_url)
    
    if odds_response.status_code != 200:
        print(f"Error fetching odds: {odds_response.status_code}")
        return None
    
    odds_data = odds_response.json()
    print(odds_data)
    # Find the odds for the specified quarterback
    for bookmaker in odds_data['bookmakers']:
        for market in bookmaker['markets']:
            if market['key'] == 'player_pass_yds':
                for outcome in market['outcomes']:
                    if qb_name.lower() in outcome['description'].lower():
                        return {
                            'player': outcome['description'],
                            'line': outcome['point'],
                            'over': outcome['price'],
                            'under': next(o['price'] for o in market['outcomes'] if o['name'] == 'Under')
                        }
    
    print(f"No odds found for QB: {qb_name}")
    return None

# Example usage:
# result = get_qb_passing_yards_odds("KC", "Patrick Mahomes")
# if result:
#     print(f"Odds for {result['player']}:")
#     print(f"Line: {result['line']} yards")
#     print(f"Over: {result['over']}")
#     print(f"Under: {result['under']}")