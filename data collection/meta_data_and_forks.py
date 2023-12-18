import requests
import json
import os
from datetime import datetime

# GitHub Personal Access Token
token = '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'  # Replace with your token

# GraphQL URL
graphql_url = 'https://api.github.com/graphql'

# Headers for authentication
headers = {
    'Authorization': f'token {token}',
    'Content-Type': 'application/json'
}

def send_graphql_query(query):
    response = requests.post(graphql_url, json={'query': query}, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Query failed with status code {response.status_code}: {response.text}")

def fetch_repo_metadata(owner, repo_name, year):
    # GraphQL query
    query = f"""
    query {{
      repository(owner: "{owner}", name: "{repo_name}") {{
        forks {{
          totalCount
        }}
        languages(first: 1, orderBy: {{field: SIZE, direction: DESC}}) {{
          edges {{
            node {{
              name
            }}
          }}
        }}
        issues(filterBy: {{since: "{year}-01-01T00:00:00Z"}}, states: OPEN) {{
          totalCount
        }}
        pullRequests(states: OPEN) {{
          totalCount
        }}
        releases {{
          totalCount
        }}
        defaultBranchRef {{
          target {{
            ... on Commit {{
              history(since: "{year}-01-01T00:00:00Z") {{
                totalCount
              }}
            }}
          }}
        }}
      }}
    }}
    """
    data = send_graphql_query(query)

    # Extracting data from response
    repo_data = data['data']['repository']
    forks_count = repo_data['forks']['totalCount']
    main_language = repo_data['languages']['edges'][0]['node']['name'] if repo_data['languages']['edges'] else None
    issues_count = repo_data['issues']['totalCount']
    prs_count = repo_data['pullRequests']['totalCount']
    releases_count = repo_data['releases']['totalCount']
    commits_count = repo_data['defaultBranchRef']['target']['history']['totalCount']

    return {
        "year": year,
        "main_language": main_language,
        "commits": commits_count,
        "issues": issues_count,
        "pull_requests": prs_count,
        "releases": releases_count,
        "forks": forks_count
    }

def process_repository(repo_name):
    if repo_name in repo_owner_map:
        owner = repo_owner_map[repo_name]
        for year in range(2018, 2023):  # Loop through years 2018 to 2022
            try:
                metadata = fetch_repo_metadata(owner, repo_name, year)
                # Ensure the year directory exists
                os.makedirs(f'1/{repo_name}/{year}', exist_ok=True)
                # Save to a file in the respective year folder of the repository
                with open(f'1/{repo_name}/{year}/metadata_new.json', 'w') as file:
                    json.dump(metadata, file)
                print(f"Metadata saved for {repo_name} in year {year}")
            except Exception as e:
                print(f"Failed to process {repo_name} for year {year}: {e}")

# Read repositories and their owners from file
with open('repositories.json', 'r') as file:
    repo_data = json.load(file)
    repo_owner_map = {repo['name']: repo['owner'] for repo in repo_data}

# Process each repository
for repo_name in repo_owner_map.keys():
    process_repository(repo_name)
