import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to check if a file is a text file
def is_text_file(file_path):
    text_extensions = ['.md', 'license', 'LICENSE']  # List of text file extensions
    return any(file_path.endswith(ext) for ext in text_extensions)

# Function to add commit data to the appropriate list
def add_commit_to_list(commit_list, commit_sha, filename, patch, committed_date, commit_type, commit_counter):
    commit_list.append({
        'number': commit_counter[commit_type],
        'commit_sha': commit_sha,
        'file': filename,
        'diff': patch,
        'committed_date': committed_date
    })
    commit_counter[commit_type] += 1

# Function to process each repository and year
def process_repository(repo, year, token):
    owner = repo['owner']
    repo_name = repo['name']
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'

    # GraphQL query
    query = """
    {
      repository(owner: "%s", name: "%s") {
        defaultBranchRef {
          target {
            ... on Commit {
              history(since: "%sT00:00:00Z", until: "%sT23:59:59Z") {
                edges {
                  node {
                    oid
                    messageHeadline
                    committedDate
                    author {
                      name
                      email
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    """ % (owner, repo_name, start_date, end_date)

    headers = {
        'Authorization': 'bearer ' + token,
        'Content-Type': 'application/json'
    }

    response = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)

    if response.status_code != 200:
        print(f"Query failed for {repo_name} in {year}. Status code: {response.status_code}")
        return

    commit_data = response.json()
    code_commits = []
    text_commits = []
    commit_counter = {'code': 1, 'text': 1}

    for edge in commit_data['data']['repository']['defaultBranchRef']['target']['history']['edges']:
        commit_sha = edge['node']['oid']
        committed_date = edge['node']['committedDate']
        commit_url = f'https://api.github.com/repos/{owner}/{repo_name}/commits/{commit_sha}'

        commit_response = requests.get(commit_url, headers=headers)
        if commit_response.status_code == 200:
            commit_details = commit_response.json()
            for file_data in commit_details['files']:
                if 'patch' in file_data:
                    if is_text_file(file_data['filename']):
                        add_commit_to_list(text_commits, commit_sha, file_data['filename'], file_data['patch'], committed_date, 'text', commit_counter)
                    else:
                        add_commit_to_list(code_commits, commit_sha, file_data['filename'], file_data['patch'], committed_date, 'code', commit_counter)

    directory = f'{repo_name}/{year}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f'{directory}/code_commits.json', 'w', encoding='utf-8') as file:
        json.dump(code_commits, file, indent=4)

    with open(f'{directory}/text_commits.json', 'w', encoding='utf-8') as file:
        json.dump(text_commits, file, indent=4)

    print(f"Commit data for {repo_name} in {year} saved to JSON files.")

# Main script
def main():
    # Replace with your new personal access token
    token = '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'

    # Load repositories from a JSON file
    with open('repositories.json', 'r') as file:
        data = json.load(file)

    # Define the starting number
    start_from_number = 410  # Change this to your desired starting number

    # Filter repositories starting from the specified number
    repositories = [{'name': repo['name'], 'owner': repo['owner']} for repo in data if repo['number'] >= start_from_number]

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_repository, repo, year, token) for repo in repositories for year in range(2018, 2023)]

        for future in as_completed(futures):
            future.result()  # This will also raise exceptions if any occurred

    print("All data processed.")

if __name__ == "__main__":
    main()
