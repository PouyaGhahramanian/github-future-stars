import os
import json
import pandas as pd

def concatenate_diffs(data_list):
    return ' '.join(item['diff'] for item in data_list if 'diff' in item)

repo_dir = '1'
data = []

# Iterate through each repository
for repo_name in os.listdir(repo_dir):
    repo_path = os.path.join(repo_dir, repo_name)
    if os.path.isdir(repo_path):
        # Read the repository-wide yearly stars JSON file
        yearly_stars_path = os.path.join(repo_path, 'yearly_stars.json')
        yearly_stars_data = {}
        if os.path.exists(yearly_stars_path):
            with open(yearly_stars_path, 'r') as file:
                yearly_stars_data = json.load(file)

        for year in range(2018, 2023):
            year_path = os.path.join(repo_path, str(year))
            if os.path.isdir(year_path):
                code_commits_diff = text_commits_diff = ""
                metadata = {}

                for file_name in ['code_commits.json', 'text_commits.json', 'metadata.json']:
                    file_path = os.path.join(year_path, file_name)
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as file:
                            if file_name in ['code_commits.json', 'text_commits.json']:
                                data_list = json.load(file)
                                if file_name == 'code_commits.json':
                                    code_commits_diff = concatenate_diffs(data_list)
                                else:
                                    text_commits_diff = concatenate_diffs(data_list)
                            else:
                                metadata = json.load(file)

                row = {
                    'repository': repo_name,
                    'year': year,
                    'code_commits_diff': code_commits_diff,
                    'text_commits_diff': text_commits_diff,
                    'main_language': metadata.get('main_language', ''),
                    'contributors': metadata.get('contributors', 0),
                    'commits': metadata.get('commits', 0),
                    'issues': metadata.get('issues', 0),
                    'pull_requests': metadata.get('pull_requests', 0),
                    'releases': metadata.get('releases', 0),
                    'current_stars': yearly_stars_data.get(str(year - 1), 0),
                    'estimated_stars': yearly_stars_data.get(str(year), 0)
                }

                data.append(row)

# Create DataFrame
df = pd.DataFrame(data)
print(df)
# Save DataFrame as CSV
output_csv_file = 'repository_data.csv'
df.to_csv(output_csv_file, index=False)
print(f"Data saved to {output_csv_file}")
