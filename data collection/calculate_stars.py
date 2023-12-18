import csv
import json
import os
from datetime import datetime

def calculate_yearly_stars(file_path):
    yearly_stars = {}
    last_recorded_stars_per_year = {}

    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row

        for row in reader:
            try:
                date_str = ' '.join(row[1].split()[:5])
                cumulative_stars = int(row[2])
                date = datetime.strptime(date_str, "%a %b %d %Y %H:%M:%S")
                year = date.year
                
                # Keep track of the last recorded star count for each year
                last_recorded_stars_per_year[year] = cumulative_stars

            except ValueError as e:
                print(f"Error parsing date: {row[1]} - {e}")

    # Now, calculate the net increase in stars for each year
    sorted_years = sorted(last_recorded_stars_per_year.keys())
    for i, year in enumerate(sorted_years):
        if i == 0:  # First year, no previous year to compare to
            yearly_stars[year] = last_recorded_stars_per_year[year]
        else:
            previous_year = sorted_years[i-1]
            yearly_stars[year] = last_recorded_stars_per_year[year] - last_recorded_stars_per_year[previous_year]

    return yearly_stars

# Directory containing the repositories
repo_dir = '1'

# Process each repository folder
for repo_name in os.listdir(repo_dir):
    repo_path = os.path.join(repo_dir, repo_name)
    if os.path.isdir(repo_path):
        csv_file = os.path.join(repo_path, 'star-history-20231216.csv')
        if os.path.exists(csv_file):
            stars_data = calculate_yearly_stars(csv_file)
            # Save the yearly stars data to a JSON file
            json_file = os.path.join(repo_path, 'yearly_stars.json')
            with open(json_file, 'w', encoding='utf-8') as jf:
                json.dump(stars_data, jf)
            print(f"Yearly stars data saved for {repo_name}")
        else:
            print(f"No star history file found for {repo_name}")
