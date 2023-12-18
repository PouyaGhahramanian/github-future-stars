# github-future-stars
**Forecasting Future Stars**: A Predictive Model For GitHub Repository Popularity

This repository contains the implementation of our predictive model (`StarHub`) for GitHub repository popularity in the future using PyTorch and the Transformers library. It also includes an additional `Featurehub` model that predicts the popularity of a repository by using only the meta-data of that repository.

This project is our final project for the **Bilkent CS 588: Data Science for Software Engineering** Course. We extend our gratitude to **Professor Eray Tüzün** for his invaluable guidance and to our Teaching Assistant, **Muhammad Umair Ahmed**, for his insightful comments and support throughout all stages of the project.

**Group 8, Members**:
- Sepehr Bakhshi
- Pouya Ghahramanian
- Mehmet Kadri Gofralilar
- Kousar Kousar

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [MetaDataEmbedding](#metadataembedding)
  - [Starhub](#starhub)
  - [Featurehub](#featurehub)
- [Data Collection](#data-collection)
- [Dataset](#dataset)
- [Running Experiments](#running-experiments)
- [Results](#results)
  - [Training MSE Plots](#training-mse-plots)
  - [Test MSE Results](#test-mse-results)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)

## Installation

You need to install the following dependencies in Python3 for this project.

```bash
pip3 install pandas scikit-learn torch transformers
```

## Usage
### MetaDataEmbedding
This module is used for embedding metadata such as year, language, and numerical features. It is used in our StarHub and FeatureHub models.

```python
from models import MetaDataEmbedding

# Initialize MetaDataEmbedding
embedding = MetaDataEmbedding(num_years, num_languages, num_numerical_features, embedding_size)
```

### StarHub
Starhub combines metadata, text, and code embeddings to predict the popularity of a repository in the future.

```python
from models import Starhub

# Initialize Starhub
starhub = Starhub(num_years, num_languages, meta_data_input_size, meta_data_embedding_size, output_size, mode='regression')
```

### FeatureHub
A simplified version of Starhub focusing on metadata and current stars.

```python
from models import Featurehub

# Initialize Featurehub
featurehub = Featurehub(num_years, num_languages, meta_data_input_size, meta_data_embedding_size, output_size, mode='regression')
```

### Data Collection

Our code for collecting data is located in the `data collection` folder. This folder contains scripts and instructions for gathering and preprocessing the dataset used in our experiments. Please refer to this folder for detailed steps on how to collect and prepare the data.

#### Dataset Structure

The dataset is structured with the following columns:

- `repository`: The name of the GitHub repository.
- `year`: The year of data collection.
- `code_commits_diff`: Differences in code between commits.
- `text_commits_diff`: Differences in text (documentation, comments, etc.) between commits.
- `main_language`: The primary programming language of the repository.
- `commits`: The number of commits made in the repository within the year.
- `issues`: The number of issues opened in the repository within the year.
- `pull_requests`: The number of pull requests made in the repository within the year.
- `releases`: The number of releases published in the repository within the year.
- `current_stars`: The number of stars the repository has at the start of the respective year.
- `future_stars`: The number of stars for the repository at the end of the year.

For more detailed information on how each column is generated and processed, please refer to the scripts and documentation in the `data collection` folder.

### Dataset

#### Overview

The dataset used in our project consists of GitHub repository data spanning over 5 years, with a focus on the following aspects:

- **Data Size**: Originally includes data from 91 repositories.
- **Languages**: Covers 19 different main programming languages:
    - JavaScript, C++, Go, Java, HTML, Kotlin, TypeScript, Unknown, Python, Rust, PHP, Shell, SCSS, C#, Swift, Dockerfile, C, FreeMarker, Vue.
- **Filtered Data**: After removing entries with unknown future stars count, the dataset size is reduced to 359 entries.
- **Input Features**: Includes the number of current stars, metadata, and text and code from commits.
- **Target Output**: Predicts the stars count of the repository for the next year (future stars field of the repository).

#### Detailed Breakdown

- **Number of Repositories per Language**:
    - TypeScript: 68, Python: 49, JavaScript: 40, Unknown: 39, C++: 30, Java: 24, Go: 18, Rust: 18, C: 10, HTML: 10, Dockerfile: 8, Swift: 8, Vue: 7, SCSS: 5, Shell: 5, Kotlin: 5, PHP: 4, FreeMarker: 3.
- **Train-and-Test Split**:
    - Data before 2022 is used for training, and data after 2022 is used for testing.
    - Train size: 298 entries, Test size: 61 entries (16.99% of the total).

#### Statistical Details

- **Current Stars**:
    - Minimum: 0
    - Maximum: 37,290
    - Mean: 6,730.11
- **Future Stars**:
    - Minimum: 420
    - Maximum: 39,960
    - Mean: 10,742.34

#### Distribution Plots

##### Current Stars Distribution
![Current Stars Distribution Plot](/figs/current_stars_distribution.jpg)

##### Future Stars Distribution
![Future Stars Distribution Plot](/figs/future_stars_distribution.jpg)

### Running Experiments

Use `main.py` to run experiments. To execute the script, run the following command in your terminal:

```bash
python3 main.py
```
The script includes the necessary code to prepare and process the dataset, tokenize text and code, create datasets and dataloaders, define the models, and perform training and testing.

### Results

#### Training MSE Plots

##### Starhub Model
![Starhub Training MSE Plot](/figs/mse_error_starhub.jpg)

##### Featurehub Model
![Featurehub Training MSE Plot](/figs/mse_error_featurehub.jpg)

#### Test MSE Results

- **Optimizer**: Adam
- **Learning Rate**: 1e-3
- **Epochs**: 15
- **Meta-data Embedding Size**: 32
- **Text and Code Embedding Size**: 768
- **Classification Head**: Linear Layer ((32 + 768 + 768), 1)

##### Starhub Model
- **Test MSE**: 0.0005929418730374891

The Starhub model shows a low test MSE, indicating good generalization on the test data.

##### Featurehub Model
- **Initial Test MSE (15 epochs)**: 0.30648621916770935
- **Improved Test MSE (5 epochs)**: 0.2335744183510542

The initial test MSE for Featurehub suggested overfitting. To mitigate this, an early stop training approach was adopted, which improved the test MSE significantly when training was stopped at 5 epochs.

#### Observations

- The Starhub model demonstrates strong performance with low test MSE.
- The Featurehub model initially showed signs of overfitting. However, with early stopping at 5 epochs, the test MSE improved, indicating better generalization.
- The training plots for both models further clarify their learning behaviors during the training process.
















