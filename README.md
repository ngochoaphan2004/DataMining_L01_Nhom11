# Crime Data Mining Project

## Description
This project applies data mining techniques to analyze crime data from Los Angeles. It includes data preprocessing, association rule mining using the Apriori algorithm, and crime type classification using Random Forest.

## Features
- **Data Preprocessing**: Cleans and preprocesses raw crime data, handles outliers, and prepares data for analysis.
- **Association Rule Mining**: Uses Apriori algorithm to discover frequent itemsets and association rules between crime categories, areas, and premises.
- **Crime Classification**: Employs Random Forest classifier to predict crime types based on features like time, location, and date.
- **Visualization**: Includes plots for model evaluation and results.

## Environment Setup
1. Ensure Python >= 3.11 is installed.
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
4. Create a `.env` file in the project root and set necessary environment variables (e.g., data file paths):
   ```
   DATA_PATH=C:\path\to\your\data\Crime_Data_from_2020_to_Present.csv
   OUTPUT_PATH=C:\path\to\output\
   ```

## Installation
1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. **Preprocessing**: Run `Preprocessing.py` to clean the raw data and generate `Crime_Data_Selected_Cleaned.csv`.
2. **Apriori Analysis**: Run `Apriori.py` to perform association rule mining on the cleaned data.
3. **Random Forest Classification**: Run `RandomForest.py` to train and evaluate the classification model.

Note: Ensure the input CSV file path is correctly set in the scripts.

## Files Description
- `Preprocessing.py`: Handles data cleaning, outlier removal, and feature engineering.
- `Apriori.py`: Implements Apriori algorithm for finding association rules.
- `RandomForest.py`: Builds and evaluates Random Forest model for crime type prediction.
- `requirements.txt`: Lists all Python dependencies.
- `Crime_Data_Selected_Cleaned.csv.csv`: Input file for Apriori.py and RandomForest.py.

## Requirements
- Python >= 3.11
- Libraries: numpy, pandas, matplotlib, mlxtend, scikit-learn, scipy, lightgbm

## Dataset
The project uses crime data from Los Angeles (Crime_Data_from_2020_to_Present.csv). Ensure the file is available in the specified path before running the scripts.