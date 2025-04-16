Crop Yield Prediction Project
Overview
This project analyzes agricultural yield data to predict crop yields using machine learning. The dataset contains information on crop production across various states, districts, and seasons in India. The project involves data preprocessing, feature engineering, and training a Random Forest Regressor to predict yield, achieving high performance with an R² score of 0.9962 and RMSE of 0.0713.
The code is implemented in a Jupyter Notebook (Coders Crew.ipynb) using Python, leveraging libraries like Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.
Dataset
The dataset (cleaned_data.csv) contains the following columns:

State: State where the crop is grown (e.g., Uttar Pradesh, Tamil Nadu).
District: District within the state (e.g., Jalaun, Madurai).
Crop: Type of crop (e.g., Rice, Sugarcane).
Crop_Year: Year of cultivation (1997–2015 in the sample).
Season: Growing season (e.g., Kharif, Rabi, Whole Year).
Area: Cultivation area (in hectares).
Production: Total production (in tons or quintals).
Yield: Yield (Production / Area, in tons/hectare).

The dataset has 276,262 rows after preprocessing, with no missing values.
Project Workflow
1. Data Preprocessing

Loading: Loaded cleaned_data.csv using Pandas.
Yield Validation: Compared Yield with Calculated_Yield (Production / Area). Found 5,920 rows with minor discrepancies (e.g., 0.70 vs. 0.69). Used Calculated_Yield as the target (Target_Yield) for consistency.
Missing Values: Confirmed no missing values post-cleaning.
Distribution Analysis:
Target_Yield is highly skewed (skewness: 14.60). Applied log transformation (Log_Target_Yield) to stabilize variance.
Analyzed categorical feature distributions (e.g., Uttar Pradesh dominates with 35,725 rows).


Output: Saved distribution plot as target_yield_distribution.png.

2. Feature Engineering

Target Encoding: Encoded high-cardinality categorical features (State, District, Crop, Season) with their mean Target_Yield.
Interaction Feature: Created Crop_Season (e.g., Rice_Kharif) and label-encoded it.
Final Features:
Crop_Year, Area, Production, Target_Yield, Log_Target_Yield, State_target_encoded, District_target_encoded, Crop_target_encoded, Season_target_encoded, Crop_Season_encoded.


Output: Saved processed dataset as processed_data_for_training.csv, along with encoding mappings (target_encodings.pkl, crop_season_encoder.pkl).

3. Modeling

Model: Random Forest Regressor (n_estimators=100).
Target: Log_Target_Yield (due to skewness).
Train-Test Split: 80% training (221,009 rows), 20% testing (55,253 rows), stratified by Crop_Year.
Metrics:
RMSE: 0.0713 (low error relative to log-transformed target).
R²: 0.9962 (excellent fit, explaining 99.62% of variance).
MAE: 0.0308 (mean absolute error).


Feature Importance: Crop_target_encoded is the most important feature (84.26%), followed by Production (6.65%) and Area (3.97%).
Outputs:
Trained model saved as rf_regressor_model.pkl.
Predictions saved as regression_predictions.csv.



4. Visualizations

Regression Plot: Scatter plot of Yield vs. Production with regression line (saved in notebook output).
Distribution Plot: Histogram of Target_Yield with KDE (target_yield_distribution.png).

Requirements

Python: 3.12.7 (as used in the notebook).
Libraries:pip install pandas numpy matplotlib seaborn scikit-learn


Dataset: cleaned_data.csv (place in the project directory).

Setup and Running

Clone the Repository:git clone https://github.com/your-username/crop-yield-prediction.git
cd crop-yield-prediction


Install Dependencies:pip install -r requirements.txt

Or manually install the libraries listed above.
Prepare Data:
Ensure cleaned_data.csv is in the project directory. If not, generate it using the preprocessing steps in agriyield.ipynb (referenced in prior context).


Run the Notebook:
Open Coders Crew.ipynb in Jupyter Notebook:jupyter notebook


Execute all cells sequentially to preprocess data, train the model, and generate outputs.


Outputs:
processed_data_for_training.csv: Processed dataset.
target_yield_distribution.png: Yield distribution plot.
target_encodings.pkl, crop_season_encoder.pkl: Encoding mappings.
rf_regressor_model.pkl: Trained model.
regression_predictions.csv: Model predictions.



Results

Model Performance: The Random Forest Regressor achieves an R² of 0.9962 and RMSE of 0.0713, indicating excellent predictive accuracy for Log_Target_Yield.
Key Insight: Crop type (Crop_target_encoded) is the dominant predictor, suggesting crop-specific factors heavily influence yield.
Visualizations: The regression plot shows a positive relationship between Production and Yield, with some variability.

Future Improvements

Additional Features: Incorporate weather data (e.g., rainfall, temperature) or soil data to enhance predictions.
Model Exploration: Try XGBoost or LightGBM for potentially better performance.
Classification Task: Convert to a classification problem (e.g., high vs. low yield) for alternative use cases.
Hyperparameter Tuning: Optimize Random Forest parameters (e.g., max_depth, min_samples_split) using grid search.

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m 'Add feature').
Push to the branch (git push origin feature-name).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or feedback, reach out via GitHub issues or contact [your-email@example.com].
