# INTERPRETABLE-MACHINE-LEARNING-FOR-PREDICTING-HETEROGENEOUS-CATALYST-PERFORMANCE


BACKGROUND OF STUDY

Heterogeneous catalysis is essential for many industrial processes, from producing fertilizers to refining fuels and developing sustainable energy solutions like hydrogen production. The performance of a catalyst; how well it drives a reaction, how selective it is toward the desired product, and how long it lasts; depends on a mix of factors. These include the materials used (like metals or promoters), physical properties (such as surface area or particle size), and how the catalyst is made (e.g., temperature settings during preparation). Historically, finding the best catalyst has meant running countless experiments, relying on expertise, and a fair bit of trial and error. This approach is slow, expensive, and struggles to keep up with the complexity of modern catalyst data. 

Machine learning offers a way to speed things up by finding patterns in large datasets that humans might miss. The catch is that many machine learning models, especially complex ones like neural networks, act like black boxes; you get a prediction, but it’s hard to know why. In catalysis, where understanding what makes a catalyst work is just as important as predicting its performance, this lack of clarity is a problem. Interpretable models, like simpler linear regressions or tree-based methods with explainability tools, can solve this by not only predicting outcomes but also showing which factors matter most. For example, they can highlight whether a specific metal or preparation method is key to a catalyst’s success.This project tackles these challenges by building a machine learning pipeline to predict a catalyst’s performance, measured by the quality_score_0_1 in the catalyst_dataset.csv dataset. The goal is to create a model that’s both accurate and easy to understand, so researchers can use the results to guide real-world catalyst design. The pipeline covers everything from cleaning the data to selecting the best model and explaining its predictions. It’s designed to be clear, repeatable, and adaptable, making it useful for anyone working on catalysis, whether in a lab or a data science role.


PROJECT OVERVIEW

This repository contains a complete machine learning pipeline for predicting the performance of heterogeneous catalysts, based on the quality_score_0_1 metric. The project focuses on building a model that’s accurate and transparent, showing not just what the predictions are but why they were made. It includes steps for data cleaning, feature creation, model training, and interpretation, all organized in a way that’s easy to follow and reuse. The code is modular, well-documented, and intended for researchers, engineers, or data scientists who want to explore or build on this work.


DATASET DESCRIPTION

The catalyst_dataset.csv file includes 120,000 entries and 47 columns, covering a wide range of catalyst properties. The data captures:

A. PHYSICAL PROPERTIES: 

Measurements like support_surface_area (m²/g), particle_size (nm), pore_volume (cm³/g), and pore_diameter (nm), which describe the catalyst’s structure.

B. CHEMICAL COMPOSITION: 

Details like active_metal (e.g., platinum, palladium, nickel), promoter (e.g., lanthanum, cerium), binder_type, and loading amounts (metal_loading_wt_pct, promoter_loading_wt_pct).

C. SYNTHESIS CONDITIONS: 

Settings such as drying_temp (°C), calcination_temp (°C), reduction_temp (°C), and preparation_method (e.g., impregnation, co-precipitation).

D. PERFORMANCE METRICS: 

Outcomes like stability_hours, conversion_percent, selectivity_target_product_percent, and lifetime_cycles.

E. TARGET VARIABLE: 

The quality_score_0_1, a score from 0 to 1 that sums up the catalyst’s overall performance, based on its activity, selectivity, and stability.

The dataset has a mix of numerical and categorical data, with missing values in promoter (about 5% missing) and binder_type (about 3% missing). These gaps are handled during data preparation to ensure the model works well.


METHODOLOGY 

Data Science PipelineThe project follows a clear, 11-step process to ensure the analysis is thorough and repeatable:Data Loading: The catalyst_dataset.csv file is loaded into a pandas DataFrame for easy handling.

EXPLORATORY DATA ANALYSIS (EDA)

The dataset is explored to understand its structure, check for patterns, and spot issues. This includes plotting distributions, checking correlations, and identifying outliers or missing values.

DATA PREPROCESSING:

Categorical variables like support_material, preparation_method, and active_metal are converted into numbers using one-hot encoding or similar methods.
Missing values in promoter and binder_type are filled using practical approaches, like replacing them with the most common value or marking them as “missing.”
Numerical features are scaled to have similar ranges, so no single feature dominates the model.

FEATURE ENGINEERING

New features are created to improve the model, such as:

total_loading_wt_pct: Adding metal_loading_wt_pct and promoter_loading_wt_pct. Interaction terms, like combining support_surface_area and particle_size, to capture combined effects.


FEATURE SELECTION 

Recursive Feature Elimination (RFE) with a RandomForestRegressor picks the top 20 features to keep the model focused and efficient.

DATA SPLITTING 

The data is divided into training (80%) and testing (20%) sets, ensuring both sets represent the data well.

MODEL SELECTION

Two interpretable models are used:

A. Elastic Net: A linear model that balances simplicity and flexibility by combining two types of regularization.

B. Random Forest: A tree-based model that handles complex patterns, with tools to explain its predictions.

An XGBoost model is tested for comparison, but it’s less interpretable unless paired with explainability tools.


MODEL TRAINING

Models are trained on the training data, with settings fine-tuned using cross-validation to find the best performance.

MODEL EVALUATION

Models are tested using metrics like:

A. R-squared (R²): How much of the variation in the data the model explains.

B. Mean Squared Error (MSE): The average squared difference between predictions and actual values.

C. Mean Absolute Error (MAE): The average absolute difference between predictions and actual values.


MODEL INTERPRETATION

The results are explained using:

A. Elastic Net coefficients to show which features matter most in a linear model.

B. SHAP values to break down predictions and show how each feature contributes, both overall and for specific cases.

C. Feature importance rankings from the Random Forest model to double-check key factors.


SUMMARY AND CONCLUSION

The findings are summarized, highlighting the best model, key insights, and suggestions for improving catalysts.

Key Findings and Results

Model Performance

A. The Elastic Net model, with settings alpha=0.001 and l1_ratio=0.7, performed best.

Cross-validated R²: 0.887 ± 0.012, showing it generalizes well.

Test Set Metrics:R²: 0.887

MSE: 0.000049

MAE: 0.0055

B. The Random Forest model was close (R²: 0.874, MSE: 0.000052), but its complexity made it harder to interpret without extra tools. 

C. The XGBoost model scored slightly higher (R²: 0.891) but needed more computing power and explanation effort.


IINTERPRETABLE INSIGHTS

The Elastic Net coefficients and SHAP analysis pointed to these key factors 

A. driving quality_score_0_1:lifetime_cycles (Coefficient: 0.0122, SHAP Importance: 0.15): 

Catalysts that last through more cycles perform better, showing durability is crucial.


B. stability_hours (Coefficient: 0.0003, SHAP Importance: 0.09): 

Catalysts that stay stable longer score higher, highlighting the value of longevity.


C. cost_per_kg_usd (Coefficient: -0.0002, SHAP Importance: 0.07): 

More expensive catalysts tend to score lower, suggesting a balance between cost and quality.


D. selectivity_target_product_percent (Coefficient: 0.0019, SHAP Importance: 0.12): 

Catalysts that produce more of the desired product have higher scores, emphasizing selectivity.


E. promoter_La (Coefficient: 0.046, SHAP Importance: 0.08): 

Using lanthanum as a promoter boosts performance, likely because it stabilizes the catalyst’s active sites.


F. SHAP plots also showed how features like support_surface_area and particle_size work together to improve performance in non-obvious ways.


TECHNICAL STACK

The project uses a set of reliable Python tools:

A. Python: Version 3.8 or higher.

B. Pandas: For handling and cleaning data.

C. NumPy: For numerical calculations.

D. Scikit-learn: For building and testing the machine learning pipeline.

E. Matplotlib and Seaborn: For clear, professional-looking plots.

F. XGBoost: For testing a more advanced model.

G. SHAP: For explaining model predictions in detail.

H. Jupyter Notebook: For running and documenting the analysis interactively.


REPOSITORY STRUCTURE

LINK TO DATASET: https://drive.google.com/file/d/1oAIZLULMFAtd7DIw_906u60RlNHXU-6I/view?usp=drive_link

├── data/
│   └── catalyst_dataset.csv                     # Input dataset
├── notebooks/
│   └── catalyst_performance_prediction.ipynb    # Main analysis notebook
├── src/
│   └── utils.py                                # Helper functions for data processing and visualization
├── figures/
│   └── *.png                                   # Plots from EDA and model interpretation
├── requirements.txt                            # List of required libraries
├── README.md                                   # Project overview and instructions
└── LICENSE                                     # License details


LICENSE

This project is licensed under the MIT License. See the LICENSE file for details.

Contributing

We welcome contributions to improve the pipeline or add new ideas. To contribute:Fork the repository and create a new branch for your changes.

Make your updates, ensuring the code is clear and well-documented.

Submit a pull request, describing what you changed and why.

Report any bugs or suggestions by opening an issue.

Please keep contributions focused on making the pipeline more interpretable, efficient, or useful for catalysis research.


