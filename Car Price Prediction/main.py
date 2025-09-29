# %%
# ----------------------------------------------------
# 1. MODEL TRAINING & SAVING
# ----------------------------------------------------
from sklearn.linear_model import LinearRegression
from data_carwrar import Feat, Targ
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from imblearn.pipeline import Pipeline  # instead of sklearn.pipeline.Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
# Load your data

import random
import joblib as jl

# --- FIX 2: Specify the correct column index for your single feature ---
numerical_features = [0, 1]
categotirical_features = []

preprocessor = ColumnTransformer(
    transformers=[
        ('num', QuantileTransformer(), numerical_features),
        ("col", OneHotEncoder(), categotirical_features)
    ])

pipe = Pipeline([
    ("model", SVC(probability=True))
])

# Encode the target variable
r = random.randint(1, 1000)
print(r)
# --- FIX 1: Use 'stratify' to ensure class distribution is maintained ---
# This guarantees your rare class (with 1 sample) is in the training set.
X_train, X_test, y_train, y_test = train_test_split(
    Feat, Targ, test_size=0.1, random_state=420
)

# --- PARAMETER GRID ---
param_grid = [

    {
        'model': [RandomForestRegressor()],
        'model__n_estimators': [600, 500, 400, 1000],
        'model__max_depth': [3, 5, 7],
        'model__max_leaf_nodes': [v for v in range(2, 100, 10)]
    }

]

pg = [{
    'model': [LinearRegression()],
}
]
# Set up the grid search
# Note: With only 1 sample of a class, cv cannot be more than 1.
# GridSearchCV is less meaningful here, but this will allow the code to run.
# The real issue is the lack of data for the minority class.
mod1 = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=4,  # This will still fail as stratify can't create 2 folds from 1 sample
    n_jobs=-1,
    error_score='raise'
)

estimators = [
    ('lr', LinearRegression(n_jobs=-1)),
    ('rfr', RandomForestRegressor(max_depth=42, n_estimators=13, max_leaf_nodes=102)),
    ('xgb', XGBRegressor()),
    ('xsgb', XGBRegressor()),
    ('xsdgb', XGBRegressor()),

]

# Create the VotingClassifier
# Use voting='soft' to use predicted probabilities
mod = VotingRegressor(estimators=estimators)
try:
    mod.fit(X_train, y_train)

    # Print the results
    print(f"Test set accuracy: {mod.score(X_test, y_test):.4f}")
except ValueError as e:
    print(f"Error during fitting: {e}")

jl.dump(mod, "model.pkl")
# %%

# ----------------------------------------------------
# 2. GRADIO APP
# ----------------------------------------------------
import pandas as pd
import gradio as gr
from gradio_rangeslider import RangeSlider
import joblib as jl
import numpy as np
from data_carwrar import Feat

Model = jl.load(r"C:\Users\SOHAM\PycharmProjects\PythonProject\model.pkl")

ALL_BRANDS = ['audi', 'bmw', 'chevrolet', 'datsun', 'fiat', 'ford', 'honda', 'hyundai', 'isuzu', 'jeep', 'mahindra',
              'maruti', 'mercedes', 'nissan', 'opel',
              'renault', 'skoda', 'ssangyong', 'tata', 'toyota', 'volkswagen', 'volvo']

ALL_BODYTYPES = ['hatchback', 'sedan', 'suv', 'luxury sedan', 'luxury suv']

ALL_FUEL_TYPES = ['petrol', 'diesel', 'electric', 'petrol & cng', 'petrol & lpg']

ALL_TRANSMISSIONS = ['manual', 'automatic']

ALL_CITIES = ['ahmedabad', 'bengaluru', 'chennai', 'faridabad', 'ghaziabad', 'gurgaon', 'hyderabad', 'kolkata',
              'lucknow', 'mumbai', 'new delhi', 'noida', 'pune']

ALL_WARRANTY_AVAILS = ['yes', 'no']

ALL_SOURCES = ['inperson_sale', 'online']

ALL_RATINGS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1,
               2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5,
               3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]

ALL_AGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

ALL_AVAILABILITIES = ['in_stock', 'in_transit', 'out_of_stock', 'pickup_pending']

FINAL_MODEL_COLUMNS = [
    'kms_run',
    'times_viewed',
    'assured_buy',
    'is_hot',
    'total_owners',
    'car_rating',
    'fitness_certificate',
    'reserved',
    'warranty_avail',
    'body_type_hatchback',
    'body_type_luxury sedan',
    'body_type_luxury suv',
    'body_type_sedan',
    'body_type_suv',
    'car_brand_audi',
    'car_brand_bmw',
    'car_brand_chevrolet',
    'car_brand_datsun',
    'car_brand_fiat',
    'car_brand_ford',
    'car_brand_honda',
    'car_brand_hyundai',
    'car_brand_isuzu',
    'car_brand_jeep',
    'car_brand_mahindra',
    'car_brand_maruti',
    'car_brand_mercedes',
    'car_brand_nissan',
    'car_brand_opel',
    'car_brand_renault',
    'car_brand_skoda',
    'car_brand_ssangyong',
    'car_brand_tata',
    'car_brand_toyota',
    'car_brand_volkswagen',
    'car_brand_volvo',
    'fuel_type_diesel',
    'fuel_type_electric',
    'fuel_type_petrol',
    'fuel_type_petrol & cng',
    'fuel_type_petrol & lpg',
    'car_availability_in_stock',
    'car_availability_in_transit',
    'car_availability_out_of_stock',
    'car_availability_pickup_pending',
    'city_ahmedabad',
    'city_bengaluru',
    'city_chennai',
    'city_faridabad',
    'city_ghaziabad',
    'city_gurgaon',
    'city_hyderabad',
    'city_kolkata',
    'city_lucknow',
    'city_mumbai',
    'city_new delhi',
    'city_noida',
    'city_pune',
    'car_age',
    'transmission_automatic',
    'transmission_manual',
    'ad_age_days',
    'source_inperson_sale',
    'source_online'
]

custom_css1 = """
.custom-heading-label {
    background-color: #2e8b57; /* A deep green color */
    color: white;             /* White text */
    font-weight: bold;
    padding: 8px 12px;
    border-radius: 6px;
    display: inline-block; /* Makes the background fit the text width */
    margin-bottom: 10px;
}
"""

custom_css2 = """
#fare-input .value-display::after {
  content: "£";
  margin-left: 5px; /* Adds a small space between the number and the symbol */
}
"""

ad_age_days = Feat["ad_age_days"].mean()
times_viewed = Feat["times_viewed"].mean()
assured_buy = Feat["assured_buy"].mean()
is_hot = Feat["is_hot"].mean()
fitness_certificate = Feat["fitness_certificate"].mean()
reserved = Feat["reserved"].mean()


def CarPricePredictor(car_brand, body_type, fuel_type, transmission, city, car_age, car_rating, warranty_avail,
                      car_availability, total_owners, kms_run, source):
    input_data = pd.DataFrame(0, index=[0], columns=FINAL_MODEL_COLUMNS)

    avg_kms_run = (kms_run[0] + kms_run[1]) / 2
    avg_car_age = (car_age[0] + car_age[1]) / 2
    avg_car_rating = (car_rating[0] + car_rating[1]) / 2

    input_data.loc[0, 'kms_run'] = avg_kms_run
    input_data.loc[0, 'total_owners'] = total_owners
    input_data.loc[0, 'car_age'] = avg_car_age
    input_data.loc[0, 'car_rating'] = avg_car_rating

    input_data.loc[0, 'times_viewed'] = times_viewed
    input_data.loc[0, 'assured_buy'] = assured_buy
    input_data.loc[0, 'is_hot'] = is_hot
    input_data.loc[0, 'fitness_certificate'] = fitness_certificate
    input_data.loc[0, 'reserved'] = reserved
    input_data.loc[0, 'ad_age_days'] = ad_age_days

    input_data.loc[0, f'car_brand_{car_brand}'] = 1
    input_data.loc[0, f'body_type_{body_type}'] = 1
    input_data.loc[0, f'fuel_type_{fuel_type}'] = 1
    input_data.loc[0, f'transmission_{transmission}'] = 1
    input_data.loc[0, f'city_{city}'] = 1
    input_data.loc[0, f'car_availability_{car_availability}'] = 1
    input_data.loc[0, f'source_{source}'] = 1

    input_data.loc[0, 'warranty_avail'] = 1 if warranty_avail.lower() == 'yes' else 0

    prediction = Model.predict(input_data)

    return f"Rs. {prediction[0]:,.2f}"


with gr.Blocks(theme='JohnSmith9982/small_and_pretty', css=custom_css2) as interface:
    gr.Markdown("# Car Price Prediction (ML Model)")
    gr.Markdown("Enter Car Details to Predict Price")

    with gr.Row():
        with gr.Column(variant='panel', scale=1):
            gr.Markdown("## Vehicle Details ", elem_classes=["custom-heading-label"])
            with gr.Group():
                car_brand_input = gr.Dropdown(ALL_BRANDS, label="Car Brand")
            with gr.Group():
                body_type_input = gr.Dropdown(ALL_BODYTYPES, label="Body Type")
            with gr.Group():
                fuel_type_input = gr.Dropdown(ALL_FUEL_TYPES, label="Fuel Type")
            with gr.Group():
                transmission_input = gr.Radio(ALL_TRANSMISSIONS, label="Transmission")
            with gr.Group():
                city_input = gr.Dropdown(ALL_CITIES, label="City")

        with gr.Column(variant='panel', scale=1):
            gr.Markdown("## Usage & Ownership ", elem_classes=["custom-heading-label"])
            with gr.Group():
                car_age_range = RangeSlider(minimum=min(ALL_AGES), maximum=max(ALL_AGES), value=[2, 5], step=1,
                                            label="Car Age")
            with gr.Group():
                car_rating_range = RangeSlider(minimum=min(ALL_RATINGS), maximum=max(ALL_RATINGS), value=[4.0, 4.5],
                                               step=0.1, label="Car Rating-Rating in xyz.com")
            with gr.Group():
                warranty_avail_input = gr.Radio(ALL_WARRANTY_AVAILS, label="Warranty Availability")
            with gr.Group():
                car_availability_input = gr.Radio(ALL_AVAILABILITIES, label="Car Availability")
            with gr.Group():
                total_owners_input = gr.Slider(0, 5, step=1, label="Total Owners")
            with gr.Group():
                kms_run_range = RangeSlider(minimum=0, maximum=650000, value=[5000, 10000], step=1,
                                            label="KMS Run - KMS already run by the car")
            with gr.Group():
                source_input = gr.Radio(ALL_SOURCES, label="Source")

    with gr.Row():
        with gr.Column(variant='panel', scale=1):
            with gr.Group():
                output_text = gr.Textbox(label="Predicted Price (in Rs.)")
            with gr.Group():
                predict_button = gr.Button("Predict", variant="primary")

    gr.Examples(
        examples=[
            ['bmw', 'luxury sedan', 'petrol & cng', 'manual', 'bengaluru', [1, 2], [4.5, 5], 'yes', 'in_transit', 1,
             [15000, 20000], 'inperson_sale'],
            ['honda', 'suv', 'electric', 'automatic', 'pune', [4, 5], [3.5, 4], 'no', 'in_stock', 3, [35000, 50000],
             'online'],
            ['volkswagen', 'hatchback', 'diesel', 'manual', 'mumbai', [2, 4], [2.5, 4.5], 'yes', 'pickup_pending', 5,
             [19500, 62000], 'online']
        ],
        inputs=[car_brand_input, body_type_input, fuel_type_input, transmission_input, city_input, car_age_range,
                car_rating_range, warranty_avail_input, car_availability_input, total_owners_input, kms_run_range,
                source_input]
    )

    predict_button.click(
        fn=CarPricePredictor,
        inputs=[car_brand_input, body_type_input, fuel_type_input, transmission_input, city_input,
                car_age_range, car_rating_range, warranty_avail_input, car_availability_input, total_owners_input,
                kms_run_range, source_input],
        outputs=[output_text]
    )

interface.launch()
