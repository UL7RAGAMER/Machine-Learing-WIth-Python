import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.svm import SVC
import joblib as jl
import gradio as gr
from sklearn.metrics import accuracy_score


# ----------------------------------------------------
# 1. MODEL TRAINING & SAVING
# ----------------------------------------------------

# df = sb.load_dataset('titanic')
# median_age = df['age'].median()
# df['age'] = df['age'].fillna(median_age)
# cleandf = df
# cleandf['embarked'].fillna('S', inplace=True)
# cleandf['embark_town'].fillna('Southampton', inplace=True)
# Feat = cleandf[['class', 'sex', 'embarked', 'age', 'sibsp', 'parch', 'fare']].to_numpy()
# Targ = cleandf[['alive']].to_numpy()
# cleandf['FamilySize'] = cleandf['sibsp'] + cleandf['parch'] + 1
#
# # Create a figure with 3 subplots
# fig, axes = plt.subplots(1, 1, figsize=(18, 5))
#
# # Plot 1: Survival by Sex
# sb.countplot(ax=axes, data=cleandf, x='age', hue='alive', palette='Blues')
# axes.set_title('Survival by Sex')
# pd.set_option('display.max_columns',None)
# print(cleandf)
#
#
# #Labelise The Strings In Dataset
# labelencoder = LabelEncoder()
# cleandf['deck'] = labelencoder.fit_transform(df['deck'])
# cleandf['sex'] = labelencoder.fit_transform(df['sex'])
# cleandf['alive'] = labelencoder.fit_transform(df['alive'])
# cleandf['adult_male'] = labelencoder.fit_transform(df['adult_male'])
# cleandf['FamilySize'] = labelencoder.fit_transform(df['FamilySize'])
# cleandf['pclass'] = labelencoder.fit_transform(df['pclass'])
# X = cleandf[['pclass','sex','FamilySize','adult_male','fare','deck']]
#
# Y = cleandf['alive']
# Yscaled = labelencoder.fit_transform(Y)
#
#
# #Parameters initialization
# param_grid ={
#
#         'n_neighbors': list(range(100,300,100)),
#         'weights': ['uniform', 'distance'],
#         'metric': ['euclidean', 'manhattan']
#     }
#
#
# #Splitting Data
# Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Yscaled,test_size=0.2, random_state=690)
#
# #Standardard Scaler
# scaler = StandardScaler()
# Xtrainscaled=scaler.fit_transform(Xtrain)
# Xtestscaled=scaler.fit_transform(Xtest)
#
# #Initialize KNN
# model = KNeighborsClassifier()
#
#
# #GridsearchCV initialization
# grid = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     cv = 5,
#     scoring='accuracy',
#     verbose=3,
#     n_jobs = -1
# )
#
# #Fit and predict
# grid.fit(Xtrainscaled,Ytrain)
# prediction = grid.predict(Xtestscaled)
#
# #Printing Ytest and predict if people arev alive or not after the tragedy
# #print(Ytest)
# print(f"Predictions: {prediction}")
# print(classification_report(Ytest, prediction))
# var = pd.DataFrame(grid.cv_results_)
# print(var.sort_values("rank_test_score").head())
# print(grid.best_params_)
#
# jl.dump(grid.best_estimator_, 'model n2/titanic_knn_model_n2.pkl')
# jl.dump(scaler, 'model n2/titanic_scaler_n2.pkl')


# ----------------------------------------------------
# 2. GRADIO APP
# ----------------------------------------------------
# Scale the input data using the loaded scaler object

# Load the pre-trained model and scaler
try:
    model = jl.load('model n2/titanic_knn_model_n2.pkl')
    scaler= jl.load('model n2/titanic_scaler_n2.pkl')
except FileNotFoundError:
    print("Files not found. Please run the training section again")
    exit()

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

# Define the CSS to add the pound symbol after the number
custom_css2 = """
#fare-input .value-display::after {
  content: "£";
  margin-left: 5px; /* Adds a small space between the number and the symbol */
}
"""


#creating Gradio web app
def titanic_survival_predictor(pclass, deck, fare,  embark_town, name, age, sex, family_size):
    # Preprocess the user's inputs
    sex_encoded = 1 if sex== 'male' else 0
    deck_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7, 'Unknown': 8}
    deck_encoded = deck_map[deck]
    adult_male = 1 if sex== 'male' and age>=18 else 0
    pclass_map = {'First': 0, 'Second': 1, 'Third': 2}
    pclass_num = pclass_map[pclass]
    embark_town_map = {'Southampton' : 'S', 'Cherbourg': 'C', 'Queenstown': 'Q', 'Unknown' : 'U'}
    embark_town_char = embark_town_map[embark_town]
    survive_img = "IMG_20250623_215035_710.jpg"
    perish_img = "IMG_20250623_215042_465.jpg"
    # Create a DataFrame with the correct features and column names
    input_data = pd.DataFrame([[pclass_num, sex_encoded, family_size, adult_male,  fare, deck_encoded]],
                              columns = ['pclass', 'sex', 'FamilySize', 'adult_male', 'fare', 'deck'])

    scaled_input = scaler.transform(input_data)

    # Make the prediction using the loaded model object
    predictions = model.predict(scaled_input)
    img=survive_img if predictions[0] == 1 else perish_img
    return "Good Fortunes to you, You Survived" if predictions[0] == 1 else "Your Fate has met an Abrupt End, You Perished", img

with gr.Blocks(theme='JohnSmith9982/small_and_pretty', css=custom_css2) as interface:
    gr.Markdown("# Titanic Survival Prediction (ML Model)")
    gr.Markdown("Enter Passenger Details to Predict Survival")

    #Define all input components with labels
    with gr.Row():
        with gr.Column(variant="panel", scale=1):
            gr.Markdown("## Passenger Details", elem_classes=["custom-heading-label"])
            with gr.Group():
                name_input = gr.Textbox(label='Name',info="Enter Full Name")
            with gr.Group():
                age_input = gr.Slider(1, 100, step=1, label='Age')
            with gr.Group():
                sex_input = gr.Radio(['male', 'female'], label='Sex')
            with gr.Group():
                family_size_input = gr.Slider(1, 10, step=1, label='Family Size',info="Family Size Including You and Family you are Travelling With")

        with gr.Column(variant="panel", scale=1):
            gr.Markdown("## Seat Details", elem_classes=["custom-heading-label"])
            with gr.Group():
                pclass_input = gr.Radio(['First', 'Second', 'Third'], label='Passenger Class')
            with gr.Group():
                deck_input = gr.Dropdown(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'Unknown'], label="Deck", value="Unknown")
            with gr.Group():
                fare_input = gr.Slider(1, 512.33, step=0.01, label='Fare', elem_id="fare-input")
            with gr.Group():
                embark_town_input = gr.Radio(['Southampton', 'Cherbourg', 'Queenstown', 'Unknown'], label="Port of Embarkation", value="Southampton")

    # Define the output component
    with gr.Row():
        with gr.Column(variant="panel", scale=1):
            with gr.Group():
                output_text = gr.Textbox(label="Prediction Result")
            with gr.Group():
                predict_button = gr.Button("Predict", variant='primary')
        with gr.Column(variant="panel", scale=0):
            with gr.Group():
                output_image = gr.Image(label="Your Fate", width=300, height=300)

    # Create button to trigger prediction
    predict_button.click(
        fn=titanic_survival_predictor,
        inputs=[pclass_input, deck_input, fare_input, embark_town_input, name_input, age_input, sex_input,
                family_size_input],
        outputs=[output_text, output_image]
    )
    # Setting up Examples for inputs
    gr.Examples(
        examples=[
            ["First", "B", 263.0, "Cherbourg", "Ben D Knee", 45, "male", 1],
            ["Third", "D", 15.0, "Southampton", "Jon Snow", 22, "male", 2],
            ["Third", "Unknown", 8.0, "Queenstown", "Deez Nuts", 5, "female", 3]
        ],
        inputs=[pclass_input,deck_input,fare_input,embark_town_input,name_input,age_input,sex_input,family_size_input]
    )

interface.launch()