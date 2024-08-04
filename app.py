from joblib import load
import gradio as gr
import numpy as np

# Lists for categorical data
years_list = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
gender_list = ["Female", "Male", "Other"]
smoking_history_list = ['never', 'not current', 'current', 'No Info', 'ever', 'former']
race_list = ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"]
location_list = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
                'Colorado', 'Connecticut', 'Delaware', 'District of Columbia',
                'Florida', 'Georgia', 'Guam', 'Hawaii', 'Idaho', 'Illinois',
                'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
                'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
                'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
                'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
                'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
                'Pennsylvania', 'Puerto Rico', 'Rhode Island', 'South Carolina',
                'South Dakota', 'Tennessee', 'Texas', 'United States', 'Utah',
                'Vermont', 'Virgin Islands', 'Virginia', 'Washington',
                'West Virginia', 'Wisconsin', 'Wyoming']

# Generator function for encoding categorical variables
def encode_category(value, category_list):
    return [1 if value == category else 0 for category in category_list]

def diabetes_prediction(year, gender, age, location, race, hypertension, heart_disease, smoking_history, bmi, hbA1c_level, blood_glucose_level):
    X = []

    X.append(year)
    X.append(age)
    X.extend(encode_category(race, race_list))
    X.append(1 if hypertension == "True" else 0)
    X.append(1 if heart_disease == "True" else 0)
    X.extend([bmi, hbA1c_level, blood_glucose_level])
    X.extend(encode_category(gender, gender_list))
    X.extend(encode_category(location, location_list))
    X.extend(encode_category(smoking_history, smoking_history_list))

    # Load model and make prediction
    model = load("diabetes_prediction_model.joblib")
    prediction_proba = model.predict_proba(np.array(X).reshape(1, -1))[0]
    prediction = model.predict(np.array(X).reshape(1, -1))[0]

    # Extract confidence for both classes
    confidence_diabetic = prediction_proba[1]
    confidence_non_diabetic = prediction_proba[0]

    if prediction == 1:
        message = f"The individual is diabetic with {confidence_diabetic * 100:.1f}% confidence"
    else:
        message = f"The individual is non-diabetic with {confidence_non_diabetic * 100:.1f}% confidence"

    return message

# Gradio interface
demo = gr.Interface(
    title="Diabetes Prediction",
    description="<center>A machine learning model that predicts if an individual is diabetic or non-diabetic (Binary Classification)<center/><br/>",
    fn=diabetes_prediction,
    inputs=[
        gr.Dropdown(years_list, value=2015, label="Year Collected", info="The year the data was collected in"),
        gr.Radio(gender_list, value="Female", label="Gender", info="The gender of the individual"),
        gr.Slider(1, 120, value=18, label="Age", info="The age of the individual in years"),
        gr.Dropdown(location_list, value="Alabama", label="Location", info="The state or region where the individual resides"),
        gr.Dropdown(race_list, value="Asian", label="Race", info="The race of the individual"),
        gr.Radio(["True", "False"], value="True", label="Has Hypertension",  info="If individual has hypertension"),
        gr.Radio(["True", "False"], value="True", label="Has Heart Disease", info="If individual has heart disease"),
        gr.Dropdown(smoking_history_list, value="never", label="Smoking History", info="The individual's smoking history"),
        gr.Slider(value=18.5, minimum=10, maximum=250, step=0.1, label="BMI", info="Body Mass Index of the individual"),
        gr.Slider(value=5.5, minimum=1, maximum=9, step=0.1, label="HbA1c Level", info="The HbA1c level (a measure of blood sugar levels over time)"),
        gr.Slider(value=100, minimum=50, maximum=380, step=1, label="Blood Glucose Level", info="The blood glucose level in mg/dL")
    ],
    submit_btn="Predict",
    outputs=["text"],
)

demo.launch(share=True)
