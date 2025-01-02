from flask import Flask, request, render_template, jsonify 
import numpy as np
import pandas as pd
import pickle
import ast

app = Flask(__name__)

medications = pd.read_csv('datasets/medications.csv')
symptoms_df = pd.read_csv('datasets/symptoms.csv')
diseases_df = pd.read_csv('datasets/diseases.csv')


mlp = pickle.load(open('model/MLP.pkl','rb'))

def helper(dis):
    med = [medications[medications['Disease'] == dis]['Medication']]
    med = [med for med in med[0]]
    return ast.literal_eval(med[0])

symptoms_dict = pd.Series(symptoms_df.Index.values, index=symptoms_df.Symptom).to_dict()
diseases_list = pd.Series(diseases_df.Disease.values, index=diseases_df.Index).to_dict()


# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[mlp.predict([input_vector])[0]]


@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')

        if not symptoms:
            return render_template('index.html', message="Please enter your symptoms.")

        try:
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]

            predicted_disease = get_predicted_value(user_symptoms)
            medications = helper(predicted_disease)

            return render_template('index.html', predicted_disease=predicted_disease, medications=medications)
        except Exception as e:
            # Log the error and provide a user-friendly message
            print(f"Error occurred: {e}")
            return render_template('index.html', message="You might have enterd wrong or mispelled input")

    return render_template('index.html')



# about view funtion and path
@app.route('/about')
def about():
    return render_template("about.html")

# developer view funtion and path
@app.route('/developer')
def developer():
    return render_template("developer.html")

if __name__ == '__main__':

    app.run(debug=True)