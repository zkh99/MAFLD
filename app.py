from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load model and scaler
clf = pickle.load(open(r'C:\zaza\documents\University\my subjects\arshad\And beyond\Thesis file\codes\fattyliver_model.pkl', 'rb'))
scaler = pickle.load(open(r'C:\zaza\documents\University\my subjects\arshad\And beyond\Thesis file\codes\fattyliver_scaler.pkl', 'rb'))


app = Flask(__name__)


# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    PLT = float(request.form['PLT'])
    FHCAD = float(request.form['FHCAD'])
    Age = float(request.form['Age'])
    Waist = float(request.form['Waist'])
    DBP = float(request.form['DBP'])
    CRP = float(request.form['CRP'])
    VitD = float(request.form['VitD'])
    Chl = float(request.form['Chl'])
    HDL = float(request.form['HDL'])
    LDL = float(request.form['LDL'])
    TG = float(request.form['TG'])
    ALT = float(request.form['ALT'])
    ALKP = float(request.form['ALKP'])
    CAD = float(request.form['CAD'])
    PCI = float(request.form['PCI'])
    CVA = float(request.form['CVA'])
    HOMA = float(request.form['HOMA'])
    BMI = float(request.form['BMI'])
    AIP = float(request.form['AIP'])

    # Create DataFrame from inputs
    input_data = pd.DataFrame(
        [[PLT, FHCAD, Age, Waist, DBP, CRP, VitD, Chl, HDL, LDL, TG, ALT, ALKP, CAD, PCI, CVA, HOMA, BMI, AIP]],
        columns=['PLT', 'FHCAD', 'Age', 'Waist', 'DBP', 'CRP', 'VitD', 'Chl', 'HDL', 'LDL', 'TG', 'ALT', 'ALKP', 'CAD',
                 'PCI', 'CVA', 'HOMA', 'BMI', 'AIP'])

    # Scale input data
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = clf.predict(scaled_input)

    # Render prediction result template
    return render_template('Result.html', prediction=prediction[0])

@app.route('/Prediction')
def Prediction():
    return render_template('Prediction.html')

@app.route('/MAFLD')
def MAFLD():
    return render_template('MAFLD.html')

@app.route('/Contact')
def Contact():
    return render_template('Contact.html')

if __name__ == '__main__':
    app.run(debug=True)