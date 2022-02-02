import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)
model = pickle.load(open('cardio.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index7.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active','bmi']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    #if output == 1:
     #   res_val = "** breast cancer **"
    #else:
     #   res_val = "no breast cancer"
        

    return render_template('result.html', prediction_text=output)
#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    app.run()
    #app.run(host='0.0.0.0', port=port)
