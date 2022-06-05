import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
std = pickle.load(open('std1.pkl', 'rb'))
model = pickle.load(open('classifier1.pkl', 'rb'))
model1=pickle.load(open('decition1.pkl', 'rb'))
model2=pickle.load(open('logistic1.pkl', 'rb'))
model3=pickle.load(open('randomforest1.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('ditect.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    pred = model.predict( std.transform(final_features) )
    pred1 = model1.predict( std.transform(final_features) )
    pred2 = model2.predict( std.transform(final_features) )
    pred3= model3.predict( std.transform(final_features) )
    return render_template('result.html', prediction = pred,prediction1 = pred1,prediction2 = pred2,prediction3 = pred3)

if __name__ == "__main__":
    app.run(debug=True)