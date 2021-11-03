from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

filename="final_model.pkl"
model=pickle.load(open(filename, 'rb'))

@app.route('/')
def index():
	
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def main():

        MSSubClass       =  int(request.form['MSSubClass'])
        LotFrontage	    =   float(request.form['LotFrontage'])
        LotArea  = int(request.form['LotArea'])
        PoolArea    =  int(request.form['PoolArea'])
        BedroomAbvGr       =  int(request.form['BedroomAbvGr'])
        MiscVal     =  int(request.form['MiscVal'])
        

        input_variables = pd.DataFrame([[MSSubClass, LotFrontage, LotArea, PoolArea, BedroomAbvGr,
                                           MiscVal]],
                                       columns=['MSSubClass', 'LotFrontage', 'LotArea', 'PoolArea', 'BedroomAbvGr',
                                               'MiscVal'])


        prediction = model.predict(input_variables)[0]
        if prediction == 0.0:
            prediction = "Poor Heart Condition"
        elif prediction == 1.0:
            prediction = "Good Heart Condition"
       
        return render_template('result1.html', result=prediction)



if __name__ == '__main__':
    app.run()


