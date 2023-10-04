import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print(request.form)
    online_order = request.form.get('Online Order')
    book_table= request.form.get('Book Table')
    votes = request.form.get('Votes')
    location = request.form.get('Location')
    rest_type = request.form.get('Restaurant Type')
    cuisines = request.form.get('Cuisines')
    cost = request.form.get('Cost')
    menu_item = request.form.get('Menu Item')
    #features = [int(x) for x in request.form.values()]
    print([[online_order, book_table, votes, location, rest_type, cuisines, cost, menu_item]])
    #
    final_features = np.array([[online_order, book_table, votes, location, rest_type, cuisines, cost, menu_item]])
    prediction = model.predict(final_features)

    output = round(prediction[0], 1)

    return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)