from flask import Flask, render_template, request
from crop_recommendation.app import crop_recommendation_app 
from yield_prediction.app import yield_prediction_app  

app = Flask(__name__)

app.register_blueprint(crop_recommendation_app, url_prefix='/crop')
app.register_blueprint(yield_prediction_app, url_prefix='/yield')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        return 'Thank you for your message!'
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
