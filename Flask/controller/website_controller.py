from app import app 
from flask import render_template, request 


#base url
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/homepage')
def homepage():
    return render_template('homepage.html')



@app.route('/result')
def result():
    return render_template('/result.html')





if __name__ == '__main__':
    app.run(debug=True)