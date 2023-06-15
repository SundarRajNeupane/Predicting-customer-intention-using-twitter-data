from flask import Flask, render_template, request
app= Flask(__name__)
# import Mainn
import pickle

model=pickle.load(open('model.pkl','rb'))

    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

@app.route('/tweets', methods = ['POST'])
def homepage1():
    input_list=request.form['fname']
    print(input_list)
    # final = Mainn.tfidf_vectorizer.transform(input_list)
    #final=TfidfVectorizer.transform(input_list)
    # print(final)
    # prediction = model.predict_proba(final)
    # pred_of_SVM = Mainn.SVM.predict(final)
    # pred_of_RFC = Mainn.RFC.predict(final)
    # if(pred_of_RFC == 1):
    #     pred_of_RFC_text = "Positive"
    # else:
    #     pred_of_RFC_text = "Negative"
        
    # if(pred_of_SVM == 1):
    #     pred_of_SVM_text = "Positive"
    # else:
    #     pred_of_SVM_text = "Negative"
    # print(prediction)
    # output = '{0:.{1}f}'.format(prediction[0][1],2)
    
    
    return render_template('homepage.html',tweet=input_list)
                        #    pred_of_RFC='The prediction through RFC : {}'.format(pred_of_RFC_text),
                        #    pred_of_SVM='The prediction through SVM : {}'.format(pred_of_SVM_text),
                        #    pred= 'the probability is {}'.format(output),)

@app.route('/result')
def result():
    return render_template('/result.html')


if __name__ == '__main__':
    app.run(debug=True)


