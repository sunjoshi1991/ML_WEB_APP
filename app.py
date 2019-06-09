### import flask and ither required modules
### pip install flask

from flask import Flask,render_template, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

### initiate flask app

app = Flask(__name__)

@app.route('/')

### home html which is to take input (comment) from user
def home():
	return render_template('home.html')


@app.route('/predict', methods = ['POST'])


### fucntion that actually predicts the result based on the comments given
### we use naive_bayes calsssifer to predict the result as this task is a binaty classification , naive_bayes is most commonly used
def predict():
	df = pd.read_csv('YoutubeSpamMergedData.csv')
	data = df[['CONTENT','CLASS']]

	### choose feature and lable to classify
	data_X =data['CONTENT']
	data_y = data.CLASS
	corpus = data_X
	### 
	cv = CountVectorizer()
	X = cv.fit_transform(corpus)
	## train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X,data_y, test_size = 0.3, random_state = 42)
	### build model
	model = MultinomialNB()
	model.fit(X_train, y_train)
	model.score(X_test, y_test)

	###  convert given comment to token count array and use model to predict 
	if request.method =='POST':
		comment = request.form['comment']
		orig_data = [comment]
		vect = cv.transform(orig_data).toarray()
		predictions = model.predict(vect)

		### results.html is the predicted class based on user comment
	return render_template('results.html', prediction = predictions)



if __name__=='__main__':
	app.run(debug=True)