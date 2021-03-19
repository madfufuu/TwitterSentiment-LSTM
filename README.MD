# Twitter Sentiment Analysis with LSTM 
-Importance of the Neutral Classification
CS 6375 Machine Learning Final Project

Preprocessing: <br/>
&nbsp; splitdata.py <br/>
&nbsp; example: python splitdata.py training_size sequence_length dataset.csv

Model Training:<br/>
&nbsp; train.py<br/>
&nbsp; To train the model, use command line to run "train.py" <br/>
&nbsp; The hyperparameters are manually set inside of the program. <br/>
&nbsp; This program outputs a trained model using LSTM network with the 1.6 million <br/>
tweets dataset to predict the sentiment of tweets.

Model Evaluation: <br/>
&nbsp; evaluate.py <br/>
&nbsp; example: python evaluate.py testSet.csv x_test.csv model.h5 <br/>
&nbsp; Use the program "evaluate.py" to evaluate the performance of the model.

Running instructions: <br/>
* First, run splitdata.py, specify the training size, sequence length and the path to 
the full dataset.
This program will output a couple of files required to train and do model eval:
    testSet.csv - the test set for our model
    trainSet.csv - the train set for our model
    x_test.csv - the parameters for our test set

* Second, run train.py with the training dataset. It will train a model using
hyperparameters specified inside of the train.py file.
This program outputs a fully trained model ready for evaluation.
Third, run the evaluate.py with the testset, parameters for the test set and also
the trained model to evaluate the model being tested.
This program will output the final evaluation metrics for the model as a
confusion matrix.
To efficiently evaluate our models, you can download the pre-trained models
from the google drive link provided below:
    https://drive.google.com/drive/folders/16SFLQ0mKEd_nCTdNlr4Z144le5EfW61w?usp=sharing
    
Link to dataset (sentiment140): https://www.kaggle.com/kazanova/sentiment140
        

    
