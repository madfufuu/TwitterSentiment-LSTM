'''evaluate.py

CS 6375.003 Final Project: evaluate.py
Authors: Usuma Thet, Merissa Fulfer, Rishi Dandu, Yuncheng Gao

    python evaluate.py testSet.csv x_test.csv model.h5

    testSet.csv is the testing dataset
    x_test.csv contains all of the parameters for a single sentence
    model.h5 is the model for which will be evaluated for.

'''

from keras.models import load_model
import sys
import time
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
import pandas as pd
from numpy import loadtxt

#nltk.download('stopwords')

POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
threshold = (0.3, 0.7)
tokenizer = Tokenizer()

def main(argv):
    
    # Get filenames from arguments
    if len(argv) != 3:
        raise Exception("Incorrect number of arguments") 
    test_set_name = argv[0]
    x_test_name = argv[1]
    model_name = argv[2]
    SEQUENCE_LENGTH = int(model_name[:3])
    print(SEQUENCE_LENGTH)
    x_test = loadtxt(x_test_name, delimiter=',')
    test_set = pd.read_csv(test_set_name)
    model = load_model(model_name)
    print(model)
    
    print("Check1")


    def classification(score, neutral=True):
        if neutral:        
            label = NEUTRAL
            if score <= threshold[0]:
                label = NEGATIVE
            elif score >= threshold[1]:
                label = POSITIVE

            return label
        else:
            return NEGATIVE if score < 0.5 else POSITIVE


    def predict(text, neutral=True):
        start = time.time()
        x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
        score = model.predict([x_test])[0]
        label = classification(score, neutral=neutral)

        return {"label": label, "score": float(score),
        "elapsed_time": time.time()-start}
    

    print(predict("Miss me with that weeb shit terrible bad bad."))

    print(predict("this is awesome good! great haha nice"))
    
    predictionY = []
    testY = list(test_set.target)
    scores = model.predict(x_test, verbose=1, batch_size=8000)

    # Doing predictions without considering neutrals
    predictionY = [classification(score, neutral=False) for score in scores]

    #print("Done")

    print(classification_report(testY, predictionY))
    print(accuracy_score(testY, predictionY))
    print("Evaluation done!")
  
if __name__ == "__main__":
    main(sys.argv[1:])