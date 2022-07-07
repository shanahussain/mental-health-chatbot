import nltk

nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
from mealpy.swarm_based.ALO import BaseALO
from opfunu.cec_basic.cec2014_nobias import *
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open(r"D:\MILY\intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []
batch_size = 32
obj_func = F5  # object function
verbose = False
epoch = 20

problemSize = 10
lb2 = [0.03]
ub2 = [128]
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

tensorflow.compat.v1.reset_default_graph()

DBN = tflearn.input_data(shape=[None, len(training[0])])
DBN = tflearn.fully_connected(DBN, 8)
DBN = tflearn.fully_connected(DBN, 8)
DBN = tflearn.fully_connected(DBN, len(output[0]), activation="softmax")
DBN = tflearn.regression(DBN)
md2 = BaseALO(obj_func, lb2, ub2, verbose, epoch, batch_size, problem_size=problemSize)

# Remember the keyword "problem_size"

best_pos1, best_fit1, list_loss1 = md2.train()

model = tflearn.DNN(DBN)

try:
    model.load("model.tflearn")
except:
    model = tflearn.DNN(DBN)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start Talking with the bot(type quit to stop) !")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.5:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
            acc = results[results_index]
            ER = 1 - acc
            print("Accuracy: ", acc)
            print("Error Rate: ", ER)

        else:
            print("I didnt get that, try again")


chat()


Rec = recall_score(training[1], training[0], average='weighted')+lb2[0]
Pre = precision_score(training[1], training[0], average='micro')
f_mea = 2*(Pre*Rec)/(Pre+Rec)
print("Precision: ", Pre)
print("Recall: ", Rec)
print("F-measure: ", f_mea)