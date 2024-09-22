import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Dropout, Input
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read()) # dosyayı oku arkasından degiskene yukle

words = [] # patterndeki tum kelimeleri bir dosyada tutacagız
classes = [] # sohbetin tag'ını konusunu atayacagız
documents = [] # hem patternin tokenlarını hem de tag'ı birlikte tutuyor
ignore_letters = ['?', '!', ',', '.', "'"]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) # kullanıcının cümlelerini kelimeye(token) çevir
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes: # tum tagları bir listede tutuyoruz(aynı tagda birden fazla pattern olabilir)
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words)) # set: kelimeleri tekilleştirme, sorted: liste olarak çıktı almak için
print(words)

classes = sorted(set(classes)) # sıralı olması onemli, ilgili tag'a 1 yazacagız sonrasında
print(classes)

pickle.dump(words, open("words.pkl","wb"))  # binary olarak bu verileri pkl formatında bir dosyaya yazdık
pickle.dump(classes, open("classes.pkl", "wb"))


X_training = []
y_training = []
output_empty = [0] * len(classes) # bagofwords uzunlugu kadar liste/array(matris) !!

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0) # bagofwords uzunlugu kadar liste/array(matris) 0 ve 1ler

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    #training.append([bag, output_row]) # bag içinde olanlar 1 olmayanlar 0, output_row'da tag'i barındırıyorusa 1 verir
                                       # classes = sorted(set(classes)) sıralı oldugu icin ilgili tag'a gore yukarıda 1 geliyor
    X_training.append(bag) # !!!!
    y_training.append(output_row) # !!!


train_x = np.array(X_training)
train_y = np.array(y_training)

# ANN Model Kurulumu

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

sgd = SGD(learning_rate = 0.01, momentum=0.9, nesterov = True)
model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics = ["accuracy"])

model.fit(np.array(train_x), np.array(train_y), epochs = 300, batch_size = 5, verbose = 2)
model.save("chatbot_model.keras")
print("Done")









