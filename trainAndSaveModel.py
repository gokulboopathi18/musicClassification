import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from matplotlib import pyplot as plt

from keras import models
from keras import layers

DATA_FOLDER = 'data'
FEATURES_FILE = os.path.join(DATA_FOLDER, 'features.csv')
MODEL_JSON_FILE = 'model.json'
MODEL_WEIGHTS_FILE = 'weights.h5'

# read all the data from the features file
df = pd.read_csv(FEATURES_FILE)
print('+ DATASET')
print(df.sample(5, random_state=10))
print()

# normalize everything
scaler = StandardScaler()
df.iloc[:, 1:-1] = scaler.fit_transform(df.iloc[:, 1:-1])
print('+ NORMALIZED')
print(df.sample(5, random_state=10))
print()

# enocode genre labels
encoder = LabelEncoder()
df.iloc[:, -1:] = encoder.fit_transform(df.iloc[:, -1:])
classes = df['label'].unique()
print(df.sample(5, random_state=10))
print()

# split into training and testing set
trainDf, testDf = train_test_split(df, test_size=0.2)

X = trainDf.iloc[:,1:-1]
y = trainDf.iloc[:, -1:]

testX = testDf.iloc[:,1:-1]
testy = testDf.iloc[:, -1:]

print('+ training dataset: ', X.shape, y.shape)
print('+ testing dataframe:', testX.shape, testy.shape)
print()

# build model
model = models.Sequential()
print(X.shape)
model.add(layers.Dense(512,activation='relu',input_shape=(X.shape[1],)))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train model
training_history = model.fit(X, y, epochs = 200, batch_size=512).history
print(training_history.keys())
plt.plot(training_history['accuracy'])
plt.plot(training_history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.savefig('training.png')
print()

# evaluation
print('+ EVALUATION')
test_loss, test_acc = model.evaluate(testX, testy)
print('(*) test_acc: ',test_acc)
print()

preds = model.predict(testX)
predy = np.argmax(preds, axis=1)

print('(*)precision scores :')
ps =  metrics.precision_score(testy, predy, average=None)
for i, label in enumerate(classes):
    print(label,'\t',ps[i])
print()

roc = metrics.roc_auc_score(testy, preds, multi_class='ovr')
print('(*) roc score :', roc)
print()

# save model
with open(MODEL_JSON_FILE, 'w') as f:
    f.write(model.to_json())
model.save_weights(MODEL_WEIGHTS_FILE)
print('+ MODEL SAVED')
