import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from matplotlib import pyplot as plt

from keras import models

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
classes = df['label'].unique()
print('+ NORMALIZED')
print(df.sample(5, random_state=10))
print()

# enocode genre labels
encoder = LabelEncoder()
df.iloc[:, -1:] = encoder.fit_transform(df.iloc[:, -1:])
print(df.sample(5, random_state=10))
print()


df = df.sample(500)

X = df.iloc[:,1:-1]
y = df.iloc[:, -1:]

# load model
json = open(MODEL_JSON_FILE, 'r')
model = models.model_from_json(json.read())
model.load_weights(MODEL_WEIGHTS_FILE)
json.close()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# evaluation
print('+ EVALUATION')
test_loss, test_acc = model.evaluate(X, y)
print('(*)test_acc: ',test_acc)
print()

preds = model.predict(X)
predy = np.argmax(preds, axis=1)

print('(*)precision scores :')
ps =  metrics.precision_score(y, predy, average=None)
for i, label in enumerate(classes):
    print(label,'\t',ps[i])
print()

roc = metrics.roc_auc_score(y, preds, multi_class='ovr')
print('(*) roc score :', roc)
print()
