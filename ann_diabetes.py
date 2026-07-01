# importing dataset and libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('diabetes.csv')
df.head()
df.isnull().sum()
df.duplicated()


# DATA IS CLEANED AND NOT HAVING ANY NULL AND DUPLICATED VALUES
(df['Glucose']==0).sum() # to check that is the dataset have any null value filled with zero

print(df[['Glucose','BloodPressure','SkinThickness','BMI','Insulin',]].eq(0).sum())

# this dataset have values which are filled with zero
# so we have to fill them with median becuase median is not effected by outliers while mean is

zero_col = ['Glucose','BloodPressure','SkinThickness','BMI','Insulin']

for col in zero_col:
    df[col] = df[col].replace(0, df[col].median())

print(df[['Glucose','BloodPressure','SkinThickness','BMI','Insulin',]].eq(0).sum())
# All the values are filled with median 
df.describe()

# splitting data into input feature and output feature
x = df.drop('Outcome',axis=1)
y = df['Outcome']


# Train Test Split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=24,test_size=0.2)

print(x_train.shape)
print(y_train.shape)

# performing standard scalar
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

# building ANN architecture 

model = Sequential()
model.add(Dense(16,activation='relu',input_shape=(8,)))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

# compile the model 

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

# training the model;

history = model.fit(x_train,y_train,
                    epochs = 100,
                    batch_size = 32,
                    validation_split = 0.2,
                    verbose = 1
                    )



from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Get predictions
y_pred = (model.predict(x_test) > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Diabetes', 'Diabetes'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=150)

# Classification Report
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
