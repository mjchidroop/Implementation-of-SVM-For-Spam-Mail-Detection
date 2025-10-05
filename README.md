# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Feature Extraction: Text data is transformed using Bag of Words (CountVectorizer), converting emails into a matrix of token counts.

2. Training: SVM fits a hyperplane using the linear kernel to separate spam and ham messages based on transformed features.

3. Prediction: Each email is classified by testing on the fitted SVM model.

4. Evaluation: Confusion matrix and classification report quantify the model’s predictive accuracy (98% on test data) with visualization.

## Program:
```py
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: CHIDROOP M J
RegisterNumber:  25018548
*/

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load spam dataset
data = pd.read_csv('spam.csv', encoding='ISO-8859-1')
data = data.rename(columns={'v1': 'label', 'v2': 'text'})

# Encode labels (ham: 0, spam: 1)
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

# Feature extraction (Bag of Words)
cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(data['text'])
y = data['label_num']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train, y_train)

# Predictions
y_pred = svc.predict(X_test)

# Confusion Matrix plot
conf = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(conf, cmap='Blues', interpolation='none')
plt.title('Confusion Matrix (SVM Spam Detection)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.yticks([0, 1], ['Ham', 'Spam'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, conf[i, j], ha='center', va='center', color='red')
plt.colorbar()
plt.tight_layout()
plt.show()

# Print evaluation metrics
print(classification_report(y_test, y_pred))
print('Model Test Accuracy:', svc.score(X_test, y_test))

```

## Output:
<img width="704" height="721" alt="Screenshot 2025-10-05 162329" src="https://github.com/user-attachments/assets/1897e70e-024c-454d-b8af-f2535292b75e" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
