import pandas as pd
from scipy.io import arff
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

data,meta = arff.loadarff(r'C:\Users\musta\OneDrive\Masaüstü\Yeni Klasör\Autism_Data.arff')

df = pd.DataFrame(data)

print(df.head())
df = df[['gender', 'A1_Score', 'A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score','A9_Score','A10_Score','jundice','autism','used_app_before','Class/ASD']]
def clean_data(value):
    
    if value not in [1, 0, 'yes', 'no', 'self', 'parent', 'f', 'm']:
        return None
    return value


df = df.applymap(clean_data)

df = df.dropna()
df = df.replace({'yes': 1, 'no': 0, 'f': 1, 'm': 0})



X = df.drop(columns=['gender', 'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'jundice', 'autism', 'used_app_before', 'Class/ASD']) #featurelar
y = df['autism']  # target


class BernoulliNB_custom:
    def __init__(self):
        self.class_probs = None  # Prior probabilities
        self.feature_probs = None  # Conditional probabilities

    def fit(self, X, y):
        # Kategorik sınıfları ve her bir sınıf için öncelikli olasılıkları hesapla
        self.class_probs = y.value_counts() / len(y)
        
        # Özelliklerin her sınıf için olasılıklarını hesapla
        self.feature_probs = {}
        for feature in X.columns:
            self.feature_probs[feature] = {}
            for class_value in self.class_probs.index:
                # Her sınıf için 1'lerin oranını hesapla (Bernoulli modeli)
                class_data = X[y == class_value]
                feature_prob = (class_data[feature].sum() + 1) / (len(class_data) + 2)  # Laplace smoothing
                self.feature_probs[feature][class_value] = feature_prob

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            posteriors = {}
            for class_value in self.class_probs.index:
                
                posterior = np.log(self.class_probs[class_value])
                for feature, value in row.items():
                    if value == 1:
                        
                        posterior += np.log(self.feature_probs[feature].get(class_value, 0))
                    else:
                       
                        posterior += np.log(1 - self.feature_probs[feature].get(class_value, 1))
                posteriors[class_value] = posterior
            
            
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)


bnb_model_custom = BernoulliNB_custom()
start_time = time.time()
bnb_model_custom.fit(X_train, y_train)
fit_time = time.time() - start_time


start_time = time.time()
y_pred = bnb_model_custom.predict(X_test)
predict_time = time.time() - start_time


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Python ile eğitim süresi: {fit_time} saniye")
print(f"Python ile tahmin süresi: {predict_time} saniye")
print(f"Doğruluk: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")