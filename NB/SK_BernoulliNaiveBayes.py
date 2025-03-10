import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
import time
from sklearn.naive_bayes import BernoulliNB
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bnb_model = BernoulliNB()
start_time = time.time()
bnb_model.fit(X_train, y_train)
fit_time = time.time() - start_time

start_time = time.time()
y_pred = bnb_model.predict(X_test)
predict_time = time.time() - start_time

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Scikit-learn ile eğitim süresi: {fit_time} saniye")
print(f"Scikit-learn ile tahmin süresi: {predict_time} saniye")
print(f"doğruluk: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")