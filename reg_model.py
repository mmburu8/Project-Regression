import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pickle

student_df = pd.read_csv(r"C:\Users\ad\Desktop\MBURU\regression\Student_Performance.csv")
# Label variable transformation
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()
# Fit and transform the column
student_df['Extracurricular Activities'] = label_encoder.fit_transform(student_df['Extracurricular Activities'])

# Split into features and labels
X = student_df.drop("Performance Index", axis=1).values
y = student_df['Performance Index'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 8)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

poly_svr = SVR(kernel='poly', degree=3, C=1).fit(X_train, y_train.ravel())

pickle.dump(poly_svr, open('polynomial.pkl', 'wb'))
pickle.dump(sc, open('scaling.pkl', 'wb'))

