# FIN 410 HW #4 Random Forrest and Support Vector Machine
# Samuel M. Reisgys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from six import StringIO
import pydotplus
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn import svm


#1 Create a new data X with variables ['Age', 'Sex', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', and 'Ca'] 
# and Create a new data y with variable - ['AHD'].
print("#1 Creating new data X...\n")

url = "https://media.githubusercontent.com/media/Ajim63/FIN-410-Python-Codes-and-Data/main/Python%20Codes/Data/Heart.csv"
df = pd.read_csv(url, index_col="Unnamed: 0")
df.dropna(inplace=True)

x = df[['Age', 'Sex', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca']]
y = df["AHD"]

#2 Divide the both X and y data into 60% training and 40% testing.
print("#2 Dividing x and y into training/test set...\n")
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.40, random_state=42)

#3 Fit a Decision Tree Classifier on training data and print classification report on train and 
#test data that will will show, precision, recall, f1score, support and accuracy,
print("#3 Fitting Decision Tree and calculating report statistics...\n")

clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

report = classification_report(y_test,y_pred)
print(report)

#4 Print the classification tree
print("#4 Printing the classification tree...\n--> Type 'Image(graph.create_png())' in the console to see the classification tree\n")
predictors = ['Age', 'Sex', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca']
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = predictors,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

#5 Fit a support vector machine on training data with linear kernel and print classification report 
#on train and test data that will will show, precision, recall, f1score, support and accuracy,
print("#5 Fitting support vector machine and print classficiation report...\n")

clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(classification_report(y_test,y_pred))


