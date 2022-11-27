from sklearn.naive_bayes import BernoulliNB
import numpy as np

#------------------------Dataset-------------------------------------
x_train = np.array([
    [0, 1, 1],
    [0, 0, 0],
    [0, 0, 0],
    [1, 1, 1]
])
y_train = ['Y' , 'N' , 'Y' 'Y']
x_test = np.array([[0, 1, 0]])

#------------------------Scikit learn-------------------------------------
clf = BernoulliNB(alpha=1 , fit_prior=True)
clf.fit(x_train , y_train)
predict_proba = clf.predict_proba(x_test)
print ('scikit learn predicted probillities: ' , predict_proba)

pred= clf.predict(x_test)
print ('scikit learn predicted: ' , pred)
