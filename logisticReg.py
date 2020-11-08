import numpy
import math
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# #from sklearn import preprocessing


# Build a linear_model.LogisticRegression model of the breast_cancer data
# set in sklearn. Report the estimated generalization error rate of the model,
# using 4-fold cross-validation. Does the result suggest good modeling? Justify
# your answer by comparing against the accuracy of a random (coin toss) predictor.
# (b) Examine the model trained on the full data set. What input attribute has the
# greatest impact on the log odds in the sense that a unit increase in that attribute
# leads to the greatest change in the log odds? What input attribute has the second
# greatest effect? What about the third greatest? Explain in detail, justifying your
# claims in terms of model specifics. Refer to the attributes explicitly by name,
# relying on the documentation provided with the data set. Read it carefully. Don’t
# rely on the example in the documentation, though, as it appears to contain errors.
# (c) The preceding part considers the magnitude of the effect of each attribute on the
# log odds, without accounting for the fact that different attributes are on different
# scales. Therefore, the comparison isn’t really “apples to apples”. Let’s try an
# alternate approach. Generate a second version of the data set in which each
# attribute has been mean-centered and scaled by its standard deviation. Train
# a LogisticRegression model on the modified data set. Report the coefficient
# values of the trained model, and identify (by name) the top three attributes in
# terms of their effect on the log odds, as in the preceding part. Explain in detail.
# StandardScaler can be used to do the centering and scaling in sklearn.

D = numpy.loadtxt('regressionDataPS3.txt', delimiter = ',')


data = datasets.load_breast_cancer()

XBreast = data.data
yBreast = data.target
breast = []


data2 = datasets.load_breast_cancer()

XBreast2 = data2.data
yBreast2 = data2.target

var = []
varStore = []
biasA = []
Eout = []
Aavg = []
Bavg = []
x =-1


## QUESTION:  1 code
def sampleComplexity(dvc, n):
    epsilon = .1
    delta = .1
    newN = (n*2)**dvc

    result = (8/(delta**2))*numpy.log(((4*newN)+4)/delta)

    return round(result)
#end Q 1


#code for # QUESTION:  3
prediction = []
CrossVal = []

y = D[:,-1]
X = D[:,:-1]

def prob3():
    for n in range(10, 510, 10):

        rowsx = X[0:n,:]
        rowsy = y[0:n]

        model = LinearRegression()
        model.fit(rowsx,rowsy)
        temp = model.score(rowsx, rowsy)

        prediction.append(temp)

        CrossVal.append(numpy.mean(cross_val_score(model, rowsx, rowsy, cv=10)))

    plt.title("Breast Cancer precition")
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration number")
    plt.plot(CrossVal)
    plt.plot(prediction)
    plt.show()
#end Q 3


#code for # QUESTION:  4
def prob4():
    X_train, X_test, y_train, y_test = train_test_split(XBreast, yBreast)
    logisticRegr = LogisticRegression(max_iter = 5000)
    logisticRegr.fit(X_train, y_train)

    breast.append(numpy.mean(cross_val_score(logisticRegr, XBreast, yBreast, cv=4)))


    #train test split for normalization
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(XBreast2, yBreast2)

    #scale data
    sc = StandardScaler(with_mean = True).fit(X_train_2)

    #transform data into more meaningful format
    X_train_std = sc.transform(X_train_2)
    # X_test_std = sc.transform(X_test_2)

    #logistic regression on scaled data
    logisticRegr2 = LogisticRegression(max_iter = 5000)
    logisticRegr2.fit(X_train_std, y_train_2)

    print("Value without normalization:", breast,"\n")

    print("Feature names and coefficients prior to scaling:\n")
    print(data.feature_names, "\n")
    print(logisticRegr.coef_, "\n")

    print("Feature names and coefficients after scaling:")
    print(data2.feature_names, "\n")
    print(logisticRegr2.coef_, "\n")


print("Values for QUESTION: 1")
print("For dvc = 1:")
print(sampleComplexity(1,10935))


print("For dvc = 3:")
print(sampleComplexity(3,30000))


print("For dvc = 10:")
print(sampleComplexity(10,100356))


print("For dvc = 30:")
print(sampleComplexity(30,323105))


prob4()
