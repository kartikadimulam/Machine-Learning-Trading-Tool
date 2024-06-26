import numpy as np
import matplotlib.pyplot as plt

#print(plt.style.available)
plt.style.use('seaborn-v0_8-dark-palette')

from sklearn import svm

np.random.seed(0)
X = np.r_[np.random.randn(20,2)-[2,2], np.random.randn(20,2)+[2,2]]
Y = [0]*20 + [1]*20

clf = svm.SVC(kernel='linear')
clf.fit(X,Y)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5,5)
yy = a*xx - (clf.intercept_[0]) / w[1]

b = clf.support_vectors_[0]
yy_down = a*xx + (b[1] - a * b[0])

b = clf.support_vectors_[-1]
yy_up = a*xx + (b[1] - a * b[0])

plt.figure(figsize=(10,7))

plt.plot(xx,yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=80,
            facecolors = 'none')
plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.Paired)
plt.title('Support Vector Classifier', fontsize=14)
plt.xlabel('X-coordinate', fontsize=12)
plt.ylabel('Y-coordinate', fontsize=12)
plt.show()
