import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.stats import ttest_ind
from copy import deepcopy
import statsmodels.api as sm
from scipy.stats import shapiro

repeat = 100
obs = 50

distance = 2
mu_pos = [distance/np.sqrt(2),distance/np.sqrt(2)]
mu_neg = [-distance/np.sqrt(2),-distance/np.sqrt(2)]
var_1 = 1
cov_1 = 0
var_2 = 6
sigma = [[var_1,cov_1],[cov_1,var_2]]
#sigma_2 = sigma_1 #[[var_2,cov_2],[cov_2,var_2]]

lda_scores = np.empty(repeat)
lr_scores = np.empty(repeat)
for i in range(0, repeat):
    features = 2
    classes = 2

    x_pos = np.random.multivariate_normal(mu_pos, sigma, int(obs / 2))
    x_neg = np.random.multivariate_normal(mu_neg, sigma, int(obs / 2))
    X = np.vstack((x_pos, x_neg))
    y = np.hstack((np.ones(int(obs / 2)), np.zeros(int(obs / 2))))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.8)

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

    lr = linear_model.LogisticRegression()
    grid_search = GridSearchCV(lr, param_grid, cv=5)

    grid_search.fit(X_train, y_train)
    lr_scores[i] = grid_search.score(X_test, y_test)
    #get the actual model with grid_search.best_estimator_

    phi = 0.5
    mu_pos_hat = np.average(x_pos,0)
    mu_neg_hat = np.average(x_neg,0)
    sigma_hat = (1/obs)*((x_pos - mu_pos_hat).T.dot((x_pos - mu_pos_hat)) + (x_neg - mu_neg_hat).T.dot(x_neg - mu_neg_hat))
    sigma_hat_inv = np.linalg.inv(sigma_hat)

    theta = sigma_hat_inv.dot(mu_pos_hat - mu_neg_hat)
    theta_0 = 0.5*(mu_neg_hat.T.dot(sigma_hat_inv).dot(mu_neg_hat) - mu_pos_hat.T.dot(sigma_hat_inv).dot(mu_pos_hat)) - np.log((1-phi)/phi)
    lda = deepcopy(grid_search.best_estimator_)
    lda.coef_ = np.array([theta])
    lda.intercept_ = np.array(theta_0)

    lda_scores[i] = lda.score(X_test, y_test)

y_scores = np.vstack((lda_scores.reshape(repeat,1), lr_scores.reshape(repeat,1)))
t1 = np.vstack((np.ones((repeat,1)), np.zeros((repeat,1)) ))
X_dummy = sm.add_constant(t1)
results = sm.OLS(y_scores, X_dummy).fit()
print(results.summary())

plt.figure(1, figsize=(4,4))
plt.scatter(X[y == 0][:,0], X[y == 0][:,1], marker="+", c="red", s=10)
plt.scatter(X[y == 1][:,0], X[y == 1][:, 1], marker="+", c="blue", s=10)
plt.scatter(mu_pos[0], mu_pos[1], marker="o", c="black", s=20)
plt.scatter(mu_neg[0], mu_neg[1], marker="o", c="black", s=20)
plt.show()