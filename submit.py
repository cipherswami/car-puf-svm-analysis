#############################################################
# File: submit.py <Assignment 1 - Intro to ML: CS771>
# Authors:	
#	231040024 - Aravind Potluri <aravidp23@iitk.ac.in>
#	231040090 - Kintali Saicharan <saicharan23@iitk.ac.in>
#	231040089 - Tangudu Sai Pavan <saipavan23@iitk.ac.in>
#	231040114 - Sushmita Chandra <schandra23@iitk.ac.in>
#	000210790 - Pula Jathin reddy <jathin21@iitk.ac.in>
#	000210806 - Rahul Narayan <rahuln21@iitk.ac.in>
##############################################################

# Dependencies
import numpy as np
import sklearn
from scipy.linalg import khatri_rao

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Helping Functions
def map(X):
    X_int = X.astype(int)
    matrix = np.outer(X_int, X_int)
    rows, _ = matrix.shape
    mask = np.triu_indices(rows, k=1)
    feat = np.concatenate((matrix[mask], X))
    return feat

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################
    X_train_mapped = my_map(X_train)
    model = LinearSVC(dual=False,loss='squared_hinge', C = 10, tol=1e-4, penalty='l2', multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, max_iter=1000)
    model.fit(X_train_mapped, y_train)
    w = model.coef_[0]
    b = model.intercept_[0]
    return w, b

################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
    X = np.cumprod( np.flip( 2 * X - 1 , axis = 1 ), axis = 1 )
    return np.apply_along_axis(map, axis=1, arr=X)