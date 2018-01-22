# This module hosts a set of functions to use sklearn's random forest algorithms
# to calibrate, test and validate random forest regression and classification
# models.

import numpy as np
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# This function constructs the matrix hosting the range of dependent variables
# that will be used to train and validate the random forest model
#
# The input data is a dictionary containing the dependent variables that will
# be ingested (numpy arrrays). These can be continuous data or classified data.
# The latter will be converted into integer labels so that they are easily
# incorporated into the RF analysis. The target variable is also read in. It
# should have the same shape as the input variables. In the case of time series,
# it should have the same shape as the time series data. Time invariant data that
# is also considered will need to match the shape not including the timestep
#
# The output is a numpy array containing the dependent variables to be ingested
# (It has dimensions: number of variables x number of data) and the corresponding
# vector of target variables that will act as training and validation. 
#
# There is scope to include time series here as input. In this case, need to
# give the number of timesteps, and the dimension axis of the input variables
# corresponding to this (default assumed is axis 0).
#
# A sanity check will be performed to ensure that all the data have the correct
# dimensions and an error will be returned if this is not the case, with the
# offending variable excluded from the analysis (replaced by -9999).
def construct_variables_matrices(dependent_variables_dict,target_variable,time_series=False, timestep_axis=0):

    variables = dependent_variables_dict.keys()
    n_vars = len(variables)
    n=target_variable.size
    if time_series:
        if timestep_axis == 0:
            target_variable_vector = np.ravel(target_variable)
        
    dep_variables_matrix = np.zeros((n,n_vars))-9999.
    
    # Loop through the dependent variables and check that we have the correct
    # dimensions, then load into matrices if ok
    for vv in range(0,n_vars):
        var = dependent_variables_dict[variables[vv]]
        var_dimensions = np.asarray(var.shape)
        if time_series==True:            
            n_steps = target_variable.shape[timestep_axis]
            if var.shape == target_variable.shape:
                print '\t',variables[vv],'\t',var.shape,'\ttime series'
                if timestep_axis == 0:
                    dep_variables_matrix[:,vv]=np.ravel(var)
            else:
                axes = np.arange(target_variations.size)
                data_dimensions = np.asarray(target_variable.shape)[axes!=timestep_axis]
                n_data = np.product(data_dimensions)
                if np.asarray(var.shape)==data_dimensions:
                    print '\t',variables[vv],'\t',var.shape,'\tconstant'
                    for tt in range(0,n_steps):
                        dep_variables_matrix[tt*n_data:(tt+1)*n_data,vv] = np.ravel(var)
                    else:
                        print '\tERROR - dimensions not consistent'

        # Simple case for non time series data
        else:
            print '\t',variables[vv],'\t',var.shape
            if var.shape == target_variable.shape:
                dep_variables_matrix[:,vv]=np.ravel(var)
            else:
                print '\tERROR - dimensions not consistent'

    # Finally mask out rows containing nans etc.
    dep_finite = np.sum(np.isfinite(dep_variables_matrix),axis=1)==n_vars
    target_finite = np.isfinite(target_variable_vector)
    mask = np.all((dep_finite,target_finite),axis=0)
    
    return dep_variables_matrix[mask], target_variable_vector[mask], variables

# Calibrate and validate the random forest regression model
# Inputs are the matrix of dependent variables, and the associated vector of
# target variables. Other optional arguements pertain to the random forest
# parameters.
# Current iteration does not allow nodata values
# 
def random_forest_regression_model_calval(dependent,target,n_trees_in_forest = 100, min_samples_leaf = 50, n_cores = 1):
    # split the data into calibration and validation datasets
    dep_cal, dep_val, target_cal, target_val = train_test_split(dependent,target,train_size=0.5)

    # build random forest regression model
    rf = RandomForestRegressor(n_estimators=n_trees_in_forest, min_samples_leaf=min_samples_leaf,  bootstrap=True, oob_score=True, n_jobs=n_cores)
    rf.fit(dep_cal,target_cal)

    # report calibration and validation statistics
    cal_score = rf.score(dep_cal,target_cal)
    val_score = rf.score(dep_val,target_val)
    print "\t\tcalibration score = %.3f" % cal_score
    print "\t\tvalidation score = %.3f" % val_score

    # get variable importance
    importances = rf.feature_importances_
    
    return rf, cal_score, val_score, importances

# Calibrate and validate the random forest classifier model
# Inputs are the matrix of dependent variables, and the associated vector of
# target variables. Other optional arguements pertain to the random forest
# parameters.
# Current iteration does not allow nodata values
# 
def random_forest_classifier_calval(dependent,target,n_trees_in_forest = 100, min_samples_leaf = 50, n_cores = 1):
    # split the data into calibration and validation datasets
    dep_cal, dep_val, target_cal, target_val = train_test_split(dependent,target,train_size=0.5)

    # build random forest regression model
    rf = RandomForestClassifier(n_estimators=n_trees_in_forest, min_samples_leaf=min_samples_leaf,  bootstrap=True, oob_score=True, n_jobs=n_cores)
    rf.fit(dep_cal,target_cal)

    # report calibration and validation statistics
    cal_score = rf.score(dep_cal,target_cal)
    val_score = rf.score(dep_val,target_val)
    print "\t\tcalibration score = %.3f" % cal_score
    print "\t\tvalidation score = %.3f" % val_score

    # get variable importance
    importances = rf.feature_importances_
    return rf, cal_score, val_score, importances


# This function applies the calibrated random regression model
# Inputs are the dependent variables on which the prediction is made and
# the calibrated random forest regression model
def apply_random_forest_regression_model(dependent,rf):
    model = rf.predict(dependent)
    return model

# This function applies the calibrated random classifier model
# Inputs are the dependent variables on which the prediction is made and
# the calibrated random forest regression model
def apply_random_forest_regression_model(dependent,rf):
    model = rf.predict(dependent)
    prob = rf.predict_proba(dependent)
    return model, prob

    
# This function just runs the validation script for a specified set of
# dependent and target variables.
def validate_random_forest_model(dependent,target,rf):
    val_score = rf.score(dependent,target)
    print "\t\tvalidation score = %.3f" % val_score
    return val_score

"""
def random_forest_regression(rs_variables,target_variable,n_trees_in_forest = 100, min_samples_leaf = 50, n_cores = 1):
    print "\t downsampling with random forest regression"
    # first of all get rid of nodata values in target data to build random forest
    rs = rs_variables[np.isfinite(target_variable),:]
    target = target_variable[np.isfinite(target_variable)]
    # split the data into calibration and validation datasets
    rs_cal, rs_val, target_cal, target_val = train_test_split(rs,target,train_size=0.5)
    # build random forest regressor
    randomforest = RandomForestRegressor(n_estimators=n_trees_in_forest, min_samples_leaf=min_samples_leaf,  bootstrap=True, oob_score=True, n_jobs=n_cores)
    randomforest.fit(rs_cal,target_cal)

    print "\t\tcalibration score = %.3f" % randomforest.score(rs_cal,target_cal)
    print "\t\tvalidation score = %.3f" % randomforest.score(rs_val,target_val)
    rs_rf = randomforest.predict(rs_variables)
    return rs_rf
"""
