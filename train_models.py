import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestRegressor as RForestRegress
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.naive_bayes import BernoulliNB

from sklearn.grid_search import GridSearchCV
import scipy.stats as scst

import os
from sys import argv, path
import numpy as np
import time
import datetime

default_input_dir = "/home/alex/Documents/Development/MLSchool2015/data/hackathon/" #"/home/ubuntu/Data"
default_output_dir = "res"

import os
running_on_codalab = False
run_dir = os.path.abspath(".")
codalab_run_dir = os.path.join(run_dir, "program")
if os.path.isdir(codalab_run_dir): 
    run_dir=codalab_run_dir
    running_on_codalab = True
    print "Running on Codalab!"
lib_dir = os.path.join(run_dir, "lib")
res_dir = os.path.join(run_dir, "res")

# Our libraries  
path.append (run_dir)
path.append (lib_dir)
import data_io                       # general purpose input/output functions
from data_io import vprint           # print only in verbose mode
from data_manager import DataManager # load/save data and get info about them

datanames = data_io.inventory_data(default_input_dir)
verbose = True
debug_mode = 0
zipme = True
max_time = 90
max_cycle = 1
execution_success = True
the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
submission_filename = '../automl_sample_submission_' + the_date


overall_start = time.time()
if len(datanames)>0:
    vprint( verbose,  "************************************************************************")
    vprint( verbose,  "****** Attempting to copy files (from res/) for RESULT submission ******")
    vprint( verbose,  "************************************************************************")
    OK = data_io.copy_results(datanames, res_dir, default_output_dir, verbose) # DO NOT REMOVE!
    if OK: 
        vprint( verbose,  "[+] Success")
        datanames = [] # Do not proceed with learning and testing
    else:
        vprint( verbose, "======== Some missing results on current datasets!")
        vprint( verbose, "======== Proceeding to train/test:\n")
    # =================== End @RESULT SUBMISSION (KEEP THIS) ==================

    # ================ @CODE SUBMISSION (SUBTITUTE YOUR CODE) ================= 
    overall_time_budget = 0
    for basename in datanames: # Loop over datasets
        
        vprint( verbose,  "************************************************")
        vprint( verbose,  "******** Processing dataset " + basename.capitalize() + " ********")
        vprint( verbose,  "************************************************")
        
        # ======== Learning on a time budget:
        # Keep track of time not to exceed your time budget. Time spent to inventory data neglected.
        start = time.time()
        
        # ======== Creating a data object with data, informations about it
        vprint( verbose,  "======== Reading and converting data ==========")
        D = DataManager(basename, default_input_dir, replace_missing=True, filter_features=True, verbose=verbose)
        print D
        
        # ======== Keeping track of time
        if debug_mode<1:
            time_budget = D.info['time_budget']        # <== HERE IS THE TIME BUDGET!
        else:
            time_budget = max_time
        overall_time_budget = overall_time_budget + time_budget
        time_spent = time.time() - start
        vprint( verbose,  "[+] Remaining time after reading data %5.2f sec" % (time_budget-time_spent))
        if time_spent >= time_budget:
            vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
            execution_success = False
            continue
        
        # ========= Creating a model, knowing its assigned task from D.info['task'].
        # The model can also select its hyper-parameters based on other elements of info.  
        # vprint( verbose,  "======== Creating model ==========")
        # M = MyAutoML(D.info, verbose, debug_mode)
        # print M
        
        # ========= Iterating over learning cycles and keeping track of time
        time_spent = time.time() - start
        vprint( verbose,  "[+] Remaining time after building model %5.2f sec" % (time_budget-time_spent))        
        if time_spent >= time_budget:
            vprint( verbose,  "[-] Sorry, time budget exceeded, skipping this task")
            execution_success = False
            continue

        time_budget = time_budget - time_spent # Remove time spent so far
        start = time.time()              # Reset the counter
        time_spent = 0                   # Initialize time spent learning
        time_spent_last = 0                   # Initialize time spent learning
        cycle = 0
        GPU =  False
        
        while cycle <= 1: #max_cycle:
            begin = time.time()
            vprint(verbose,  "=========== " + basename.capitalize() +" Training cycle " + str(cycle) +" ================") 
            n_estimators = 10
            
            sparse = False
            if D.info['is_sparse'] == 1:
                sparse = True
            if cycle == 1:
                models = [RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=5, max_features=0.5),
                        #LogisticRegression(penalty="l1"),
                        #LinearSVC(C=0.5),
                        KNeighborsClassifier(),
                        ExtraTreeClassifier(),
                        AdaBoostClassifier(),
                        GradientBoostingClassifier()]
            else:
                models = [RandomForestClassifier(n_estimators)]
            task = D.info['task']
            fit_models = []
            if task == 'binary.classification':
                for m in models:
                    print m
                    fit_models.append(m.fit(D.data["X_train"], D.data["Y_train"]))
                
#                 y_test, y_val = train_models(D.data["X_train"], D.data["Y_train"], models,
#                                             D.data["X_test"], D.data["X_valid"])
            else:
                vprint( verbose,  "[-] task not recognised")
                break         
            vprint( verbose,  "[+] Fitting success, time spent so far %5.2f sec" % (time.time() - start))
            
#             print y_test, y_val
            # Make predictions
            y_valid = np.zeros((len(models), D.data["X_valid"].shape[0]))
            y_test = np.zeros((len(models), D.data["X_test"].shape[0]))
            if task == 'binary.classification' and not GPU:
                for i in xrange(len(fit_models)):
                    y_valid[i] = fit_models[i].predict(D.data["X_valid"])
                    y_test[i] = fit_models[i].predict(D.data["X_test"])
                
                Y_valid = scst.mode(y_valid, axis=1)
                Y_test = scst.mode(y_test, axis=1)
                #Y_valid = m.predict(D.data["X_valid"])
                #Y_test = m.predict(D.data["X_test"])
                #Y_test =  M.predict_proba(D.data['X_test'])[:, 1]
                # print Y_valid, Y_test
            
            
            vprint( verbose,  "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
                # Write results
            filename_valid = basename + '_valid_' + str(cycle).zfill(3) + '.predict'
            data_io.write(os.path.join(default_output_dir, filename_valid), Y_valid)
            filename_test = basename + '_test_' + str(cycle).zfill(3) + '.predict'
            data_io.write(os.path.join(default_output_dir,filename_test), Y_test)
            vprint( verbose,  "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))         
            time_spent = time.time() - start 
            vprint( verbose,  "[+] End cycle, remaining time %5.2f sec" % (time_budget-time_spent))
            cycle += 1
            time_spent_last = time.time() - begin
            time_budget = time_budget - time_spent_last # Remove time spent so far
            
    if zipme and not(running_on_codalab):
        vprint( verbose,  "========= Zipping this directory to prepare for submit ==============")
        data_io.zipdir(submission_filename + '.zip', ".")
        
    overall_time_spent = time.time() - overall_start
    if execution_success:
        vprint( verbose,  "[+] Done")
        vprint( verbose,  "[+] Overall time spent %5.2f sec " % overall_time_spent + "::  Overall time budget %5.2f sec" % overall_time_budget)
    else:
        vprint( verbose,  "[-] Done, but some tasks aborted because time limit exceeded")
        vprint( verbose,  "[-] Overall time spent %5.2f sec " % overall_time_spent + " > Overall time budget %5.2f sec" % overall_time_budget)
              
    if running_on_codalab: 
        if execution_success:
            exit(0)
        else:
            exit(1)
