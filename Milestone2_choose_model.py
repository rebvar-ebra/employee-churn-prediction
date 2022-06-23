from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.linear_model import LogisticRegression


def train_save_model(X,y):
    print(" ---------------------- Splitting train and test dataset : 0.7 ~ 0.3 ------------- ")
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 5)
    print('')
    
    print(" ---------------------  Trying different models ----------------------")
    print(' ')
    
    tree= DecisionTreeClassifier()  
    forest= RandomForestClassifier()
    knn= KNeighborsClassifier()
    svm= SVC()
    lr = LogisticRegression()

    models= [tree, forest, knn, svm, lr]
    choose_model = {}

    for model in models:
        model.fit(X_train, y_train) # train on the train set
        y_pred= model.predict(X_test) # predict on the test set
        accuracy= accuracy_score(y_test, y_pred) # this gives us how often the algorithm predicted correctly
        clf_report= classification_report(y_test, y_pred) # with the report, we have a bigger picture, with precision and recall for each class
        print(f"The accuracy of model {type(model).__name__} is {accuracy:.2f}")
        print(clf_report)
        print("\n")
        choose_model[type(model).__name__] = accuracy
        
    print(' --------------------- Chose the model based on the accuracy ------------------')
    print('')
    print('')

    
    if max(choose_model, key=choose_model.get) == 'SVC':
        print('----------- Started training with SVC and tuning hyper paramters -------------- ')
        print('')

        model = SVC()
        kernel = ['poly', 'rbf', 'sigmoid']
        C = [50, 10, 1.0, 0.1, 0.01]
        # gamma = ['scale']
        # define grid search
        grid = dict(kernel=kernel) #,C=C,gamma=gamma)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
        grid_result = grid_search.fit(X_train, y_train)
        model = grid_result.best_estimator_
        model.fit(X_train, y_train) # training the best model
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        print('Best-score:', grid_result.best_score_)
        print ( 'train-acc: ', model.score(X_train, y_train))
        print('test-acc:', grid_result.score(X_test, y_test))
        joblib.dump(grid_result.best_estimator_, "SupportVectorMachineModel")
    
    elif max(choose_model, key=choose_model.get) == 'DecisionTreeClassifier':
        print('----------- Started training with Decision tree classifier and tuning hyper paramters -------------- ')
        print('')
        
        params = [{'max_depth':[1,2,3,4,5,6,7,8,9,10]}]
        params = [{'max_depth':[1,2,3,4,5,6,7,8,9,10],'min_samples_leaf':[1,5,30]}]
        grid_search = GridSearchCV(estimator=tree.DecisionTreeClassifier(random_state=9),param_grid=params,scoring='accuracy',cv=10,return_train_score=True)
        grid_result = grid_search.fit(X_train, y_train)
        model = grid_result.best_estimator_
        model.fit(X_train, y_train) # training the best model
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        print('Best-score:', grid_result.best_score_)
        print ( 'train-acc: ', model.score(X_train, y_train))
        print('test-acc:', grid_result.score(X_test, y_test))
        joblib.dump(grid_result.best_estimator_, "DecisionTreeModel")
    
    elif max(choose_model, key=choose_model.get) == 'LogisticRegression':
        print('----------- Started training with Logistic regression and tuning hyper paramters -------------- ')
        print('')
        
        model = LogisticRegression()
        solvers = ['newton-cg', 'lbfgs', 'liblinear']
        penalty = ['l2']
        c_values = [100, 10, 1.0, 0.1, 0.01]
        # define grid search
        grid = dict(solver=solvers,penalty=penalty,C=c_values)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
        grid_result = grid_search.fit(X_train, y_train)
        model = grid_result.best_estimator_
        model.fit(X_train, y_train) # training the best model
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        print('Best-score:', grid_result.best_score_)
        print ( 'train-acc: ', model.score(X_train, y_train))
        print('test-acc:', grid_result.score(X_test, y_test))
        joblib.dump(grid_result.best_estimator_, "LogisticRegressionModel")
        
    elif max(choose_model, key=choose_model.get) == 'RandomForestClassifier':
        
        print('----------- Started training with Random forest classifier and tuning hyper paramters -------------- ')
        print('')
        
        model = RandomForestClassifier()
        n_estimators = [10, 100, 1000]
        max_features = ['sqrt', 'log2']
        # define grid search
        grid = dict(n_estimators=n_estimators,max_features=max_features)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
        grid_result = grid_search.fit(X_train, y_train)
        model = grid_result.best_estimator_
        model.fit(X_train, y_train) # training the best model
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        print('Best-score:', grid_result.best_score_)
        print ( 'train-acc: ', model.score(X_train, y_train))
        print('test-acc:', grid_result.score(X_test, y_test))
        joblib.dump(grid_result.best_estimator_, "RandomRorestModel")
    
    else:
        print('----------- Started training with KNearest neighbor and tuning hyper paramters -------------- ')
        print('')
        
        model = KNeighborsClassifier()
        n_neighbors = range(1, 21, 2)
        weights = ['uniform', 'distance']
        metric = ['euclidean', 'manhattan', 'minkowski']
        # define grid search
        grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
        grid_result = grid_search.fit(X_train, y_train)
        model = grid_result.best_estimator_
        model.fit(X_train, y_train) # training the best model
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        print('Best-score:', grid_result.best_score_)
        print ( 'train-acc: ', model.score(X_train, y_train))
        print('test-acc:', grid_result.score(X_test, y_test))
        joblib.dump(grid_result.best_estimator_, "KNearestNeighborModel")
    
    print(' ')
    print(' ------------------- Saved the model into repository --------------')
        
print('finish')
    