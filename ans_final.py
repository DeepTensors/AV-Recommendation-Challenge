import Templates as t

## Reading the data
problem_data = t.pd.read_csv('./train/problem_data.csv')
train_submission = t.pd.read_csv('./train/train_submissions.csv')
user_data = t.pd.read_csv('./train/user_data.csv')
test_csv = t.pd.read_csv('test.csv')

merge = t.pd.read_csv('./train/merge.csv')
mergeTest = t.pd.read_csv('./train/mergeTest.csv')

##Checking for null values
nullValues = t.checkNullValues(merge)

## Points , Tags , Level type , country have null values

### Filling the NA values
t.fillNA(merge,0,True)
t.fillNA(mergeTest,0,True)


### Dropping Attempts , Tags , Country , problem id , user id
X = merge.drop(['attempts_range' , 'tags' ,'problem_id' ,'user_id','country'],axis = 1)
Y = merge.attempts_range
mergeTest = mergeTest.drop(['tags','ID','problem_id' ,'user_id','country'],axis = 1)

### SPLITTING
X_train , X_test , y_train ,y_test = t.Spilt(X,Y,0.8,0)

## Categorical value
categorical_features_indices = t.np.where(X.dtypes != t.np.float)[0]

## Doing CatBoost
##Quantile, LogLinQuantile, Poisson, MAPE, R2.
##Logloss, CrossEntropy, MultiClass, MultiClassOneVsAll, AUC, Accuracy, Precision, Recall, F1, TotalF1, MCC or custom objective object
model = t.CatRegressor(1000,7,0.1,'Quantile' , X_train , y_train , categorical_features_indices,X_test,y_test)

submission = t.pd.DataFrame()
submission['ID'] = test_csv['ID']
submission['attempts_range'] =(model.predict(mergeTest))
submission['attempts_range'] = submission.attempts_range.apply(lambda x: int(x))
submission.to_csv("Submission.csv",index = False)
