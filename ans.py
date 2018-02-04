import Templates as t

## Reading the data
problem_data = t.pd.read_csv('./train/problem_data.csv')
train_submission = t.pd.read_csv('./train/train_submissions.csv')
user_data = t.pd.read_csv('./train/user_data.csv')
test_csv = t.pd.read_csv('test.csv')

## INNER JOIN
merge1 = t.merge(user_data , train_submission ,ON = 'user_id')
merge = t.merge(problem_data , merge1 , ON = 'problem_id')
merge2 = t.merge(test_csv , user_data ,ON = 'user_id')
mergeTest = t.merge(problem_data , merge2 , ON = 'problem_id')

##Checking for null values
nullValues = t.checkNullValues(merge)

## Points , Tags , Level type , country have null values

### Filling the NA values
t.fillNA(merge,0,True)
t.fillNA(mergeTest,0,True)

### Converting DataFrame to SFrame
sf = t.graphlab.SFrame(data = merge) 

### Not filling NA using imputer of graphLab
sf_impute = merge['submission_count','problem_solved','contribution','follower_count','rating','rank']


### Dropping Attempts , Tags , Country , problem id , user id
X = merge.drop(['attempts_range' , 'tags' ,'problem_id' ,'user_id','country'],axis = 1)
Y = merge.attempts_range
mergeTest = mergeTest.drop(['tags','ID','problem_id' ,'user_id','country'],axis = 1)

### SPLITTING
X_train , X_test , y_train ,y_test = t.Spilt(X,Y,0.8,1)

## Categorical value
categorical_features_indices = t.np.where(X.dtypes != t.np.float)[0]

## Doing CatBoost
model = t.CatRegressor(200,4, 0.1 , 'RMSE' , X_train , y_train , categorical_features_indices,X_test,y_test)

submission = t.pd.DataFrame()
submission['ID'] = test_csv['ID']
submission['attempts_range'] =(model.predict(mergeTest))
submission['attempts_range'] = submission.attempts_range.apply(lambda x: round(x))
submission.to_csv("Submission.csv",index = False)
