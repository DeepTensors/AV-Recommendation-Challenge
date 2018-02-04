import Templates as t

## Reading the data
problem_data = t.graphlab.SFrame('./train/problem_data.csv')
train_submission = t.graphlab.SFrame('./train/train_submissions.csv')
user_data = t.graphlab.SFrame('./train/user_data.csv')
test_csv = t.graphlab.SFrame('test.csv')

## INNER JOIN
merge1 = t.mergeGraphLab(user_data , train_submission ,ON = 'user_id')
merge = t.mergeGraphLab(problem_data , merge1 , ON = 'problem_id')
merge2 = t.mergeGraphLab(test_csv , user_data ,ON = 'user_id')
mergeTest = t.mergeGraphLab(problem_data , merge2 , ON = 'problem_id')


## Points , Tags , Level type , country have null values
###Using imputer of graphLab

sf_impute = merge['submission_count','problem_solved','contribution','follower_count','rating','rank','points']
imputer = t.GLImputer('points',sf_impute)
merget = t.GLFitImputer(imputer,sf_impute)

sf_impute1 = merge['submission_count','problem_solved','contribution','follower_count','rating','rank','level_type']
imputer1 = t.GLImputer('level_type',sf_impute1)
merget1 = t.GLFitImputer(imputer1,sf_impute1)

### For test File
sf_impute = mergeTest['submission_count','problem_solved','contribution','follower_count','rating','rank','points']
imputer = t.GLImputer('points',sf_impute)
merget = t.GLFitImputer(imputer,sf_impute)

sf_impute1 = mergeTest['submission_count','problem_solved','contribution','follower_count','rating','rank','level_type']
imputer1 = t.GLImputer('level_type',sf_impute1)
merget1 = t.GLFitImputer(imputer1,sf_impute1)


## Putting new values
merge['points'] = merget['predicted_feature_points']
merge['level_type'] = merget1['predicted_feature_level_type']

## For test file
mergeTest['points'] = merget['predicted_feature_points']
mergeTest['level_type'] = merget1['predicted_feature_level_type']


merge.save('./train/merge.csv', format = 'csv')
mergeTest.save('./train/mergeTest.csv' , format = 'csv')