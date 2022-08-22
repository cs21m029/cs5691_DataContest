#####################################################################borderline Smoot Final Submission######################################################################


#0.46 accuracy
import numpy as np
#from google.colab import drive
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import BorderlineSMOTE  


#drive.mount('/content/drive')




# path = 'drive/My Drive/PRML'
# Train1 = pd.read_csv(path+'/Dataset_1_Training.csv')
# Test1 = pd.read_csv(path+'/Dataset_1_Testing.csv')
# Train2 = pd.read_csv(path+'/Dataset_2_Training.csv')
# Test2 = pd.read_csv(path+'/Dataset_2_Testing.csv')

Train1 = pd.read_csv('Dataset_1_Training.csv')
Test1 = pd.read_csv('Dataset_1_Testing.csv')
Train2 = pd.read_csv('Dataset_2_Training.csv')
Test2 = pd.read_csv('Dataset_2_Testing.csv')

Train1 = Train1.transpose()
Train2 = Train2.transpose()
Test1 = Test1.transpose()
Test2 = Test2.transpose()


# 0.475 with scalling
# Co1 scaler logicstic and max_iteration = 1000
# Co2 scaler logicstic and max_iteration = 1000
# Co3 scaler logicstic and max_iteration = 1000
# Co4 scaler Adaboost and 0.2  , 51 
# Co5 scaler Adaboost and 0.2  , 51 
# Co6 scaler Adaboost and 0.2  , 51 


##########################################################################################################################################################co1 logic

x_train = Train1.iloc[1:,0:22283].values
y_train = Train1.iloc[1:,22283].values
x_test = Test1.iloc[1:,0:22283].values

smote = BorderlineSMOTE (sampling_strategy='minority' , random_state=42)
x_train, y_train = smote.fit_sample(x_train.astype(float), y_train.astype(int))

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)


# from sklearn.decomposition import PCA
# pca = PCA()
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

#from sklearn.linear_model import LogisticRegression
#classifierCO1 = LogisticRegression(random_state = 0,max_iter=1000).fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test)

#from sklearn.ensemble import RandomForestClassifier
#classifierCO1 = RandomForestClassifier(random_state = 0,n_estimators = 43,criterion = 'entropy' , max_depth= 5).fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test)

from sklearn.linear_model import LogisticRegression
classifierCO1 = LogisticRegression(random_state = 0,max_iter=1000).fit(x_train.astype(float), y_train.astype(int))
#classifierCO1 = AdaBoostClassifier(n_estimators= 25 , learning_rate= 0.23 , base_estimator = classifierCO1Logic ).fit(x_train.astype(float), y_train.astype(int))
y_pred = classifierCO1.predict(x_test.astype(float))

# classifierCO1 = GradientBoostingClassifier(n_estimators= 25 , learning_rate= 0.25 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

final_pred = y_pred

print(final_pred.shape)



###########################################################################################################################################################co2 logic
#entropy 17 , gini 52
x_train = Train1.iloc[1:,0:22283].values
y_train = Train1.iloc[1:,22284].values
x_test = Test1.iloc[1:,0:22283].values





# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# from sklearn.decomposition import PCA
# pca = PCA()
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

#from sklearn.ensemble import RandomForestClassifier
#classifierCO1 = RandomForestClassifier(random_state = 0,n_estimators = 53,criterion = 'gini' , max_depth= 3).fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test)

from sklearn.linear_model import LogisticRegression
classifierCO1 = LogisticRegression(random_state = 0,max_iter=1000).fit(x_train.astype(float), y_train.astype(int))
#classifierCO1 = AdaBoostClassifier(n_estimators= 25 , learning_rate= 0.23  , base_estimator = classifierCO1Logic).fit(x_train.astype(float), y_train.astype(int))
y_pred = classifierCO1.predict(x_test.astype(float))

# classifierCO1 = GradientBoostingClassifier(n_estimators= 25 , learning_rate= 0.25 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

final_pred = np.append(final_pred, y_pred)

print(final_pred.shape)


################################################################################################################################################co3 with logicstric

x_train = Train2.iloc[1:,0:54675].values
y_train = Train2.iloc[1:,54675].values
x_test = Test2.iloc[1:,0:54675].values

smote = BorderlineSMOTE (sampling_strategy='minority' , random_state=42)
x_train, y_train = smote.fit_sample(x_train.astype(float), y_train.astype(int))

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# from sklearn.decomposition import PCA
# pca = PCA()
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifierCO1 = LogisticRegression(random_state = 0,max_iter=1000).fit(x_train.astype(float), y_train.astype(int))
y_pred = classifierCO1.predict(x_test)

#from sklearn.linear_model import LogisticRegression
#classifierCO1Logic = LogisticRegression(random_state = 0,max_iter=1000)
# classifierCO1 = AdaBoostClassifier(n_estimators= 51 , learning_rate= 0.2 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

# classifierCO1 = GradientBoostingClassifier(n_estimators= 25 , learning_rate= 0.41 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

final_pred = np.append(final_pred, y_pred)

print(final_pred.shape)

##################################################################################################################################################### CO4 adaboost


#co4

x_train = Train2.iloc[1:,0:54675].values
y_train = Train2.iloc[1:,54676].values
x_test = Test2.iloc[1:,0:54675].values

smote = BorderlineSMOTE(sampling_strategy='minority' , random_state=42)
x_train, y_train = smote.fit_sample(x_train.astype(float), y_train.astype(int))

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# from sklearn.decomposition import PCA
# pca = PCA()
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

#from sklearn.neighbors import KNeighborsClassifier
#classifierCO1 = KNeighborsClassifier(n_neighbors = 3 , metric = 'minkowski' , p=2 )
#classifierCO1.fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test.astype(float))

#from sklearn.svm import SVC
#classifierCO1 = SVC(kernel = 'linear' , random_state = 0 )
#classifierCO1.fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test.astype(float))

#from sklearn.linear_model import LogisticRegression
#classifierCO1Logic = LogisticRegression(random_state = 0,max_iter=1000)
# classifierCO1 = GradientBoostingClassifier(n_estimators= 20 , learning_rate= 0.499 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

# classifierCO1 = GradientBoostingClassifier(n_estimators= 25 , learning_rate= 0.23 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

classifierCO1 = AdaBoostClassifier(n_estimators= 51 , learning_rate= 0.2).fit(x_train.astype(float), y_train.astype(int))
y_pred = classifierCO1.predict(x_test.astype(float))

final_pred = np.append(final_pred, y_pred)

print(final_pred.shape)






###########################################################################################################################################################co5
x_train = Train2.iloc[1:,0:54675].values
y_train = Train2.iloc[1:,54677].values
x_test = Test2.iloc[1:,0:54675].values



# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# from sklearn.decomposition import PCA
# pca = PCA()
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

#svm
#from sklearn.svm import SVC
#classifierCO1 = SVC(kernel = 'linear' , random_state = 0 )
#classifierCO1.fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test.astype(float))

#logicstic
#from sklearn.linear_model import LogisticRegression
#classifierCO1 = LogisticRegression(random_state = 0,max_iter=1000).fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test)

# classifierCO1 = GradientBoostingClassifier(n_estimators= 20 , learning_rate= 0.499 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

# classifierCO1 = GradientBoostingClassifier(n_estimators= 25 , learning_rate= 0.25 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

classifierCO1 = AdaBoostClassifier(n_estimators= 51 , learning_rate= 0.2).fit(x_train.astype(float), y_train.astype(int))
y_pred = classifierCO1.predict(x_test.astype(float))

final_pred = np.append(final_pred, y_pred)

print(final_pred.shape)

########################################################################################################################################################co6

x_train = Train2.iloc[1:,0:54675].values
y_train = Train2.iloc[1:,54678].values
x_test = Test2.iloc[1:,0:54675].values



# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# from sklearn.decomposition import PCA
# pca = PCA()
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

#from sklearn.neighbors import KNeighborsClassifier
#classifierCO1 = KNeighborsClassifier(n_neighbors = 15 , metric = 'minkowski' , p=2 )
#classifierCO1.fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test)

#from sklearn.svm import SVC
#classifierCO1 = SVC(kernel = 'linear' , random_state = 0 )
#classifierCO1.fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test.astype(float))

#from sklearn.linear_model import LogisticRegression
#classifierCO1Logic = LogisticRegression(random_state = 0,max_iter=1000)

#best PCA and adaboost
# classifierCO1 = GradientBoostingClassifier(n_estimators= 20 , learning_rate= 0.499 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))


# classifierCO1 = GradientBoostingClassifier(n_estimators= 25 , learning_rate= 0.25 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

classifierCO1 = AdaBoostClassifier(n_estimators= 51 , learning_rate= 0.2).fit(x_train.astype(float), y_train.astype(int))
y_pred = classifierCO1.predict(x_test.astype(float))



final_pred = np.append(final_pred, y_pred)

print(final_pred.shape)


########################################################################################################################################################OUTPUT


l = []
for i in range(0,1056):
  l.append(i)

csv = pd.DataFrame({'Id': l, 'Predicted': final_pred})

csv.to_csv('two.csv',index = False)
csv = pd.read_csv('two.csv' )
print(csv)

#####################################################################Final Submission######################################################################




#0.475 accuracy

#drive.mount('/content/drive')




# path = 'drive/My Drive/PRML'
# Train1 = pd.read_csv(path+'/Dataset_1_Training.csv')
# Test1 = pd.read_csv(path+'/Dataset_1_Testing.csv')
# Train2 = pd.read_csv(path+'/Dataset_2_Training.csv')
# Test2 = pd.read_csv(path+'/Dataset_2_Testing.csv')

Train1 = pd.read_csv('Dataset_1_Training.csv')
Test1 = pd.read_csv('Dataset_1_Testing.csv')
Train2 = pd.read_csv('Dataset_2_Training.csv')
Test2 = pd.read_csv('Dataset_2_Testing.csv')

Train1 = Train1.transpose()
Train2 = Train2.transpose()
Test1 = Test1.transpose()
Test2 = Test2.transpose()


# 0.475 with scalling
# Co1 scaler logicstic and max_iteration = 1000
# Co2 scaler logicstic and max_iteration = 1000
# Co3 scaler logicstic and max_iteration = 1000
# Co4 scaler Adaboost and 0.2  , 51 
# Co5 scaler Adaboost and 0.2  , 51 
# Co6 scaler Adaboost and 0.2  , 51 


##########################################################################################################################################################co1 logic

x_train = Train1.iloc[1:,0:22283].values
y_train = Train1.iloc[1:,22283].values
x_test = Test1.iloc[1:,0:22283].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# from sklearn.decomposition import PCA
# pca = PCA()
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

#from sklearn.linear_model import LogisticRegression
#classifierCO1 = LogisticRegression(random_state = 0,max_iter=1000).fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test)

#from sklearn.ensemble import RandomForestClassifier
#classifierCO1 = RandomForestClassifier(random_state = 0,n_estimators = 43,criterion = 'entropy' , max_depth= 5).fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test)

from sklearn.linear_model import LogisticRegression
classifierCO1 = LogisticRegression(random_state = 0,max_iter=1000).fit(x_train.astype(float), y_train.astype(int))
#classifierCO1 = AdaBoostClassifier(n_estimators= 25 , learning_rate= 0.23 , base_estimator = classifierCO1Logic ).fit(x_train.astype(float), y_train.astype(int))
y_pred = classifierCO1.predict(x_test.astype(float))

# classifierCO1 = GradientBoostingClassifier(n_estimators= 25 , learning_rate= 0.25 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

final_pred = y_pred

print(final_pred.shape)



###########################################################################################################################################################co2 logic
#entropy 17 , gini 52
x_train = Train1.iloc[1:,0:22283].values
y_train = Train1.iloc[1:,22284].values
x_test = Test1.iloc[1:,0:22283].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# from sklearn.decomposition import PCA
# pca = PCA()
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

#from sklearn.ensemble import RandomForestClassifier
#classifierCO1 = RandomForestClassifier(random_state = 0,n_estimators = 53,criterion = 'gini' , max_depth= 3).fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test)

from sklearn.linear_model import LogisticRegression
classifierCO1 = LogisticRegression(random_state = 0,max_iter=1000).fit(x_train.astype(float), y_train.astype(int))
#classifierCO1 = AdaBoostClassifier(n_estimators= 25 , learning_rate= 0.23  , base_estimator = classifierCO1Logic).fit(x_train.astype(float), y_train.astype(int))
y_pred = classifierCO1.predict(x_test.astype(float))

# classifierCO1 = GradientBoostingClassifier(n_estimators= 25 , learning_rate= 0.25 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

final_pred = np.append(final_pred, y_pred)

print(final_pred.shape)


################################################################################################################################################co3 with logicstric

x_train = Train2.iloc[1:,0:54675].values
y_train = Train2.iloc[1:,54675].values
x_test = Test2.iloc[1:,0:54675].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# from sklearn.decomposition import PCA
# pca = PCA()
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifierCO1 = LogisticRegression(random_state = 0,max_iter=1000).fit(x_train.astype(float), y_train.astype(int))
y_pred = classifierCO1.predict(x_test)

#from sklearn.linear_model import LogisticRegression
#classifierCO1Logic = LogisticRegression(random_state = 0,max_iter=1000)
# classifierCO1 = AdaBoostClassifier(n_estimators= 51 , learning_rate= 0.2 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

# classifierCO1 = GradientBoostingClassifier(n_estimators= 25 , learning_rate= 0.41 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

final_pred = np.append(final_pred, y_pred)

print(final_pred.shape)

##################################################################################################################################################### CO4 adaboost


#co4

x_train = Train2.iloc[1:,0:54675].values
y_train = Train2.iloc[1:,54676].values
x_test = Test2.iloc[1:,0:54675].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# from sklearn.decomposition import PCA
# pca = PCA()
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

#from sklearn.neighbors import KNeighborsClassifier
#classifierCO1 = KNeighborsClassifier(n_neighbors = 3 , metric = 'minkowski' , p=2 )
#classifierCO1.fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test.astype(float))

#from sklearn.svm import SVC
#classifierCO1 = SVC(kernel = 'linear' , random_state = 0 )
#classifierCO1.fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test.astype(float))

#from sklearn.linear_model import LogisticRegression
#classifierCO1Logic = LogisticRegression(random_state = 0,max_iter=1000)
# classifierCO1 = GradientBoostingClassifier(n_estimators= 20 , learning_rate= 0.499 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

# classifierCO1 = GradientBoostingClassifier(n_estimators= 25 , learning_rate= 0.23 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

classifierCO1 = AdaBoostClassifier(n_estimators= 51 , learning_rate= 0.2).fit(x_train.astype(float), y_train.astype(int))
y_pred = classifierCO1.predict(x_test.astype(float))

final_pred = np.append(final_pred, y_pred)

print(final_pred.shape)






###########################################################################################################################################################co5
x_train = Train2.iloc[1:,0:54675].values
y_train = Train2.iloc[1:,54677].values
x_test = Test2.iloc[1:,0:54675].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# from sklearn.decomposition import PCA
# pca = PCA()
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

#svm
#from sklearn.svm import SVC
#classifierCO1 = SVC(kernel = 'linear' , random_state = 0 )
#classifierCO1.fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test.astype(float))

#logicstic
#from sklearn.linear_model import LogisticRegression
#classifierCO1 = LogisticRegression(random_state = 0,max_iter=1000).fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test)

# classifierCO1 = GradientBoostingClassifier(n_estimators= 20 , learning_rate= 0.499 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

# classifierCO1 = GradientBoostingClassifier(n_estimators= 25 , learning_rate= 0.25 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

classifierCO1 = AdaBoostClassifier(n_estimators= 51 , learning_rate= 0.2).fit(x_train.astype(float), y_train.astype(int))
y_pred = classifierCO1.predict(x_test.astype(float))

final_pred = np.append(final_pred, y_pred)

print(final_pred.shape)

########################################################################################################################################################co6

x_train = Train2.iloc[1:,0:54675].values
y_train = Train2.iloc[1:,54678].values
x_test = Test2.iloc[1:,0:54675].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# from sklearn.decomposition import PCA
# pca = PCA()
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

#from sklearn.neighbors import KNeighborsClassifier
#classifierCO1 = KNeighborsClassifier(n_neighbors = 15 , metric = 'minkowski' , p=2 )
#classifierCO1.fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test)

#from sklearn.svm import SVC
#classifierCO1 = SVC(kernel = 'linear' , random_state = 0 )
#classifierCO1.fit(x_train.astype(float), y_train.astype(int))
#y_pred = classifierCO1.predict(x_test.astype(float))

#from sklearn.linear_model import LogisticRegression
#classifierCO1Logic = LogisticRegression(random_state = 0,max_iter=1000)

#best PCA and adaboost
# classifierCO1 = GradientBoostingClassifier(n_estimators= 20 , learning_rate= 0.499 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))


# classifierCO1 = GradientBoostingClassifier(n_estimators= 25 , learning_rate= 0.25 ).fit(x_train.astype(float), y_train.astype(int))
# y_pred = classifierCO1.predict(x_test.astype(float))

classifierCO1 = AdaBoostClassifier(n_estimators= 51 , learning_rate= 0.2).fit(x_train.astype(float), y_train.astype(int))
y_pred = classifierCO1.predict(x_test.astype(float))



final_pred = np.append(final_pred, y_pred)

print(final_pred.shape)


########################################################################################################################################################OUTPUT


l = []
for i in range(0,1056):
  l.append(i)

csv = pd.DataFrame({'Id': l, 'Predicted': final_pred})

csv.to_csv('one.csv',index = False)
csv = pd.read_csv('one.csv' )
print(csv)
