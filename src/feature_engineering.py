
from sklearn.cluster import KMeans

def feature_engineering(X):
    X['Family'] = X['Parch'] + X['SibSp'] + 1
    X['IsAlone'] = (X['Family']==1).astype(int)
    X['HasCabin']= X['Cabin'].notna().astype(int)
    X['Pclass_Sex'] = X['Pclass'] * X['Sex']
    X['Age_Class'] = X['Age'] * X['Pclass']
    X['Fare_Family'] = X['Fare'] * X['Family']
    X['Family_Pclass'] = X['Family'] * X['Pclass']
    X['Age_Sex'] = X['Age'] * X['Sex']
    X['Fare_Sex'] = X['Fare'] * X['Sex']
    X['Fare_Class'] = X['Fare'] * X['Pclass']
    X['Age_Fare'] = X['Age'] * X['Fare']
    
    kmeans=KMeans(n_clusters=5, random_state=42)
    X['Cluster']=kmeans.fit_predict(X)
    X['Cluster']=X['Cluster'].astype('int64')
   
    X.drop(columns=['PassengerId','Name', 'Ticket', 'Cabin'], inplace=True)

    return X