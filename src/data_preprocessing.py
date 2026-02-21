from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

def basic_preprocessing(X):
    #Preprocess data: handle missing values, encode categorical variables, etc.
    categorical = list(X.select_dtypes(include=['object']).columns)
    numerical = list(X.select_dtypes(include=['int64', 'float64']).columns)

    inputer_for_numerical = SimpleImputer(strategy="median")
    inputer_for_categorical = SimpleImputer(strategy="most_frequent")
    X[numerical]=inputer_for_numerical.fit_transform(X[numerical])

    X[categorical]=inputer_for_categorical.fit_transform(X[categorical])

    encoder=OrdinalEncoder()
    X[categorical]=encoder.fit_transform(X[categorical])
    
    return X