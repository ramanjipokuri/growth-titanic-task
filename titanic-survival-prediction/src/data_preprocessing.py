import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv("https://www.kaggle.com/datasets/brendan45774/test-file")
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    return df

def preprocess_data(df):
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Define feature types
    num_features = ["Age", "Fare", "SibSp", "Parch"]
    cat_features = ["Sex", "Embarked", "Pclass"]

    # Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return preprocessor, X_train, X_test, y_train, y_test
