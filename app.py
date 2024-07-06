import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

def create_model(data):
    X = data.drop(['Marks'], axis=1)
    y = data['Marks']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Testing the model
    y_pred = model.predict(X_test)
    print("R^2 score:", r2_score(y_test, y_pred))

    # Return the model and test data
    return model, X_test, y_test

def data_cleaning():
    data = pd.read_csv("data/Student_Marks.csv")
    return data

def main():
    data = data_cleaning()
    model, X_test, y_test = create_model(data)
    with open('data/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    X_test.to_csv('data/X_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

if __name__ == '__main__':
    main()
