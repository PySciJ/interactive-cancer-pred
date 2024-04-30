import pandas as pd


def cleaned_data():
    data = pd.read_csv(r"C:\e2eproject\interactive-cancer-pred\artifacts\data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })

    return data


