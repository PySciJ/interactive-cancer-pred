import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
from data_preprocessing import cleaned_data

def train_model(data): 
  X = data.drop(['diagnosis'], axis=1)
  y = data['diagnosis']
  
  # scale the data
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  
  # split the data
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=21
  )
  
  # train the model
  model = LogisticRegression()
  model.fit(X_train, y_train)
  
  # test model
  y_pred = model.predict(X_test)
  print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
  print("Classification report: \n", classification_report(y_test, y_pred))
    
  return model, scaler

def main():
  data = cleaned_data()

  model, scaler = train_model(data)

  with open(r'C:\e2eproject\interactive-cancer-pred\artifacts\model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
  with open(r'C:\e2eproject\interactive-cancer-pred\artifacts\scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
  

if __name__ == '__main__':
  main()
