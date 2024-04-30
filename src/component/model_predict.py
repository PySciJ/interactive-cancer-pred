import pickle
import numpy as np

def get_prediction(input_data):
  model = pickle.load(open(r'C:\e2eproject\interactive-cancer-pred\artifacts\model.pkl', "rb"))
  scaler = pickle.load(open(r'C:\e2eproject\interactive-cancer-pred\artifacts\scaler.pkl', "rb"))
  
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  
  input_array_scaled = scaler.transform(input_array)
  
  prediction = model.predict(input_array_scaled)
  
  return prediction, model , input_array_scaled