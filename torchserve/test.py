import pdb
import pickle
import requests
import jsonpickle


data = pickle.load(open("test_embedding.pkl", "rb"))
data = pickle.dumps(data)
data = {"embeddings": data}
# data = jsonpickle.encode(data)
response = requests.post("http://localhost:8600/predictions/age-gender", files=data)
response = jsonpickle.decode(response.text)
print(response)
