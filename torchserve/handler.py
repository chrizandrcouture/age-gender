import os
import sys
import json
import time

def install_packages():
    try:
        import torch
    except ImportError:
        os.system('python3 -m pip install -r requirements.txt')
        print('System path is as follows: ', sys.path)

install_packages()
import torch

def unpack_dependencies():
    if not os.path.exists('./utils/util.py'):
        print('unpacking dependencies')
        os.system('tar -xzf files.tar.gz')
        time.sleep(2)

unpack_dependencies()

import io
import jsonpickle
import numpy as np
import torch
from tqdm import tqdm

from model.model import ResMLP
from utils import enable_dropout, forward_mc, read_json


def get_models(device):
    models = {"age": None, "gender": None}

    for model_ in ["age", "gender"]:
        model = ResMLP(**read_json(f"./models/{model_}.json")["arch"]["args"])
        checkpoint = f"models/{model_}.pth"
        checkpoint = torch.load(checkpoint, map_location=torch.device(device))
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        enable_dropout(model)
        models[model_] = model
    return models


def handle(data, context=None):
    if data is None:
        return None

    # data = jsonpickle.decode(data)
    embeddings = data[0]["embeddings"]
    embeddings = io.BytesIO(embeddings)
    embeddings = np.load(embeddings, allow_pickle=True)
    embeddings = embeddings.reshape(-1, 512).astype(np.float32)

    genders = []
    ages = []

    for embedding in tqdm(embeddings):
        embedding = embedding.reshape(1, 512)
        gender_mean, gender_entropy = forward_mc(_service["gender"], embedding)
        age_mean, age_entropy = forward_mc(_service["age"], embedding)
        gender = {"m": 1 - gender_mean, "f": gender_mean, "entropy": gender_entropy}
        age = {"mean": age_mean, "entropy": age_entropy}

        genders.append(gender)
        ages.append(age)

    response = {"ages": ages, "genders": genders}
    return [response]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_service = get_models(device=device)


if __name__ == "__main__":
    import pickle
    data = pickle.load(open("test_embedding.pkl", "rb"))
    data = pickle.dumps(data)
    data = {"embeddings": data}
    response = handle(data)
    print(response)
