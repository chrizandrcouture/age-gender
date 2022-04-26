import os
import glob
import shutil
import pdb
import jsonpickle
import requests
import pickle
import json
from tqdm import tqdm


def segregrate_dataset(basepath="/app/chris/", dirname=['part1', 'part2', 'part3'], outpath='indian_faces'):
    ind_cnt = 0
    other_cnt = 0
    images = []
    for dir_ in dirname:
        img_dir = os.path.join(basepath, dir_, "*.jpg")
        images.extend(glob.glob(img_dir))

    for img_path in images:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        if(len(basename.split('_'))==4):
            age, gender, race, _ = basename.split('_')
            src = img_path
            if(race == str(3)):
                print(race, type(race))
                name = "_".join(basename.split('_')[:-2]) + '_' + str(ind_cnt)
                dst_path = os.path.join(outpath, name+'.jpg')
                shutil.move(src, dst_path)
                ind_cnt += 1


import jsonpickle
import requests

def model_inference(image_path):
    with open(image_path, "rb") as stream:
        binary_image = stream.read()
    data = {"image": binary_image}

    response = requests.post("http://localhost:8600/predictions/face-detection-recognition", files=data)
    response = jsonpickle.decode(response.text)
    result = response["face_detection_recognition"]
    r = result[0]
    embedding = r["normed_embedding"]
    data = {"embeddings": pickle.dumps(embedding)}
    response = requests.post("http://localhost:8600/predictions/age-gender", files=data)
    response = jsonpickle.decode(response.text)
    return response


def process_images(image_path):
    results = {}
    images = os.listdir(image_path)
    images = [x for x in images if x.endswith(".jpg")]
    for img in tqdm(images):
        result = model_inference(os.path.join(image_path, img))
        results[img] = result
    return results

def evaluate_results(results):
    total = len(results)
    tp = 0
    fp = 0
    fn = 0
    for r in results:
        label = int(r.split("_")[0])
        pred = int(results[r]["ages"][0]["mean"])
        if label <= 13 and pred <= 13:
            tp += 1
        if label > 13 and pred <= 13:
            fp += 1
        if label <= 13 and pred > 13:
            fn += 1
        else:
            tp += 1
    return tp, fp, fn


if __name__ == "__main__":
    # segregrate_dataset()
    # results = process_images("indian_faces/")
    # json.dump(results, open("indian_face_results.json", "w"), indent=4)
    results = json.load(open("indian_face_results.json"))
    tp, fp, fn = evaluate_results(results)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print("Precision: {}, Recall: {}, F1: {}".format(precision, recall, f1))
    pdb.set_trace()