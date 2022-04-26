import cv2
import jsonpickle
import requests
import numpy as np
import pdb
import pickle
import os
from PIL import Image, ImageDraw, ImageFont

def model_inference(image_path):
    with open(image_path, "rb") as stream:
        binary_image = stream.read()
        data = {"image": binary_image}

    response = requests.post("http://localhost:8600/predictions/face-detection-recognition", files=data)
    response = jsonpickle.decode(response.text)
    result = response["face_detection_recognition"]
    bboxes = [fdr["bbox"] for fdr in result]
    if len(result) > 0:
        embedding = np.array([r["normed_embedding"] for r in result])
        data = {"embeddings": pickle.dumps(embedding)}
        response = requests.post("http://localhost:8600/predictions/age-gender", files=data)
        response = jsonpickle.decode(response.text)
        ages = response["ages"]
        genders = response["genders"]
        return ages, genders, bboxes
    else:
        return None


def frame_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            yield frame, count, fps
            count += fps # i.e. at 30 fps, this advances one second
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            cap.release()
            break


def dump_to_file(video_path, frame, idx, output_path="/app/tmp/test"):
    output_path = os.path.join(output_path, os.path.basename(video_path).replace(".mp4", "_" + str(idx) + ".jpg"))
    cv2.imwrite(output_path, frame)
    return output_path


def annotate_image(image_path, genders: list, ages: list, bboxes: list, save_path) -> None:
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("fonts/arial.ttf", 25)

    for gender, age, bbox in zip(genders, ages, bboxes):
        draw.rectangle(bbox.tolist(), outline=(0, 0, 0))
        draw.text(
            (bbox[0], bbox[1]),
            f"AGE: {round(age['mean'])}, ENTROPY: {round(age['entropy'], 4)}",
            fill=(255, 0, 0),
            font=font,
        )
        draw.text(
            (bbox[0], bbox[3]),
            "MALE " + str(round(gender["m"] * 100)) + str("%") + ", "
            "FEMALE "
            + str(round(gender["f"] * 100))
            + str("%")
            + f", ENTROPY: {round(gender['entropy'], 4)}",
            fill=(0, 255, 0),
            font=font,
        )
    image.save(save_path)

if __name__ == "__main__":
    video_paths = ['Shocking Delhi Riot Video Shows Violent Mob Cornering Outnumbered Police _ Injuring Them.mp4',
                   'Police and demonstrators clash in Indonesia over labour law protest.mp4',
                   'Latest Uttar Pradesh News _ Uttar Pradesh Local News 5.mp4', 
                   'Latest Uttar Pradesh News _ Uttar Pradesh Local News 4.mp4']

    video_path = "/app/chris/" + video_paths[3]
    for frame, idx, fps in frame_generator(video_path):
        print(video_path, idx)
        image_path = dump_to_file(video_path, frame, idx)
        results = model_inference(image_path)
        save_path = os.path.join("results/", os.path.basename(image_path))
        if results is not None:
            ages, genders, bboxes = results
            annotate_image(image_path, genders, ages, bboxes, save_path)
