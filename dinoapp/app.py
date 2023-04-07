import cv2
import numpy as np
import time
import json
import pickle
from flask_ngrok import run_with_ngrok
from flask import Flask, request, jsonify

import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


app = Flask(__name__)

run_with_ngrok(app)

@app.route('/test',methods=['GET','POST'])
def test():
    
    if request.method == "GET":
        return jsonify({'resposne':'get request called'})
    elif request.method == "POST":

        # req_json = request.json
        # name = req_json['name']
        # return jsonify ({'response': 'hi'+name})
        book_name = request.json.get("book_name")
        captured_image = request.json.get("captured_image")
        img_url  = request.json.get("img_url")
        user_id = request.json.get("user_id")

        if book_name and captured_image and img_url and user_id:

            with open("model\metadata_2000.pkl", "rb") as f:
                metadata = pickle.load(f)

        config_file = 'model\config_2000.yaml'
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        # metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        model_weights = "/content/drive/MyDrive/03 - workplace/01 NSKY/All Dinosaur_model/model_2000.pth"
        cfg.MODEL.WEIGHTS = model_weights

        predictor = DefaultPredictor(cfg)

        img = cv2.imread('model\model_2000.pth')

        outputs = predictor(img)

        new_img = np.zeros_like(img)
        color = {}

        for i in range(len(outputs["instances"])):

            instance = outputs["instances"][i].to("cpu")
            class_id = instance.pred_classes.numpy()[0]
            class_name = metadata.thing_classes[class_id]

            if class_name in ['allosaurus', 'ceratosaurus', 'pterodactylus', 'triceratops']:
                continue

            instance_mask = instance.pred_masks[0].cpu().numpy() 
            mean_color = cv2.mean(img, mask=instance_mask.astype(np.uint8))[0:3]
            mean_color = tuple(map(round,mean_color))
            new_img[np.where(instance_mask == 1)] = mean_color
            color[str(class_name)]= str(mean_color)
            color[str(class_name)+'_mask']= instance_mask.astype(np.uint8)
            
        for k,v in color.items():
            if isinstance(v, np.ndarray):
                color[k] = v.tolist()

        return jsonify(
            {"book_name": book_name,
            "user_id":user_id,
            "page_id":json.dumps(123),
            "page_iamge_link":'s3//123',
            "objects":color})


app.run()