import torch
from PIL import Image
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import pickle 
import cv2
import random
import argparse
from mscoco_classes import classes

classes_dict = {}
for classz in classes:
  id = classz["id"]
  this_class = classz["name"]
  classes_dict[id] = this_class

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="./output/faster-rcnn-flickr30k_10_epochs.pt",
                help="path to the model")
ap.add_argument("-i", "--image", default="flickr30k_entities-master/images/val/301246.jpg", help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.8, 
                help="confidence to keep predictions")
args = vars(ap.parse_args())

with open('classes_array.pickle', 'rb') as handle:
            CLASS_NAMES = pickle.load(handle)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')      

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)


# CLASS_NAMES = ["people", "clothing", "bodyparts", "animals", "vehicles", "instruments", "scene", "other", "background"]
def get_prediction(id, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.
    """

    imgs_path = "dataset/Flickr8k_Dataset/"
    img = Image.open(imgs_path + id + ".jpg")
    transform = T.Compose([T.ToTensor()])
    img = transform(img).to(device)
    pred = model([img])
    pred_class = [classes_dict[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    try:
      pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
    except:
      try:
        pred_t = [pred_score.index(x) for x in pred_score if x>0][-1]
      except:
        return [], [], []
    
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_score = pred_score[:pred_t+1]
  


    return pred_boxes, pred_class, pred_score
   
def detect_object(img_path, confidence=0.5, rect_th=2, text_size=0.5, text_th=1):
    """
    object_detection_api
      parameters:
        - img_path - path of the input image
        - confidence - threshold value for prediction score
        - rect_th - thickness of bounding box
        - text_size - size of the class label text
        - text_th - thickness of the text
      method:
        - prediction is obtained from get_prediction method
        - for each prediction, bounding box is drawn and text is written 
          with opencv
        - the final image is displayed
    """

    boxes, pred_cls, pred_score = get_prediction(img_path, confidence)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(len(boxes))
    for i in range(len(boxes)):
      cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
      cv2.putText(img,pred_cls[i]+": "+str(round(pred_score[i],3)), boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
  
if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #load fine-tuned model
    # model = torch.load(args["model"])

    #load mscoco pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)

    img_path = args["image"]
    detect_object(img_path, confidence=args["confidence"])
    