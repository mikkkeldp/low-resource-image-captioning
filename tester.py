from os import listdir
from os.path import isfile, join
from get_faster_rcnn_features import get_prediction
from PIL import Image 
import cv2
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

linear = nn.Linear(2048, 1024)
transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])

faster_rcnn_threshold = 0.3
resnet = models.resnet152(pretrained=True)  ##maybe try 152/101?
modules = list(resnet.children())[:-1]      # delete the last fc layer.
object_crop_encoder = nn.Sequential(*modules)

if torch.cuda.is_available():
    object_crop_encoder.to(device)
    linear.to(device)

def encoder_object_crop(img):
        
        img = img.unsqueeze(0)
        
        with torch.no_grad():
            features = object_crop_encoder(img)

        features = features.reshape(features.size(0), -1)
        return features


mypath = "dataset/Flickr8k_Dataset/"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


for img in tqdm(onlyfiles):
    id = img.split(".")[0]
    pred_boxes, pred_class, pred_score = get_prediction(id, faster_rcnn_threshold)
            
    #get 10 or less cropped images
    cropped_images = []
    img = Image.open(mypath + id + ".jpg")

    if len(pred_boxes) < 10:
        for box in pred_boxes: 
            crop_img = img.crop( (box[0][0], box[0][1], box[1][0], box[1][1]))
            cropped_images.append(crop_img)
    else:    
        for i in range(10):
            crop_img = img.crop( (pred_boxes[i][0][0], pred_boxes[i][0][1], pred_boxes[i][1][0], pred_boxes[i][1][1]))
            cropped_images.append(crop_img)

    # fill the remaining attention regions with full image 
    for i in range(10-len(pred_boxes)):
        cropped_images.append(img)
    
    # convert cropped images to tensors
    tensor_imgs = []
    for img in cropped_images:
        tens_img = transform(img).to(device)
        # tens_img = tens_img.unsqueeze(0)
        tensor_imgs.append(tens_img)

    
    # encode cropped image tensors
    fv_array = []
    for tens_img in tensor_imgs:
        encoded_img = encoder_object_crop(tens_img)
        encoded_img = linear(encoded_img)
        fv_array.append(encoded_img)

    # stack cropped image tensor encodings
    fv_array = torch.stack(fv_array)
    fv_array = torch.squeeze(fv_array, 2)

    with open("faster_rcnn_extracted_features/" + id + '_features.pickle', 'wb') as handle:
        pickle.dump(fv_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("faster_rcnn_extracted_features/" + id + '_classes.pickle', 'wb') as handle:
        pickle.dump(pred_class, handle, protocol=pickle.HIGHEST_PROTOCOL)

