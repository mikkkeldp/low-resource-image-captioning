import torch
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import numpy as np

import pickle
import configparser
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNNWithAttention
from PIL import Image
import cv2
import skimage.transform

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def get_coords(im, where):
    imgheight=im.shape[0]
    imgwidth=im.shape[1]
    y1 = 0
    M = int(imgheight/14)
    N = int(imgwidth/14)

    ind = 0
    for y in range(0,imgheight,M):
        for x in range(0, imgwidth, N):
            y1 = y + M
            x1 = x + N

            if ind == where:
                x = [x, x1]
                y = [y,y1]
                return int(np.mean(x)) , int(np.mean(y))

            ind+=1
    return None, None



def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((int(args["image_size"]), int(args["image_size"]))),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args["vocab_path"], 'rb') as f:
        vocab = pickle.load(f)


    #TODO :Add support for glove
    embedding_matrix = None

    # Build models
    encoder = EncoderCNN(int(args["encoded_image_size"]), args["cnn"],device).eval()
    decoder = DecoderRNNWithAttention(int(args["embed_size"]), int(args["attention_size"]), int(args["hidden_size"]), len(vocab), encoder_size=int(args["encoder_size"]), glove = args["glove"], embedding_matrix = embedding_matrix).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args["encoder_path"]), strict=False)
    decoder.load_state_dict(torch.load(args["decoder_path"]), strict=False)

    # Prepare an image
    image = load_image(args["image"], transform)
    image_tensor = image.to(device)

    id = args["image"].split("/")[-1]
    id = id.split(".")[0]
    ids = [id]
    # Generate an caption from the image
    features = encoder(image_tensor, int(args["batch_size"]), int(args["encoder_size"]), ids)

    # get generated caption and attention locations
    sampled_seqs, complete_seqs_loc = decoder.sample_beam_search(features, vocab, device, beam_size=int(args["beam_size"]))

    # print(complete_seqs_loc)



    sampled_seqs = sampled_seqs[0]

    locs = []
    beta_trim = []
    i = 0

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_seqs:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        locs.append(complete_seqs_loc[i])
        # beta_trim.append(betas[i])
        i+= 1
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    print(sentence)

    # img_dim = 244
    # attention_index = []
    # for t in range(len(sentence.split(" "))+1):
    #     current_alpha = alphas[0][t, :].cpu().detach().numpy()
    #     max_indx = np.argmax(current_alpha)
    #     attention_index.append(max_indx)

    # attention_index = attention_indexcl



    # # # plot attention

    # # ### TODO shifted location entries by 7

    # split_sentence = sentence.split(" ")
    # # split_sentence = split_sentence[]

    # print("Caption: ", " ".join(split_sentence[1:-1]))
    # im =  cv2.imread(args["image"])

    # if len(split_sentence) % 4 == 0:
    #     rows = int(len(split_sentence)/4)
    # else:
    #     rows = int(len(split_sentence)/4) + 1

    # cols = 4
    # fig, axis = plt.subplots(rows,cols, figsize=(10,10))

    # ind = 0
    # for r in range(rows):
    #     for c in range(cols):
    #         if ind < len(split_sentence):
    #             # shift location by 7
    #             x, y = get_coords(im, attention_index[ind])
    #             overlay = im.copy()
    #             cv2.circle(overlay, (x,y), 60, (255, 255, 255), -1)
    #             # cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), -1)  # A filled rectangle
    #             image_new = cv2.addWeighted(overlay, 0.8, im, 1 - 0.8, 0)
    #             axis[r][c].imshow(np.asarray(cv2.cvtColor(image_new, cv2.COLOR_BGR2RGB)))
    #             im =  cv2.imread(args["image"])
    #             axis[r][c].axis("off")
    #             axis[r][c].title.set_text(split_sentence[ind])
    #             ind += 1
    #         else:
    #            axis[r][c].axis("off")




    # plt.show()

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")
    args = config["sample"]
    main(args)
