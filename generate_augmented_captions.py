from paraphrase_generator import get_augmented_caption
from tqdm import tqdm

with open('dataset/Flickr8k.token.txt') as f:
    lines = f.readlines()

#remove new line character
for i in range(len(lines)):
    lines[i] = lines[i].replace("\n", "")
    lines[i] = lines[i].replace(" .", ".")
    lines[i] = lines[i].replace(" ,", ".")
new_captions = []

idx = 0

for line in tqdm(lines):
    
    id = line.split(".")[0]
    caption = line.split("\t")[-1]
    if caption[-1] != ".":
        caption += "."
    caption = caption.replace(".", " .")
    old_cap = id + ".jpg#" + str(idx) + "\t" + caption 
    new_captions.append(old_cap)
    idx += 1
    caption = caption.replace(" .", ".")

    aug_cap = get_augmented_caption(caption)

    aug_cap = aug_cap.replace(".", " .")
    aug_cap = aug_cap.replace(",", " ,")

    new_cap = id + ".jpg#" + str(idx) + "\t" + aug_cap 
   
    new_captions.append(new_cap)
    idx += 1
    if idx > 9: idx = 0


with open('new_caps.txt', 'w') as f:
    for item in tqdm(new_captions):
        f.write("%s\n" % item)
    

