
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  
# model = AutoModelForSeq2SeqLM.from_pretrained("BigBirdPegasusConfig")
model.to(device)
model.eval()


def get_augmented_caption(sentence):
    text =  "paraphrase: " + sentence + " </s>"

    encoding = tokenizer.encode_plus(text,padding=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")

    output = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=500,
        top_p=0.99,
        # num_beams=30,
        early_stopping=True,
        num_return_sequences=1
    )

    return tokenizer.decode(output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)







if __name__ == "__main__":
    sentence = "Two dogs playing with a ball in the park."

    text =  "paraphrase: " + sentence + " </s>"

    encoding = tokenizer.encode_plus(text,padding=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")


    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=200,
        top_p=0.999,
        early_stopping=True,
        num_return_sequences=1
    )

    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        print(line)