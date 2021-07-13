from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
import torch
import pandas as pd
import math

bertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    predictions=bertMaskedLM(tensor_input)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(predictions.squeeze(),tensor_input.squeeze()).data 
    return math.exp(loss)

