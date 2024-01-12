import cv2
import numpy as np
import torch

from modules.completion_model import CompletionModel
from modules.tokenizer import Tokenizer

tokenizer=Tokenizer(dictionary_path="./models/dictionary.gensim")
comp_model=CompletionModel(tokenizer=tokenizer,embed_dim=256)

text="将茶被放到茶托中"
vid_path="datasets/vid/1/video.mp4"

cap=cv2.VideoCapture(vid_path)

success,img=cap.read()
img=torch.Tensor(img).permute(2,0,1).unsqueeze(0)
text=tokenizer.encode(text)
text=torch.LongTensor(text)

text_token,vid_token=comp_model.encode(text,img)

print("text:",text_token.shape)
print("vid:",vid_token.shape)

img,new_vid_token,mf=comp_model.decode(text_token,vid_token)
print("img:",img.shape)
print("new_vid:",new_vid_token.shape)
print(mf)

torch.save(comp_model.state_dict(),"./model_test")