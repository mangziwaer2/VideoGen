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
print("input_shape:",img.shape)
text=tokenizer.encode(text)
text=torch.LongTensor(text)

text_token,vid_token,_=comp_model.encode(text,img)

img,new_vid_token,mf=comp_model.decode(text_token,vid_token)

frame_idx=0
while True:
    frame_idx+=1

    img,vid_token,mf=comp_model.decode(text_token,vid_token)
    img=img[0].permute(1,2,0).detach().cpu().numpy()
    cv2.imshow("vid",img)
    cv2.waitKey(30)