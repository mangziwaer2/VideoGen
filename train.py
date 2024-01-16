import os

import cv2
import torch
from torch import nn

from modules.completion_model import CompletionModel
from modules.tokenizer import Tokenizer
from dataset import Dataset
from modules.VQGAN.modules.losses import vqperceptual

from tools.parser import args

epoch=args.epoch
batch_size=1
lr=args.lr

memory_length=args.memory_length

dataset_path_root=args.dataset_path
dictionary_path=args.dictionary_path
limit_frame_length=args.limit_frame_length

print("\nargs:")
print("------------------------------------------------------------------\n")
print(f"epoch:{epoch}\n"
      f"lr:{lr}\n"
      f"limit_frame_length:{limit_frame_length}\n"
      f"memory_length:{memory_length}\n"
      f"dataset_path:{dataset_path_root}\n"
      f"dictionary_path:{dictionary_path}"
      )
print("\n------------------------------------------------------------------\n")


train_dataset=Dataset(dataset_path_root=dataset_path_root,train=True)
test_dataset=Dataset(dataset_path_root=dataset_path_root,train=False)

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

tokenizer=Tokenizer(dictionary_path=dictionary_path)

model=CompletionModel(tokenizer,embed_dim=512,max_len=2048).to(device)


vf_criterion=nn.CrossEntropyLoss()
loss_fn=vqperceptual.VQLPIPSWithDiscriminator(device=device)

if(len(train_loader)/2)<1000:
    loss_fn.discriminator_iter_start=len(train_loader)//2

print("discriminator_iter_start:",loss_fn.discriminator_iter_start)

step_update=False

optimizer_ae = torch.optim.Adam(list(model.vqmodel.encoder.parameters()) +
                                list(model.vqmodel.decoder.parameters()) +
                                list(model.vqmodel.quantize.parameters()) +
                                list(model.vqmodel.quant_conv.parameters()) +
                                list(model.vqmodel.post_quant_conv.parameters()) +
                                list(model.text_encoder.embedding.parameters()) +
                                list(model.text_encoder.block.parameters()) +
                                list(model.vid_model.parameters()) +
                                list(model.current_state_model.parameters()),
                                lr=lr, betas=(0.5, 0.9))

optimizer_disc = torch.optim.Adam(loss_fn.discriminator.parameters(),
                            lr=lr, betas=(0.5, 0.9))

schduler_ae=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ae, mode='min', factor=0.1, patience=2)
schduler_disc=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_disc, mode='min', factor=0.1, patience=2)

min_loss=100000
for e in range(epoch):

    if e>=epoch/2:
        step_update=True

    total_losses_ae=[]
    total_losses_disc=[]
    model.train()
    img=None
    for i,(description,video_path) in enumerate(train_loader):
        text=tokenizer.encode(description)
        frame_idx=0
        video_path=video_path[0]
        video_names=os.listdir(video_path)
        frame_length=len(video_names)
        video_path_sub=os.path.join(video_path,video_names[0])
        img=cv2.imread(video_path_sub)
        img=torch.Tensor(img).permute(2,0,1).unsqueeze(0).to(device)
        text=torch.LongTensor(text).to(device)
        valid_frame=torch.eye(2)[1].unsqueeze(0).repeat(batch_size, 1).to(device)

        text_token,vid_token,qloss=model.encode(text,img)

        current_total_losses_ae=[]
        current_total_losses_disc=[]

        frame_length = min(limit_frame_length,frame_length)
        video_names=video_names[:frame_length]
        loss_disc=None
        #已处理第0帧
        for frame_idx in range(len(video_names)):#当前帧状态valid_frame，1为有效，0为无效
            frame_idx+=1

            if frame_idx==frame_length:#无效帧
                valid_frame = torch.eye(2)[0].unsqueeze(0).repeat(batch_size, 1).to(device)
                img=torch.zeros_like(img)
            else:#有效帧
                video_path_sub=os.path.join(video_path,video_names[frame_idx])
                img = cv2.imread(video_path_sub)
                img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

            pred_img, vid_token, pred_vf=model.decode(text_token, vid_token, memory_length=memory_length)
            img=torch.resize_as_(img,pred_img)

            vf_loss=vf_criterion(valid_frame, pred_vf)
            loss_ae, log_dict_ae = loss_fn(qloss, img, pred_img, vf_loss, 0, i,
                                           last_layer=model.vqmodel.get_last_layer(), split="train")

            loss_disc, log_dict_disc = loss_fn(qloss, img, pred_img, vf_loss, 1, i,
                                               last_layer=model.vqmodel.get_last_layer(), split="train")

            current_total_losses_ae.append(loss_ae)

            current_total_losses_disc.append(loss_disc)

            if step_update:
                optimizer_ae.zero_grad()
                loss_ae.backward()
                optimizer_ae.step()

                optimizer_disc.zero_grad()
                loss_disc.backward()
                optimizer_disc.step()

        if not step_update:
            loss_ae= sum(current_total_losses_ae) / len(current_total_losses_ae)
            total_losses_ae.append(loss_ae.item())

            loss_disc= sum(current_total_losses_disc) / len(current_total_losses_disc)
            total_losses_disc.append(loss_disc.item())

            optimizer_ae.zero_grad()
            loss_ae.backward()
            optimizer_ae.step()

            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()

    print(F"{e}/{epoch},train_loss_ae:{(sum(total_losses_ae) / len(total_losses_ae))},train_loss_disc:{(sum(total_losses_disc) / len(total_losses_disc))}")

    torch.save(model.state_dict(), f"./models/model_{e%4}.ckpt")

    model.eval()
    total_losses_ae=[]
    total_losses_disc=[]
    img=None
    loss_disc=None

    for i,(description,video_path) in enumerate(test_loader):
        frame_idx=0
        text=tokenizer.encode(description)
        video_path=video_path[0]
        video_names=os.listdir(video_path)
        frame_length = len(video_names)
        video_path_sub=os.path.join(video_path,video_names[0])
        img=cv2.imread(video_path_sub)

        img=torch.Tensor(img).permute(2,0,1).unsqueeze(0).to(device)
        text=torch.LongTensor(text).to(device)
        valid_frame=torch.eye(2)[1].unsqueeze(0).repeat(batch_size, 1).to(device)

        text_token,vid_token,qloss=model.encode(text,img)

        current_total_losses_ae=[]
        current_total_losses_disc=[]

        frame_length = min(limit_frame_length,frame_length)
        video_names=video_names[:frame_length]
        # 已处理第0帧
        for frame_idx in range(len(video_names)):  # 当前帧状态valid_frame，1为有效，0为无效
            frame_idx+=1

            if frame_idx==frame_length:#无效帧
                valid_frame = torch.eye(2)[0].unsqueeze(0).repeat(batch_size, 1).to(device)
                img=torch.zeros_like(img)
            else:#有效帧
                video_path_sub=os.path.join(video_path,video_names[frame_idx])
                img = cv2.imread(video_path_sub)
                img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

            pred_img, vid_token, pred_vf=model.decode(text_token, vid_token)
            img=torch.resize_as_(img,pred_img)
            vf_loss=vf_criterion(valid_frame, pred_vf)

            loss_ae, log_dict_ae = loss_fn(qloss, img, pred_img, vf_loss, 0, i,
                                           last_layer=model.vqmodel.get_last_layer(), split="test")


            loss_disc, log_dict_disc = loss_fn(qloss, img, pred_img, vf_loss, 1, i,
                                               last_layer=model.vqmodel.get_last_layer(), split="test")

            current_total_losses_ae.append(loss_ae)
            current_total_losses_disc.append(loss_disc)

        total_losses_ae.append(sum(current_total_losses_ae) / len(current_total_losses_ae))
        total_losses_disc.append(sum(current_total_losses_disc) / len(current_total_losses_disc))


    total_losses_ae=sum(total_losses_ae)/len(total_losses_ae)

    total_losses_disc=sum(total_losses_disc)/len(total_losses_disc)

    schduler_ae.step(total_losses_ae)

    schduler_disc.step(total_losses_disc)

    print(F"\t\teval_loss_ae:{total_losses_ae},eval_loss_disc:{total_losses_disc}")

    if(total_losses_ae<min_loss):
        min_loss=total_losses_ae
        torch.save(model.state_dict(),"./models/best_model.ckpt")
