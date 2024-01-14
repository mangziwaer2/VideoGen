import cv2
import torch
from torch import nn

from modules.completion_model import CompletionModel
from modules.tokenizer import Tokenizer
from dataset import Dataset
from modules.VQGAN.modules.losses import vqperceptual

epoch=100
batch_size=1
lr=1e-3

dataset_path_root="datasets/vid"
dictionary_path="models/dictionary.gensim"

train_dataset=Dataset(dataset_path_root=dataset_path_root,train=True)
test_dataset=Dataset(dataset_path_root=dataset_path_root,train=False)

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

tokenizer=Tokenizer(dictionary_path=dictionary_path)

model=CompletionModel(tokenizer,embed_dim=512,max_len=2048).to(device)

mf_criterion=nn.CrossEntropyLoss()
loss_fn=vqperceptual.VQLPIPSWithDiscriminator(device=device)

optimizer_idx=-1
step_update=False

optimizer_ae = torch.optim.Adam(list(model.vqmodel.encoder.parameters())+
                                  list(model.vqmodel.decoder.parameters())+
                                  list(model.vqmodel.quantize.parameters())+
                                  list(model.vqmodel.quant_conv.parameters())+
                                  list(model.vqmodel.post_quant_conv.parameters())+
                                  list(model.text_encoder.embedding.parameters())+
                                  list(model.text_encoder.block.parameters())+
                                  list(model.vid_model.parameters())+
                                  list(model.next_state_model.parameters()),
                          lr=lr, betas=(0.5, 0.9))

optimizer_disc = torch.optim.Adam(loss_fn.discriminator.parameters(),
                            lr=lr, betas=(0.5, 0.9))

schduler_ae=scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ae, mode='min', factor=0.1, patience=2)
schduler_disc=scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_disc, mode='min', factor=0.1, patience=2)

for e in range(epoch):
    optimizer_idx+=1
    optimizer_idx%=2
    if e>=epoch/2:
        step_update=False

    loss_ave=0
    model.train()
    for i,(description,video_path) in enumerate(train_loader):
        text=tokenizer.encode(description)
        cap=cv2.VideoCapture(video_path[0])
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        success, img = cap.read()
        img=torch.Tensor(img).permute(2,0,1).unsqueeze(0).to(device)
        text=torch.LongTensor(text).to(device)
        mf=torch.eye(2)[1].unsqueeze(0).repeat(batch_size,1).to(device)

        text_token,vid_token,qloss=model.encode(text,img)

        total_losses=[]
        frame_idx=0
        while True: #计算处理视频最后一帧时mf为0
            frame_idx+=1
            if frame_idx==frame_length-1:
                mf = torch.eye(2)[0].unsqueeze(0).repeat(batch_size,1).to(device)
                print("最后一帧")
            else:
                print("帧",frame_idx,"/",frame_length,vid_token.shape)

            success,img=cap.read()

            if not success:
                print("结束")
                break
            img=cv2.resize(img,(64,128))
            img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
            pred_img,vid_token,pred_mf=model.decode(text_token,vid_token)

            mf_loss=mf_criterion(mf,pred_mf)

            if optimizer_idx == 0:
                # autoencode
                loss, log_dict_ae = loss_fn(qloss, img, pred_img,mf_loss, optimizer_idx, i,
                                                last_layer=model.vqmodel.get_last_layer(), split="train")

            if optimizer_idx == 1:
                # discriminator
                loss, log_dict_disc = loss_fn(qloss, img, pred_img, mf_loss,optimizer_idx, i,
                                                last_layer=model.vqmodel.get_last_layer(), split="train")

            total_losses.append(loss)
            print("loss:",loss)

            if step_update:

                if optimizer_idx==0:
                    optimizer_ae.zero_grad()
                    loss.backward()
                    optimizer_ae.step()

                if optimizer_idx==1:
                    optimizer_disc.zero_grad()
                    loss.backward()
                    optimizer_disc.step()

        if not step_update:
            loss=sum(total_losses)/len(total_losses)
            loss_ave+=loss.item()
            if optimizer_idx == 0:
                optimizer_ae.zero_grad()
                loss.backward()
                optimizer_ae.step()

            if optimizer_idx == 1:
                optimizer_disc.zero_grad()
                loss.backward()
                optimizer_disc.step()

    print(F"{e}/{epoch},train_loss:{loss_ave/len(train_loader)}")

    torch.save(model.state_dict(), f"./model_{e}.ckpt")

    model.eval()
    loss_ave=0
    for i,(description,video_path) in enumerate(test_loader):
        text=tokenizer.encode(description)
        cap=cv2.VideoCapture(video_path[0])
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        success, img = cap.read()
        img=torch.Tensor(img).permute(2,0,1).unsqueeze(0).to(device)
        text=torch.LongTensor(text).to(device)
        mf=torch.eye(2)[1].unsqueeze(0).repeat(batch_size,1).to(device)

        text_token,vid_token,qloss=model.encode(text,img)

        total_losses=[]
        frame_idx=0
        while True: #计算处理视频最后一帧时mf为0
            frame_idx+=1
            if frame_idx==frame_length-1:
                mf = torch.eye(2)[0].unsqueeze(0).repeat(batch_size,1).to(device)

            success,img=cap.read()
            if not success:
                break

            img=cv2.resize(img,(64,128))
            img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
            pred_img,vid_token,pred_mf=model.decode(text_token,vid_token)

            mf_loss=mf_criterion(mf,pred_mf)

            if optimizer_idx == 0:
                # autoencode
                loss, log_dict_ae = loss_fn(qloss, img, pred_img,mf_loss, optimizer_idx, i,
                                                last_layer=model.vqmodel.get_last_layer(), split="train")

            if optimizer_idx == 1:
                # discriminator
                loss, log_dict_disc = loss_fn(qloss, img, pred_img, mf_loss,optimizer_idx, i,
                                                last_layer=model.vqmodel.get_last_layer(), split="train")

            total_losses.append(loss)

        loss_ave+=sum(total_losses)/len(total_losses)

    loss_ave/=len(test_loader)

    if optimizer_idx==0:
        schduler_ae.step(loss_ave)


    if optimizer_idx==1:
        schduler_disc.step(loss_ave)

    print(F"{e}/{epoch},eval_loss:{loss_ave}")
