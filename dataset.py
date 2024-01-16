import os

import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self,dataset_path_root,train=False):
        super(Dataset, self).__init__()

        descriptions=[]
        video_paths=[]

        video_names=os.listdir(dataset_path_root)
        total_video_num=len(video_names)
        train_pos=int(total_video_num*0.8)

        start=0
        end=0
        if(train):
            start=0
            end=train_pos
        else:
            start=train_pos
            end=total_video_num

        for video_name in video_names[start:end]:
            data_path=os.path.join(dataset_path_root,video_name)

            for data in os.listdir(data_path):
                d_path=os.path.join(data_path,data)
                if(data.endswith(".txt")):
                    with open(d_path,"r",encoding="utf-8") as f:
                        content=f.read()
                        descriptions.append(content)
                if(data=="frames"):
                    video_paths.append(d_path)

        self.descriptions=descriptions
        self.video_paths=video_paths

    def __getitem__(self, idx):
        description=self.descriptions[idx]
        video_path=self.video_paths[idx]

        return description,video_path

    def __len__(self):
        return len(self.descriptions)
