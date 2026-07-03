#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:51:27 2021

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:19:54 2020

@author: cds
"""


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

import numpy as np
import torchvision.transforms as transforms
import scipy.io
from torch.autograd import Variable

class mydataloader(Dataset):
    
    def __init__(self, csv_file, root_dir, training = True):
        self.names = pd.read_csv(csv_file)
        self.root_dir = root_dir  
        self.training = training
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        
        if self.training==True:
            file_name = os.path.join(self.root_dir,self.names['FileName'][idx])             
            data = scipy.io.loadmat(file_name)

            phs  = torch.tensor(data['phs' ]).unsqueeze(dim=0)
            msk  = torch.tensor(data['msk' ]).unsqueeze(dim=0)
            sus  = torch.tensor(data['susc']).unsqueeze(dim=0)
            
            phs=phs.float()
            sus=sus.float()
            msk=msk.float()
            
            phs=phs*msk
            sus=sus*msk

            return phs, msk, sus,file_name
        else:
            root_path=self.root_dir;
            phs_path = self.root_dir+"/phs/phs-"+str(self.names['Label'][idx])+".mat"
            msk_path = self.root_dir+"/msk/msk-"+str(self.names['Label'][idx])+".mat"
            sus_path = self.root_dir+"/cos/cos-"+str(self.names['Label'][idx])+".mat"
            file_name=self.names['FileName'][idx]
            
            phs = scipy.io.loadmat(phs_path)['phs']
            msk = scipy.io.loadmat(msk_path)['msk']
            sus = scipy.io.loadmat(sus_path)['cos']
            
            phs  = torch.tensor(phs).unsqueeze(dim=0)
            msk  = torch.tensor(msk).unsqueeze(dim=0)
            sus  = torch.tensor(sus).unsqueeze(dim=0)


             
            return phs, msk, sus,self.names['Label'][idx]

    
## Check the dataloader
#%%
    
# loader = mydataloader('csv_files/test.csv', 'Data/Testing_Data',training=False)
# trainloader = DataLoader(loader, batch_size = 1, shuffle=False, num_workers=1)
# print(len(trainloader))
    
# #%%
# for i, data in enumerate(trainloader): 
#     phs, msk, sus,file_name = data
#     print(i)
#     print(phs.shape)
#     print(msk.shape)
#     print(sus.shape)
    
#     print(file_name);
        
