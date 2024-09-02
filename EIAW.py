import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from utils import dct_2d,idct_2d
import torch.nn.functional as F


class EIAW(object):
    def __init__(self, model,device, steps=10, use_monitor = True):
        self.steps = steps
        self.model = model
        self.device = device
        self.use_monitor = use_monitor

    def embed_watermark(self, images, watermark, p, mask):   
        images_dct = dct_2d(images)
        watermarked_dct = images_dct.clone().to(self.device)

        watermark_condition = watermark == 1 
        watermark_mask = watermark_condition.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        watermarked_dct[watermark_mask & mask.bool()] += (watermarked_dct[watermark_mask & mask.bool()] % p >= p / 2).float() * (p / 2)
        watermarked_dct[~watermark_mask & mask.bool()] += (watermarked_dct[~watermark_mask & mask.bool()] % p < p / 2).float() * (p / 2)

        return watermarked_dct

    def extract_watermark(self, watermarked_image, p, mask): 
        # watermarked_dct : [1, 3, 224, 224]
        watermarked_dct = dct_2d(watermarked_image)
        watermark = (watermarked_dct[0] % p < p / 2).sum(dim=0)
        watermark = (watermark >= 2).float()  
        watermark = watermark * mask[0,0,:,:]
        return watermark
    
    def monitor(self,images,labels):
        predict =  self.model(images)
        if np.argmax(predict.detach().cpu().numpy())!=int(labels.detach().cpu().numpy()):
            return True
        else:
            return False
        
    def __call__(self, inputs, labels, *args, **kwargs):

        adv_outputs = self.forward(inputs, labels, *args, **kwargs)

        return adv_outputs 

    def forward(self, images, labels,watermark, mask, p=2):

        images = images.clone().detach().to(self.device)
        watermarked_dct = self.embed_watermark(images, watermark, p, mask)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()
        adv_dct = watermarked_dct.clone().detach()

        for i in range(self.steps):
            adv_dct.requires_grad = True

            adv_image = idct_2d(adv_dct)

            outputs = self.model(adv_image)

            cost = loss(outputs, labels)

            grad = torch.autograd.grad(
                cost, adv_dct, retain_graph=False, create_graph=False)[0] * mask  

            adv_dct = adv_dct.detach() + p * grad.sign()

            adv_image = idct_2d(adv_dct)
            adv_image = torch.clamp(adv_image, min=0, max=1).detach()

            if self.use_monitor:
                if self.monitor(adv_image, labels):
                    print("success iter: ",i)
                    break
            else:
                continue

        return adv_image
    



