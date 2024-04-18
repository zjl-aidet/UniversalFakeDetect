import os
import sys
import torch
import torch.nn
import argparse
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from models import get_model


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f','--file', default='examples_realfakedir')
parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14')
parser.add_argument('-m','--model_path', type=str, default='./checkpoints/clip_vitl14/model_epoch_best.pth')
parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')

opt = parser.parse_args()
model = get_model(opt.arch) 
state_dict = torch.load(opt.model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])
if(not opt.use_cpu):
  model.cuda()
model.eval()

# Transform
stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"
trans = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
])


img = trans(Image.open(opt.file).convert('RGB'))

with torch.no_grad():
    in_tens = img.unsqueeze(0)
    if(not opt.use_cpu):
    	in_tens = in_tens.cuda()
    prob = model(in_tens).sigmoid().item()

print('probability of being synthetic: {:.2f}%'.format(prob * 100))
