import torch
from matplotlib import pyplot
from torch import nn
import torchvision.transforms as transforms 
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import torch.nn.functional as F
import shutil
from PIL import Image,ImageOps
import torchvision.transforms.functional as TF
import argparse



class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.latent_dim = latent_dim
            
        self.g = nn.Sequential(nn.Linear(self.latent_dim+10, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28*28),
            nn.Tanh()
            )
        
    def forward(self, z, labels):
        labels = torch.tensor(labels)
        c = self.label_emb(labels)
        c = c.view(1,10)        
        x = torch.cat([z, c], 1)
        x = self.g(x)
        return x







def image_generator(output_folder,g_model,sum_num,latent_dim=128):
    x_input = torch.randn(latent_dim ).reshape(1, latent_dim)
    num_list = [int(x) for x in str(sum_num)]
    
    img = torch.zeros((len(num_list),1,28,28))
    count = 0
    for i in num_list:
        
        
        img1 = g_model(x_input,i)

        img[count] = img1.view(-1,1,28,28)
    
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        count +=1
    save_image(img,output_folder+'/img_'+str(sum_num)+'_.jpg')

    return img



if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--input_num', default=123 ,help='Alpha')
    #inputs
    #img_label must be between 0~9
    args = parser.parse_args()
    input_num = int(args.input_num)
   




    # input number
    # generator model
    latent_dim = 128
    g_model = Generator(latent_dim)
    g_model.load_state_dict(torch.load('./models/gmodel_50.pt'))
    
 

   

    # output images will be saves in result folder
    output_folder = "./result"    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 
        
    image_generator(output_folder,g_model,input_num,latent_dim)
    

