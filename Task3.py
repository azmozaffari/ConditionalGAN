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


class LeNet(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

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







def select_mnist_img(label):
    transform = transforms.Compose([
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, ), (0.5, ))])


    test_data = datasets.MNIST(
                    root='../input/data',
                    train=False,
                    download=True,
                    transform=transform
    )

    batch_size = 64
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)    

    data = next(iter(test_loader))
    img,lab = data
    return(img[torch.where(lab == label)][0])








def image_summation(output_folder,g_model,c_model,img,num,latent_dim):
    
    img = img.view(1,1,28,28)
    label = torch.argmax(c_model(img))
    sum_num = label.item() + num
    print("the label of given image is =", label.item(), " the generated number should be "+str(num)+"+"+str(label.item())+"="+str(sum_num))

    img = image_generator(output_folder,g_model,sum_num,latent_dim)
    return img








if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--img_add', default='./test/img.jpg' ,help='path to img')
    parser.add_argument('--input_num', default=123 ,help='Alpha')
    #inputs
    #img_label must be between 0~9
    args = parser.parse_args()
    img_add = args.img_add     
    input_num = int(args.input_num)
   







    # input image : select the the image randomly with the same img_label from the test mnist dataset
    
    # input_img = select_mnist_img(img_label)   
    # if not os.path.exists('./test/'):
    #     os.makedirs('./test/') 
    # save_image(input_img,'./test/img.jpg')

    img = Image.open(img_add)
    gray_image = ImageOps.grayscale(img)
    input_img = TF.to_tensor(gray_image)



    # input number

    # generator model
    latent_dim = 128
    g_model = Generator(latent_dim)
    g_model.load_state_dict(torch.load('./models/gmodel_50.pt'))
    
 

    c_model = LeNet()
    c_model.load_state_dict(torch.load('./models/cmodel_50.pt'))

    # output images will be saves in result folder
    output_folder = "./result"    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 

    image_summation(output_folder,g_model,c_model,input_img,input_num,latent_dim)
    

