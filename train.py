import torch
from matplotlib import pyplot
from torch import nn
import torchvision.transforms as transforms 
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from torch.autograd import Variable
import numpy as np



class Generator(nn.Module):
    def __init__(self, latent_dim):
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
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        x = self.g(x)
        return x


class Discriminator(nn.Module):
    def __init__(self,img_dim):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.img_dim = img_dim
        self.d = nn.Sequential(
            nn.Linear(self.img_dim+10, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        x = self.d(x)
        return x

       
        
   

   
def generate_real_samples(transform,batch_size):
    train_data = datasets.MNIST(
    root='../input/data',
    train=True,
    download=True,
    transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)    
    return train_loader



 
def generate_latent_points(latent_dim, n):
    x_input = torch.randn(latent_dim * n).reshape(n, latent_dim)
    return x_input
 
def generate_fake_samples(generator, latent_dim, n):
    x_input = generate_latent_points(latent_dim, n)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, x_input.size(0))))
    X = generator(x_input,fake_labels)



    return X, fake_labels, torch.zeros((n, 1))




# save generated images and generator model in each 10 epochs 
def summarize_performance(epoch, generator,discriminator, latent_dim, n=100):
    
    if not os.path.exists('./img'):
        os.makedirs('./img')
    if not os.path.exists('./models'):
        os.makedirs('./models')


    img, labels,y = generate_fake_samples(generator, latent_dim, n)

    img = img.view(-1,1,28,28)
    for i in range(n):
        if not os.path.exists('./img/'+str(epoch)):
            os.makedirs('./img/'+str(epoch))

        save_image(img[i],'./img/'+str(epoch)+'/img'+str(i)+'_label_'+str(labels[i].item())+'.jpg')
    torch.save(generator.state_dict(), './models/gmodel'+'_'+str(epoch)+'.pt')
    torch.save(discriminator.state_dict(), './models/dmodel'+'_'+str(epoch)+'.pt')

    
 


def train(g_model, d_model, transform, latent_dim, n_epochs=100, n_batch=128, n_eval=10):    

    optimizer_gan = torch.optim.Adam(g_model.parameters(), lr=0.0001)
    optimizer_d = torch.optim.Adam(d_model.parameters(), lr=0.0001)   

    half_batch = int(n_batch / 2)
    criterion = nn.BCELoss()
    real_data_loader = generate_real_samples(transform, half_batch)



    for i in range(n_epochs):
        
        loss_discriminator = 0
        loss_generator = 0
        
        size_dicriminator = 0
        size_generator = 0
        
        for data in real_data_loader:

            # Discriminator
            x,label = data
            y = torch.ones((x.size(0),1))
            x2,lab2, y2 = generate_fake_samples(g_model, latent_dim, half_batch)   
            
            loss1 = criterion(d_model(x.view(-1, 28*28),label), y)
            loss2 =  criterion(d_model(x2,lab2), y2)            
            loss_d = loss1+loss2

            d_model.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            loss_discriminator += loss_d
            size_dicriminator += x.size(0)+x2.size(0) 

            # Generator
            x_gan = generate_latent_points(latent_dim, n_batch)
            fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, n_batch)))

            y_gan = torch.ones((n_batch, 1))
            
            gan_model = d_model(g_model(x_gan,fake_labels),fake_labels)
            loss_g = criterion(gan_model, y_gan)
            g_model.zero_grad()
            loss_g.backward()
            optimizer_gan.step()
            loss_generator += loss_g

            size_generator += x_gan.size(0)


        


        # print results and save model and images
        if (i+1) % n_eval == 0:
            print("epoch = ", i," discriminator loss = ", loss_discriminator.item()/size_dicriminator, "  generator loss = ", loss_generator.item()/size_generator)
            print()
            summarize_performance(i+1, g_model,d_model, latent_dim)





if __name__ == '__main__':  
    latent_dim = 128
    img_dim = 28*28
    batch_size = 16

    generator = Generator(latent_dim)
    discriminator = Discriminator(img_dim)

    transform = transforms.Compose([
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, ), (0.5, ))])
 

    train(generator, discriminator , transform, latent_dim)

