# Semantic Segmentation

##################### IMPORTS #####################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import PILToTensor, ToPILImage

from PIL import Image

import os 
import numpy as np
import matplotlib.pyplot as plt

##################### Dataset and DataLoader #####################

# CustomImageDataset Class
class CustomImageDataset(Dataset):
  def __init__(self, img_dir, img_labels_dir):
    self.img_labels_dir = img_labels_dir
    self.img_dir = img_dir
    self.curr_path = os.path.dirname(os.path.realpath(__file__))
    self.img_name_list = os.listdir(os.path.join(self.curr_path,img_dir))
    self.transform = PILToTensor() # Pil iamge transformation to tensor 

  def __len__(self):
    return len(self.img_name_list)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir,self.img_name_list[idx])
    label_path = os.path.join(self.img_labels_dir,self.img_name_list[idx])
    # image = read_image(img_path) # issues with read_image window error 10054
    # label = read_image(label_path)
    image = Image.open(img_path).crop((0,0,64,64)) # Crop images from 101,101 size to 64,64 size
    label = Image.open(label_path).crop((0,0,64,64))
    image = self.transform(image)/255 # Map to [0,1]
    label = self.transform(label)/65535 # I don't know why but 1 in the label is translated as 1111111111111111 in binary
    return image, label
      
dataset = CustomImageDataset('images', 'masks')
validation_split = 0.2 # 20% of the data set is used for validation
batch_size = 32
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

#####################  UNET Architecture #####################

class DoubleConv(nn.Module):
  """(convolution => [BN] => ReLU) * 2"""

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.double_conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)

class Down(nn.Module):
  """Downscaling with maxpool then double conv"""

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.maxpool_conv = nn.Sequential(
      nn.MaxPool2d(kernel_size=2),
      DoubleConv(in_channels, out_channels)
    )

  def forward(self, x):
    return self.maxpool_conv(x)

class Up(nn.Module):
  """Upscaling then double conv"""

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) #convTranspose2d Unsampling 
    self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
    x1 = self.up(x1)
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)

class Unet(nn.Module):

  def __init__(self):
    super(Unet, self).__init__()  
    self.input = DoubleConv(3, 16)
    self.down1 = Down(16, 32)
    self.down2 = Down(32, 64)
    self.down3 = Down(64, 128)
    self.down4 = Down(128, 256)  # As the image used are only 64,64 size, tha last down and the first up layers are disabled 
    self.up1 = Up(256, 128)
    self.up2 = Up(128, 64)
    self.up3 = Up(64, 32)
    self.up4 = Up(32, 16)
    self.out = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

  def forward(self, x):
    x1 = self.input(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    x = torch.sigmoid(self.out(x)) # sigmoid function used to get predictions between 0 and 1 
    return x 

##################### Some Functions #####################

def inference(tensor):
  tensor[tensor >= 0.5] = 1
  tensor[tensor < 0.5] = 0
  return tensor

# Calculate accuracy
def accuracy(prediction,label):
  nb_elt = torch.numel(prediction)
  acc = torch.square(inference(prediction) - label)
  return (1 - torch.sum(acc)/nb_elt).item()

# Calculate precision on label type 
def precision(prediction,label,label_type):
  pred_pos = torch.where(prediction == label_type)
  pred_pos_idx = [[pred_pos[0][i].item(),pred_pos[1][i].item()] for i in range(pred_pos[0].shape[0])]
  label_pos =  torch.where(label == label_type)
  label_pos_idx = [[label_pos[0][i].item(),label_pos[1][i].item()] for i in range(label_pos[0].shape[0])]
  true_pos = 0 
  for elt in pred_pos_idx:
    if elt in label_pos_idx :
      true_pos += 1
  return true_pos/pred_pos[0].shape[0]

#####################  Training Unet  #####################
torch_seed = 42
torch.manual_seed(torch_seed)
# network = Unet()
network = torch.load('./network2') # load network if previously trained 
optimizer = optim.Adam(network.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5) # Used to manage learning rate decrease in case of stagnation
criterion = nn.BCELoss() # Binary cross entropy loss 

nb_train = len(train_indices)
nb_val = len(val_indices)

nb_epoch = 0
train_loss_list = []
validation_loss_list = []
train_accuracy_list = []
validation_accuracy_list = []

for epoch in range(nb_epoch):
  stop = False
  total_train_loss = 0
  total_val_loss = 0
  total_train_accuracy = 0
  total_val_accuracy = 0
  
  network.train()
  for batch in train_loader:
    train_images, train_labels = batch
      
    optimizer.zero_grad()
    train_preds = network(train_images)
    
    train_loss = criterion(train_preds, train_labels)
    train_loss.backward()
    optimizer.step()
    train_accuracy = accuracy(train_preds, train_labels)*train_images.shape[0]
      
    total_train_loss += train_loss.item()
    total_train_accuracy += train_accuracy

  with torch.no_grad() :
    network.eval()
    for batch in validation_loader:
      val_images, val_labels = batch

      val_preds = network(val_images)
      val_loss = criterion(val_preds,val_labels)
      val_accuracy = accuracy(val_preds, val_labels)*val_images.shape[0]
      total_val_loss += val_loss.item()
      total_val_accuracy += val_accuracy
  
  avg_train_loss = total_train_loss/nb_train
  avg_val_loss = total_val_loss/nb_val
  avg_train_accuracy = total_train_accuracy/nb_train
  avg_val_accuracy = total_val_accuracy/nb_val
  train_loss_list.append(avg_train_loss)
  validation_loss_list.append(avg_val_loss)
  train_accuracy_list.append(avg_train_accuracy)
  validation_accuracy_list.append(avg_val_accuracy)

  # Learning Rate Decay 
  scheduler.step(avg_val_loss)

  # Early Stopping
  if len(validation_loss_list) >= 10 :
    last_ten = validation_loss_list[-10:]
    if min(last_ten) == last_ten[0] :
      stop = True
      print("Early stopping")

  # Save best model 
  if avg_val_loss == min(validation_loss_list) :
    torch.save(network,'./network2')

  print('epoch:', epoch+1, "average train loss:", avg_train_loss, "average train accuracy", avg_train_accuracy, 
        "average val loss", avg_val_loss, "average val accuracy", avg_val_accuracy)

  if stop == True :
    break


##################### Print Unet training results   #####################

# real_nb_epoch = len(train_loss_list)
# epochs = list(range(1,real_nb_epoch+1))

# plt.plot(epochs,train_loss_list, label='Training')
# plt.plot(epochs,validation_loss_list, label='validation')
# plt.legend()
# plt.title("Perte moyenne par epoch")
# plt.show()

# plt.plot(epochs,train_accuracy_list, label='Training')
# plt.plot(epochs,validation_accuracy_list, label='validation')
# plt.legend()
# plt.title("Accuracy moyenne par epoch")
# plt.show()

# network = torch.load('./network2')

# train_image, train_label = next(iter(train_loader))
# val_image, val_label = next(iter(validation_loader))
# trans = ToPILImage()
# network.eval()
# with torch.no_grad() :
#   train_pred = inference(network(train_image))
#   val_pred = inference(network(val_image))

# for i in range(6) : 
#   plt.imshow(trans(train_pred[i]).convert("RGB"))
#   plt.imshow(trans(train_label[i]).convert("RGB"))

#   plt.imshow(trans(val_pred[i]).convert("RGB"))
#   plt.imshow(trans(val_label[i]).convert("RGB"))

##################### Eval network on training and validation data   #####################

network = torch.load('./network2') # load network if previously trained 

nb_train = len(train_indices)
nb_val = len(val_indices)

nb_epoch = 0
train_loss_list = []
validation_loss_list = []
train_accuracy_list = []
validation_accuracy_list = []

total_train_loss = 0
total_val_loss = 0
total_train_accuracy = 0
total_val_accuracy = 0

with torch.no_grad() :
  network.eval()
  for idx,loader in enumerate([train_loader,validation_loader]) :
    for batch in loader:
      images, labels = batch
      preds = network(images)
      loss = criterion(preds,labels)
      preci = precision(preds, labels)*images.shape[0]
      total_val_loss += val_loss.item()
      total_val_accuracy += val_accuracy

avg_train_loss = total_train_loss/nb_train
avg_val_loss = total_val_loss/nb_val
avg_train_accuracy = total_train_accuracy/nb_train
avg_val_accuracy = total_val_accuracy/nb_val
train_loss_list.append(avg_train_loss)
validation_loss_list.append(avg_val_loss)
train_accuracy_list.append(avg_train_accuracy)
validation_accuracy_list.append(avg_val_accuracy)




