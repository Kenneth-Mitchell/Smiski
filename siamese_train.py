import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
import numpy as np

class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 4),
            nn.ReLU(inplace=True),
        )
        
        # Calculate the input size for the first fully connected layer
        self._initialize_fc()

    def _initialize_fc(self):
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, 100, 100)
            output = self.conv(sample_input)
            output_size = output.view(output.size(0), -1).size(1)
        
        self.fc = nn.Sequential(
            nn.Linear(output_size, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024)
        )

    def forward_once(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# Custom Dataset
class SmiskiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                images.append((img_path, self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img1_path, label1 = self.images[idx]
        
        # Randomly choose the second image
        if random.random() > 0.5:  # 50% chance of same class
            same_class_images = [img for img, label in self.images if label == label1 and img != img1_path]
            img2_path = random.choice(same_class_images) if same_class_images else img1_path
            label2 = label1
        else:
            different_class_images = [img for img, label in self.images if label != label1]
            img2_path = random.choice(different_class_images)
            label2 = img2_path


        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.from_numpy(np.array([int(label1 != label2)], dtype=np.float32))

# Data transforms
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

# Create dataset and dataloader
dataset = SmiskiDataset(root_dir='siamese_data', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the network
net = SiameseNetwork()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters())

# Training loop
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (img1, img2, label) in enumerate(dataloader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        optimizer.zero_grad()
        output1, output2 = net(img1, img2)
        loss = criterion(torch.pairwise_distance(output1, output2), label.squeeze())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1 == 0:  # Print every mini-batch
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / (i + 1):.3f}')

        if i % 10 == 9:  # Print every 10 mini-batches
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

print('Finished Training')

# Save the model
torch.save(net.state_dict(), 'siamese_net.pth')