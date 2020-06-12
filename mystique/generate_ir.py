"""
Generates Intermediate representation for a given dataset and model.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torchvision.utils import save_image

import uuid
from image_folder import ImageTensorFolder 
import os

counter = 0

def save_intermed_reps(input_imgs, intermed_reps, arch):
    """
    Save intermediate representations and corresponding images
    TODO: Find best format to save intermediate representations
    """
    img_folder = "./data/{}/".format(arch)
    intermed_reps_folder = "./data/{}/".format(arch)
    img_folder = os.path.abspath(img_folder)
    intermed_reps_folder = os.path.abspath(intermed_reps_folder)
    global counter

    if not os.path.isdir(intermed_reps_folder):
        os.makedirs(intermed_reps_folder)
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    for batch_idx in range(input_imgs.shape[0]):
        # file_id = uuid.uuid4().hex
        file_id = counter
        inp_img_path = "{}/{}.jpg".format(img_folder, file_id)
        out_tensor_path = "{}/{}.pt".format(intermed_reps_folder, file_id)
        save_image(input_imgs[batch_idx], inp_img_path)
        torch.save(intermed_reps[batch_idx].cpu(), out_tensor_path)
        counter += 1


def get_client_model():
    # TODO: Make it modular, structured and parametrized

    resnet18 = models.resnet18(pretrained=True)
    model = resnet18
    resnet_split = 9 
    client_model = nn.Sequential(*list(model.children())[:resnet_split])
    return client_model


def apply_transform(batch_size):
    num_workers = 4
    data_dir = "/tmp/imagenet/"
    val_dir = os.path.join(data_dir, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    img_loader = torch.utils.data.DataLoader(
         datasets.ImageFolder(val_dir, transforms.Compose([
                                                        transforms.Resize(256),
                                                        transforms.CenterCrop(224),
                                                        transforms.ToTensor(),
                                                        normalize])),
                             batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return img_loader


def main():
    """
    """
    num_epochs = 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    architecture = "resnet18-4"
    batch_size = 128

    client_model = get_client_model()
    decoder = client_model.to(device)

    distance = nn.MSELoss()

    optimizer = torch.optim.Adam(client_model.parameters(), weight_decay=1e-5)

    testloader = apply_transform(batch_size)

    print("client model is -------")
    print(client_model)

    for epoch in range(num_epochs):
        for data in testloader:
            img, _ = data
            img = Variable(img).to(device)

            out = client_model(img)

            save_intermed_reps(img, out, architecture)


if __name__ == '__main__':
    print("Starting process")
    main()
