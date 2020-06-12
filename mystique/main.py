import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import SubsetRandomSampler
from tensorboardX import SummaryWriter

from AEs import Autoencoder, MinimalDecoder
from image_folder import ImageTensorFolder
import os
import numpy as np
from shutil import copytree, copy2
from glob import glob
from generate_ir import get_client_model

random_seed = 100
torch.manual_seed(random_seed)
np.random.seed(random_seed)

def apply_transform(batch_size, image_data_dir, tensor_data_dir):
    """
    """
    #cifar10_mean = [0.4914, 0.4822, 0.4466]
    #cifar10_std = [0.247, 0.243, 0.261]
    train_split = 0.9

    #trainTransform = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.Normalize(cifar10_mean,
    #                                                         cifar10_std)])
    #dataset = ImageTensorFolder(img_path=image_data_dir, tensor_path=tensor_data_dir,
    #                             transform=trainTransform)
    trainTransform = transforms.Compose([transforms.ToTensor(),
                                         ])
    dataset = ImageTensorFolder(img_path=image_data_dir, tensor_path=tensor_data_dir,
                                 img_fmt="npy", tns_fmt="npy", transform=trainTransform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    trainloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False, num_workers=4,
                                              sampler=train_sampler)

    testloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=4,
                                             sampler=test_sampler)


    return trainloader, testloader

def denormalize(img, dataset="imagenet"):
    """
    data is normalized with mu and sigma, this function puts it back
    """
    if dataset == "cifar10":
        c_std = [0.247, 0.243, 0.261]
        c_mean = [0.4914, 0.4822, 0.4466]
    elif dataset == "imagenet":
        c_std = [0.229, 0.224, 0.225]
        c_mean = [0.485, 0.456, 0.406]
    for i in [0, 1, 2]:
        img[i] = img[i] * c_std[i] + c_mean[i]
    return img

def save_images(input_imgs, output_imgs, epoch, path, offset=0, batch_size=64):
    """
    """
    input_prefix = "inp_"
    output_prefix = "out_"
    out_folder = "{}/{}".format(path, epoch)
    out_folder = os.path.abspath(out_folder)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    for img_idx in range(input_imgs.shape[0]):
        inp_img_path = "{}/{}{}.jpg".format(out_folder, input_prefix, offset * batch_size + img_idx)
        out_img_path = "{}/{}{}.jpg".format(out_folder, output_prefix, offset * batch_size + img_idx)
        #inp_img = denormalize(input_imgs[img_idx])
        #out_img = denormalize(output_imgs[img_idx])
        save_image(input_imgs[img_idx], inp_img_path)
        save_image(output_imgs[img_idx], out_img_path)

def copy_source_code(path):
    if not os.path.isdir(path):
        os.makedirs(path)

    for file_ in glob(r'./*.py'):
        copy2(file_, path)
    copytree("clients/", path + "clients/")

def main():
    """
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device as {}".format(device))
    num_epochs = 500
    batch_size = 32
    train_output_freq = 10
    test_output_freq = 50

    architecture = "AE-UTKFace-race-attribute-1000.0-0.1-begin"
    output_path = "./output/{}".format(architecture)
    train_output_path = "{}/train".format(output_path)
    test_output_path = "{}/test".format(output_path)
    tensorboard_path = "{}/tensorboard/".format(output_path)
    source_code_path = "{}/sourcecode/".format(output_path)
    model_path = "{}/model.pt".format(output_path)

    writer = SummaryWriter(logdir=tensorboard_path)

    decoder = Autoencoder(input_nc=3, output_nc=3).to(device)
    #decoder = MinimalDecoder(input_nc=64, output_nc=3, input_dim=112, output_dim=224).to(device)
    torch.save(decoder.state_dict(), model_path)
    decoder.load_state_dict(torch.load(model_path))
    copy_source_code(source_code_path)

    distance = nn.MSELoss()

    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    #client_model = get_client_model().to(device)
    #for param in client_model.parameters():
    #    param.requires_grad = False

    image_data_dir = "./data/noPeek/1000.0-0.1-UTKFace-race-attribute/input"
    tensor_data_dir = "./data/noPeek/1000.0-0.1-UTKFace-race-attribute/output"
    trainloader, testloader = apply_transform(batch_size, image_data_dir, tensor_data_dir)

    round_ = 0

    for epoch in range(round_ * num_epochs, (round_ + 1) * num_epochs):
        for num, data in enumerate(trainloader, 1):
            img, ir = data
            img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
            img, ir = Variable(img).to(device), Variable(ir).to(device)

            #ir = client_model(img)
            output = decoder(ir)

            reconstruction_loss = distance(output, img)
            train_loss = reconstruction_loss

            writer.add_scalar('loss/train', train_loss.item(), len(trainloader) * epoch + num)
            writer.add_scalar('loss/train_loss/reconstruction', reconstruction_loss.item(), len(trainloader) * epoch + num)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        if (epoch + 1) % train_output_freq == 0:
            save_images(img, output, epoch, train_output_path, offset=0, batch_size=batch_size)

        for num, data in enumerate(testloader, 1):
            img, ir = data
            img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
            img, ir = Variable(img).to(device), Variable(ir).to(device)

            #ir = client_model(img)
            output = decoder(ir)

            reconstruction_loss = distance(output, img)
            test_loss = reconstruction_loss

            writer.add_scalar('loss/test', test_loss.item(), len(testloader) * epoch + num)
            writer.add_scalar('loss/test_loss/reconstruction', reconstruction_loss.item(), len(testloader) * epoch + num)

        if (epoch + 1) % test_output_freq == 0:
            for num, data in enumerate(testloader):
                img, ir = data
                img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
                img, ir = Variable(img).to(device), Variable(ir).to(device)

                #ir = client_model(img)
                output_imgs = decoder(ir)

                save_images(img, output_imgs, epoch, test_output_path, offset=num, batch_size=batch_size)

        for name, param in decoder.named_parameters():
            writer.add_histogram("params/{}".format(name), param.clone().cpu().data.numpy(), epoch)

        torch.save(decoder.state_dict(), model_path)
        print("epoch [{}/{}], train_loss {:.4f}, test_loss {:.4f}".format(epoch + 1,
                                                  num_epochs, train_loss.item(), test_loss.item()))

    writer.close()

if __name__ == '__main__':
    main()
