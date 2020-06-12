import torch.utils.data as data
import torch
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import numpy as np

class ImageTensorFolder(data.Dataset):

    def __init__(self, img_path, tensor_path, img_fmt="npy", tns_fmt="npy", transform=None):
        self.img_fmt = img_fmt
        self.tns_fmt = tns_fmt
        self.img_paths = self.get_all_files(img_path, file_format=img_fmt)
        self.tensor_paths = self.get_all_files(tensor_path, file_format=tns_fmt)

        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def get_all_files(self, path, file_format="png"):
        filepaths = path + "/*.{}".format(file_format)
        files = glob(filepaths)
        print(files[0:10])
        return files

    def load_img(self, filepath, file_format="png"):
        if file_format in ["png", "jpg", "jpeg"]:
            img = Image.open(filepath)
            # Drop alpha channel
            if self.to_tensor(img).shape[0] == 4:
                img = self.to_tensor(img)[:3, :, :]
                img = self.to_pil(img)
        elif file_format == "npy":
            img = np.load(filepath)
            #cifar10_mean = [0.4914, 0.4822, 0.4466]
            #cifar10_std = [0.247, 0.243, 0.261]
            img = np.uint8(255 * img)
            img = self.to_pil(img)
        elif file_format == "pt":
            img = torch.load(filepath)
        else:
            print("Unknown format")
            exit()
        return img

    def load_tensor(self, filepath, file_format="png"):
        if file_format == "png":
            tensor = Image.open(filepath)
            # Drop alpha channel
            if self.to_tensor(tensor).shape[0] == 4:
                tensor = self.to_tensor(tensor)[:3, :, :]
        elif file_format == "npy":
            tensor = np.load(filepath)
            tensor = self.to_tensor(tensor)
        elif file_format == "pt":
            tensor = torch.load(filepath)
            tensor.requires_grad = False
        return tensor

    def __getitem__(self, index):
        img = self.load_img(self.img_paths[index], file_format=self.img_fmt)
        intermed_rep = self.load_tensor(self.tensor_paths[index], file_format=self.tns_fmt)
        if self.transform is not None:
            img = self.transform(img)

        return img, intermed_rep

    def __len__(self):
        return len(self.img_paths)
