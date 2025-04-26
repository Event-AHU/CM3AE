import logging
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
# import matplotlib.pyplot as plt
# imagenet_mean = np.array([0.485, 0.456, 0.406])
# imagenet_std = np.array([0.229, 0.224, 0.225])
# def show_image(image, title=''):
#     # image is [H, W, 3]
#     assert image.shape[2] == 3
#     plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
#     plt.title(title, fontsize=16)
#     plt.axis('off')
#     return
class CsvDataset(Dataset):

    def __init__(self, input_filename, transforms, img_key, event_key, voxel_key, sep="\t"):
        logging.debug(f'Loading csv data from {input_filename}.')
        print("input_filename",input_filename)
        df = pd.read_csv(input_filename, sep=sep)
        
        self.images = df[img_key].tolist()
        self.events = df[event_key].tolist()
        self.voxels = df[voxel_key].tolist()
        # breakpoint()
        # self.target = df[target].tolist()

        self.transforms = transforms
        # '''防止图片损坏，使用其他图片替换'''
        # self.replace_idx = 0
        logging.debug('Done loading data.')

    def __len__(self):                                       

        return len(self.images)

    def __getitem__(self, idx):

        # try:
        #     images = self.transforms(Image.open(str(self.images[idx])))
        #     images_event = self.transforms(Image.open(str(self.images_event[idx])))
        #     target = self.target[idx]
        # except:
        #     images = self.transforms(Image.open(str(self.images[self.replace_idx])))
        #     images_event = self.transforms(Image.open(str(self.images_event[self.replace_idx])))
        #     target = self.target[idx]
        
        # print("images", images)
        # print("target", target)
        seed = torch.randint(0,100000,(1,)).item()
        # images = self.transforms(Image.open(str(self.images[idx])))
        # events = self.transforms(Image.open(str(self.events[idx])))
        image = Image.open(str(self.images[idx]))
        #print(f"Original image mode: {image.mode}")
        if image.mode != 'RGB':  # 'L' 表示单通道灰度图像
            image = image.convert('RGB')

        #if int(self.events[idx]) != 0:
        event = Image.open(str(self.events[idx]))
        torch.manual_seed(seed)
        images = self.transforms(image)
        torch.manual_seed(seed)
        events = self.transforms(event)

        
        # 将变换后的图像转换为 NumPy 数组并调整到正确的范围
        # images = images.permute(1, 2, 0).cpu().numpy()
        # images = (images * 255).astype(np.uint8)

        # events = events.permute(1, 2, 0).cpu().numpy()
        # events = (events * 255).astype(np.uint8)

        # # 使用 PIL 保存图片
        # Image.fromarray(images).save("transformed_image.jpg")
        # Image.fromarray(events).save("transformed_event.jpg")
        # show_image(transformed_image)
        # voxels = self.voxels[idx]
        # target = self.target[idx]
        data_path = str(self.voxels[idx]).replace('.JPEG','.npz')
        if data_path == '0':
            voxel = torch.zeros((10000, 16))
        else:
            data = np.load(data_path)
                
            voxel = torch.from_numpy(data['features']) #torch.Size([9339, 16]) torch.Size([5483, 16])  size需一致

            # Resize voxel to target size
            target_num = 10000  # Target number of rows for voxel
            voxel_num = voxel.size(0)
            
            if voxel_num == 0:
                # Handle the case where voxel_num is 0
                # print("Warning: voxel_num is 0, filling with default values.")
                voxel = torch.zeros(target_num, voxel.size(1))
            elif voxel_num < target_num:
                # If number of rows is less than target, repeat rows
                voxel = torch.cat([voxel] * (target_num // voxel_num), dim=0)
                # If there are remaining rows, repeat the first row
                remainder = target_num % voxel_num
                if remainder > 0:
                    voxel = torch.cat([voxel, voxel[:remainder, :]], dim=0)
            elif voxel_num > target_num:
                # If number of rows is greater than target, randomly sample rows
                indices = torch.randperm(voxel_num)[:target_num]
                voxel = voxel[indices]
        
        return images, events, voxel
        # return images, target
    
def csv_Dataset(args, input_filename, transforms):
    dataset = CsvDataset(
        input_filename,
        transforms,
        img_key=args.csv_img_key,
        event_key=args.csv_event_key,
        voxel_key = args.csv_voxel_key,
        sep=args.csv_separator)
    return dataset

            
