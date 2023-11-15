import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

import os
from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from .animal import Animal

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'animals':Animal
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    if cfg.DATASETS.NAMES == 'animals':
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, species = cfg.DATASETS.SPECIES, data_dir = cfg.DATASETS.DATA_DIR )
    else:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    
    
    train_set = ImageDataset(dataset.train, train_transforms)
    print("len of train",len(train_set))
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader_stage2 = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader_stage2 = DataLoader(
                train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader_stage2 = DataLoader(
            train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_stage1 = DataLoader(
        train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    )
    return train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes, cam_num, view_num


class dataset_test(Dataset):
    def __init__(self, root, label,unlabeled, transform=None, signal=' '):
        self._root = root
        self._label = label
        self._transform = transform
        self._unlabeled = unlabeled
        self._list_images(self._root, self._label, signal)

    def _list_images(self, root, image_names, signal):
        self.synsets = []
        self.synsets.append(root)
        self.items = []

        c = 0
        for line in image_names:
            cls = line.rstrip('\n').split(signal)
            image_name = cls.pop(0)
            
            if os.path.isfile(os.path.join(root, image_name)):
                if self._unlabeled:
                    self.items.append((os.path.join(root, image_name), image_name))
                else:
                    self.items.append((os.path.join(root, image_name), float(cls[0]), image_name))
            else:
                print(os.path.join(root, image_name))
            c += 1
        print('the total image is ', c)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # return img, img_name, (label)
        oriimg = Image.open(self.items[index][0])
        oriimg = oriimg.convert('RGB')
        if self._transform is not None:
            img = self._transform(oriimg)
             
        if self._unlabeled:
            return img, self.items[index][1]

        return img, self.items[index][1], self.items[index][2]




def load_gallery_probe_data(root, gallery_paths, probe_paths,
                            batch_size=32, num_workers=0,data_transforms=None):
    
    def get_label(label_path):
        f = open(label_path)
        lines = f.readlines()
        return lines
    
    
    gallery_list = []
    for i in gallery_paths:
        tmp = get_label(i)
        gallery_list = gallery_list + tmp

    probe_list = []
    for i in probe_paths:
        tmp = get_label(i)
        probe_list = probe_list + tmp
        

    # changed this to load for labeled data
    gallery_dataset = dataset_test(root, gallery_list, unlabeled=False, 
                                   transform=data_transforms)
    probe_dataset = dataset_test(root, probe_list, unlabeled=False,
                                 transform=data_transforms)

    gallery_iter = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    probe_iter = DataLoader(
        probe_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return gallery_iter, probe_iter


def make_test_dataloader(cfg):
    
    data_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    num_workers = cfg.DATALOADER.NUM_WORKERS
    gallery_path_dic = {'tiger':'./datalist/mytest.txt',
                    's_yak':'./datalist/yak_gallery_simple.txt',
                    'h_yak':'./datalist/yak_gallery_hard.txt',
                    'yak':'./datalist/yak_gallery_hard.txt',
                    'elephant':'./datalist/ele_new_test_gallery.txt',
                    'debug':'./datalist/debug_ele_train.txt'}
    probe_path_dic = {'tiger':'./datalist/mytest.txt',
                        's_yak':'./datalist/yak_probe_simple.txt',
                        'h_yak':'./datalist/yak_probe_hard.txt',
                        'yak':'./datalist/yak_probe_hard.txt',
                        'elephant':'./datalist/ele_new_test_probe.txt',
                        'debug':'./datalist/debug_ele_train.txt'}
    


    dataset_type = cfg.DATASETS.SPECIES
    gallery_paths = [gallery_path_dic[dataset_type], ]
    probe_paths = [probe_path_dic[dataset_type], ]
    root = cfg.DATASETS.TEST_ROOT_DIR
        
    gallery_iter, probe_iter = load_gallery_probe_data(
        root=root,
        gallery_paths=gallery_paths,
        probe_paths=probe_paths,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        num_workers=num_workers,
        data_transforms=data_transforms
    )
    dataloaders = {'gallery':gallery_iter,'query':probe_iter}
    
    dict_nclasses = {'yak':121,'tiger':107,'elephant':337,'all':565}
    num_classes = dict_nclasses[dataset_type]
    
    return  dataloaders,num_classes
