# encoding: utf-8

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle

train_path_dic = {'tiger':'./datalist/mytrain.txt',
                    'yak':'./datalist/yak_mytrain_aligned.txt',
                    'elephant':'./datalist/ele_train.txt',
                    'all':'./datalist/all_train_aligned.txt'
                    }

probe_path_dic = {'tiger':'./datalist/myval.txt',
                    'yak':'./datalist/yak_myval_aligned.txt',
                    'elephant':'./datalist/ele_val.txt',
                    'all':'./datalist/all_val_aligned.txt'
                }
class Animal(object):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'Animal-Seg-V3'

    def __init__(self, root='',species='tiger', verbose=True, pid_begin = 0, **kwargs):
        super(Animal, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = self.dataset_dir 
        self.query_dir = self.dataset_dir
        self.gallery_dir = self.train_dir

        self._check_before_run()
        self.pid_begin = pid_begin
        
        
        train = self._get_animal_train_val(train_path_dic[species],self.train_dir)
        query = self._get_animal_train_val(probe_path_dic[species],self.query_dir)
        gallery = train


        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)


    def get_imagedata_info(self, data):
        pids = []
        for _, pid,_,_ in data:
            pids += [pid]

        pids = set(pids)

        num_pids = len(pids)
        num_imgs = len(data)

        return num_pids, num_imgs,0,0
    
    
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _get_animal_train_val(self, label_path, dir_path):
        f = open(label_path)
        lines = f.readlines()

        dataset = []
 
        for line in lines:
            cls = line.rstrip('\n').split(' ')
            #print(cls)
            fn = cls.pop(0)

            if osp.isfile(osp.join(dir_path, fn)):
                dataset.append((osp.join(dir_path, fn), float(cls[0]),0, 0))
         
        return dataset