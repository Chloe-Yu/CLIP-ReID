## CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels [[pdf]](https://arxiv.org/pdf/2211.13977.pdf)
 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-reid-exploiting-vision-language-model/person-re-identification-on-msmt17)](https://paperswithcode.com/sota/person-re-identification-on-msmt17?p=clip-reid-exploiting-vision-language-model)

### Pipeline

![framework](fig/method.png)

### Installation

```
conda create -n clipreid python=3.8
conda activate clipreid
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install yacs
pip install timm
pip install scikit-image
pip install tqdm
pip install ftfy
pip install regex
```
Y Yu: I ran into errors with the above setup, which are fixed by the following.

```
conda install scikit-learn
python -m pip install charset-normalizer==2.1.0
```


### Prepare Dataset

Download the datasets, and then unzip them to `your_dataset_dir`.

### Training


For example, to run ViT-based CLIP-ReID for Yak, you need to modify the bottom of configs/yak/vit_clipreid.yml to

```
DATASETS:
  NAMES: ('animals')
  ROOT_DIR: ('your_dataset_dir') # eg.'../data/'
  TEST_ROOT_DIR: ('your_dataset_dir/yak_test_seg_isnet_pp')
  SPECIES: ('yak')
OUTPUT_DIR: 'your_output_dir' # eg.'./output/yak/vit'
```

then run 

```
CUDA_VISIBLE_DEVICES=0 python train_clipreid.py --config_file configs/yak/cnn_clipreid.yml
```

To run CNN-based (Resnet50) CLIP-ReID for ATRW, you need to modify the bottom of configs/tiger/cnn_clipreid.yml to


```
DATASETS:
  NAMES: ('animals')
  ROOT_DIR: ('../data/') #your_dataset_dir
  TEST_ROOT_DIR: ('../data/tiger/tiger_test_isnet_seg') #your_dataset_dir
  SPECIES: ('tiger')
OUTPUT_DIR: './output/tiger/cnn' # your_output_dir
```
then run,

```
CUDA_VISIBLE_DEVICES=0 python train_clipreid.py --config_file configs/tiger/cnn_clipreid.yml
```

### Evaluation

For example, to test CNN-based CLIP-ReID for elephant

```
CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --config_file configs/elephant/cnn_clipreid.yml TEST.WEIGHT 'your_trained_checkpoints_path/RN50_2.pth'
```
To enable re-ranking:
```
CUDA_VISIBLE_DEVICES=0 python test_rerank_clipreid.py --config_file configs/tiger/cnn_clipreid.yml TEST.WEIGHT './output/tiger/cnn/RN50_2.pth'
```

### Notes

Their method has two variations, which are CNN-based CLIP-ReID and Vit-based CLIP-ReID, using Resnet50 and Vit respectively for image encoder backbone. And in their experiments, Vit-based CLIP-ReID performed better. Since our method uses CNN-based backbone, CNN-based CLIP-ReID might be a fairer comparison, and experiments on Vit-based CLIP-ReID are less important for now.

In our Table 2, re-ranking is only enabled for the tiger dataset. So for evaluation, it's best to use test_rerank_clipreid.py for tiger and test_clipreid.py for elephant and yak.

configs are added in configs/elephant, configs/yak, and configs/tiger. ROOT_DIR and TEST_ROOT_DIR at end of the file needs to be modified. Other parameters can of course be modified as well.

When the DIRs are set correctly, the basic experiments to run are:

```
CUDA_VISIBLE_DEVICES=0 python train_clipreid.py --config_file configs/tiger/cnn_clipreid.yml
CUDA_VISIBLE_DEVICES=0 python test_rerank_clipreid.py --config_file configs/tiger/cnn_clipreid.yml TEST.WEIGHT './output/tiger/cnn/RN50_2.pth'

CUDA_VISIBLE_DEVICES=0 python train_clipreid.py --config_file configs/yak/cnn_clipreid.yml
CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --config_file configs/yak/cnn_clipreid.yml TEST.WEIGHT './output/yak/cnn/RN50_2.pth'

CUDA_VISIBLE_DEVICES=0 python train_clipreid.py --config_file configs/elephant/cnn_clipreid.yml
CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --config_file configs/elephant/cnn_clipreid.yml TEST.WEIGHT './output/elephant/cnn/RN50_2.pth'

```


### Acknowledgement

Codebase from [TransReID](https://github.com/damo-cv/TransReID), [CLIP](https://github.com/openai/CLIP), and [CoOp](https://github.com/KaiyangZhou/CoOp).

The veri776 viewpoint label is from https://github.com/Zhongdao/VehicleReIDKeyPointData.


### Citation

If you use this code for your research, please cite

```
@article{li2022clip,
  title={CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels},
  author={Li, Siyuan and Sun, Li and Li, Qingli},
  journal={arXiv preprint arXiv:2211.13977},
  year={2022}
}
```

