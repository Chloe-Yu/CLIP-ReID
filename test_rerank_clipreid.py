import os
from config import cfg
import argparse
import logging
import torch
from metric import evaluate_rerank
from re_ranking import re_ranking
from datasets.make_dataloader_clipreid import make_test_dataloader
from model.make_model_clipreid import make_model
from processor.processor_clipreid_stage2 import do_inference
from utils.logger import setup_logger
from tiger_eval import evaluate_tiger
import numpy as np

def extract_feature(model,dataloaders,linear_num):
    count = 0
    names = []
    

    for iter, (img, label, imgnames)  in enumerate(dataloaders):


        n, c, h, w = img.size()
        img = img.to(device) #input_img = Variable(img.cuda())
           
        ff = model(img, cam_label=None, view_label=None)
                    
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        if iter == 0:
            features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
            labels = torch.LongTensor(len(dataloaders.dataset))
            
        # start = iter*batchsize
        # end = min( (iter+1)*batchsize, len(dataloaders.dataset))

        features[ count:count+n, :] = ff
        labels[count:count+n] = label
        count += n
        names.extend(list(imgnames))
    
    return features,labels,names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False, rerank=True)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    dataloaders, num_classes = make_test_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=None, view_num = None)
    model.load_param(cfg.TEST.WEIGHT)


    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")


    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    
    linear_num = model.get_linear_num()
    
    with torch.no_grad():
        gallery_feature,gallery_label,gallery_names = extract_feature(model,dataloaders['gallery'],linear_num)
        query_feature, query_label,query_names = extract_feature(model,dataloaders['query'],linear_num)
        
    q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
    q_q_dist = np.dot(query_feature, np.transpose(query_feature))
    g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
    re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    
    remove_closest = True
    if cfg.DATASETS.SPECIES== 'yak' :
        remove_closest = False
    
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate_rerank(re_rank[i,:],query_label[i],gallery_label,remove_closest)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        
    CMC = CMC.float()
    CMC = CMC/len(query_label)
    
     
    if cfg.DATASETS.SPECIES == 'tiger':
        my_result = []
    
        for i in range(len(query_names)):
            tmp = {}
            image_name = query_names[i]

            index =np.argsort(re_rank[i, :])

            tmp['query_id'] = int(image_name.rstrip('.jpg'))
            p = 0
            gallery_tmp = []
            for j in index:
                if p == 0:
                    p += 1
                    continue
                current_name = gallery_names[j]
                gallery_tmp.append(int(current_name.rstrip('.jpg')))
            tmp['ans_ids'] = gallery_tmp
            my_result.append(tmp)
        
        metric = evaluate_tiger(my_result,'plain',path=False)
        logger.info("Validation Results ")
        logger.info(metric)
    else:
        mAP = ap / len(query_label)
        logger.info("Validation Results ")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, CMC[r - 1]))
      
    del query_feature
    del gallery_feature
    torch.cuda.empty_cache()
    
    

