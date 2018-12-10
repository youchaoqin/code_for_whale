import os
import yaml
import numpy as np

if __name__ == '__main__':
    with open('/home/ycq/Desktop/humpback_whale_identification/'
              'data/all/train_set_statistics.yml', 'r') as f:
        train_set_statistics = yaml.load(f)
        total_num = float(train_set_statistics['total_examples'])
        num_of_per_cls = train_set_statistics['per_class_num']

    cls_to_idx_dict = {}
    with open('/home/ycq/Desktop/humpback_whale_identification/'
              'data/all/cls_name_to_index.txt', 'r') as f:
        for l in f.readlines():
            one_map = l.strip().split(':')
            one_cls = one_map[0].strip()
            one_idx = int(one_map[1].strip())
            cls_to_idx_dict[one_cls] = one_idx

    focal_loss_apha = np.zeros([len(cls_to_idx_dict)])

    for cls in num_of_per_cls:
        idx = cls_to_idx_dict[cls]
        focal_loss_apha[idx] = 1.0 / num_of_per_cls[cls]

    for i in range(focal_loss_apha.shape[0]):
        print('%d : %f'%(i, focal_loss_apha[i]))

    np.save('/home/ycq/Desktop/humpback_whale_identification/'
              'data/all/inverse_freq.npy', focal_loss_apha)