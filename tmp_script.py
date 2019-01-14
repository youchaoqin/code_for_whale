import os
import cv2

if __name__ == "__main__":
    # scripts to copy val_exp images and train_exp images to a folder
    dst_folser = '/home/westwell/Desktop/dolores_storage/' \
                 'humpback_whale_identification/data/all/exp_train_val_images'
    with open('/home/westwell/Desktop/dolores_storage/humpback_whale_identification/'
              'data/all/train_exp.csv', 'r') as f:
        for l in f.readlines():
            if "Image" in l:
                continue
            anno_temp = l.strip().split(',')
            im_name = anno_temp[0].strip()
            tmp_image = cv2.imread(
                os.path.join('/home/westwell/Desktop/dolores_storage/humpback_whale_identification/'
                             'data/all/train', im_name))
            cv2.imwrite(os.path.join(dst_folser, im_name), tmp_image)

    with open('/home/westwell/Desktop/dolores_storage/humpback_whale_identification/'
              'data/all/val_exp.csv', 'r') as f:
        for l in f.readlines():
            if "Image" in l:
                continue
            anno_temp = l.strip().split(',')
            im_name = anno_temp[0].strip()
            tmp_image = cv2.imread(
                os.path.join('/home/westwell/Desktop/dolores_storage/humpback_whale_identification/'
                             'data/all/train', im_name))
            cv2.imwrite(os.path.join(dst_folser, im_name), tmp_image)
