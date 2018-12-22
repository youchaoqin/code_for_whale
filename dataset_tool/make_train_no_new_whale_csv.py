import os
import random
import numpy as np

if __name__ == "__main__":
    anno_file = "/home/westwell/Desktop/dolores_storage/humpback_whale_identification/" \
                "data/all/train.csv"
    train_csv_no_new_whale = []
    with open(anno_file, 'r') as f:
        for l in f.readlines():
            if ("Image" in l) or ('new_whale' in l):
                continue
            train_csv_no_new_whale.append(l)
    output_path = os.path.join(os.path.dirname(anno_file), 'train_no_new_whale.csv')

    # random.shuffle(train_csv_no_new_whale)
    # random.shuffle(train_csv_no_new_whale)
    # random.shuffle(train_csv_no_new_whale)
    with open(output_path, 'w') as fid:
        fid.write('Image,Id\n')
        for w in train_csv_no_new_whale:
            fid.write(w)  # already has '\n'
