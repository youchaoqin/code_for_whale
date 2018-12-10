"""make whale name to index map and index to whale map"""
import os
import numpy


if __name__ == "__main__":
    label_file_path = '../data/all/train.csv'

    name_to_index_map = {}
    index_to_name_map = {}
    id = 0
    with open(label_file_path, 'r') as lf:
        for l in lf.readlines():
            # skip the first line:
            if "Image" in l:
                continue
            cls_name = l.strip().split(',')[-1]
            cls_name = cls_name.strip()
            if cls_name in name_to_index_map:
                continue
            else:
                name_to_index_map[cls_name] = str(id)
                id += 1

    if "Id" in name_to_index_map:
        raise Exception('should not include the first row')

    for cls_name in name_to_index_map:
        id = name_to_index_map[cls_name]
        if id in index_to_name_map:
            raise Exception('duplicate id: %s, class name is %s'%(id, cls_name))
        else:
            index_to_name_map[id] = cls_name

    with open('../data/all/cls_name_to_index.txt', 'a') as f:
        for k in name_to_index_map:
            f.write('%s:%s\n'%(k, name_to_index_map[k]))

    with open('../data/all/index_to_cls_name.txt', 'a') as f:
        for k in index_to_name_map:
            f.write('%s:%s\n'%(k, index_to_name_map[k]))



