import argparse 
import os
from glob import glob
from itertools import chain

def parse_args():
    parser = argparse.ArgumentParser(description="Network alignment")
    parser.add_argument('--desti', default="dataspace/fb-tw-data/facebook")
    return parser.parse_args()

def change_content(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.split()
            data.append(data_line)
    file.close()
    
    with open(path, 'w', encoding='utf-8') as file:
        for data_line in data:
            new_data = []
            for ele in data_line:
                if '_' in ele:
                    new_data.append(ele.split('_')[-1])
            new_line = "\t".join(new_data) + "\n"
            file.write(new_line)
    file.close()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    result = (chain.from_iterable(glob(os.path.join(x[0], '*edgelist*')) for x in os.walk(args.desti)))
    result = [ele for ele in result if os.path.isfile(ele) and '.npy' not in ele]
    print(result)