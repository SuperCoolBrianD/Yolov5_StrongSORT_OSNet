#!/usr/bin/env python
__copyright__ = """
Copyright (c) 2020 Tananaev Denis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this permission
notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import os
import glob
import numpy as np
import argparse
from voxel_file_io import save_dataset_list
import os

class PandaDetectionDataset:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def get_data(self):
        search_string = os.path.join(self.dataset_dir, "*", "radar_processed.bin")
        lidar_list = np.asarray(sorted(glob.glob(search_string)))
        search_string = os.path.join(self.dataset_dir, "*", "boxes_processed.txt")
        box_list = np.asarray(sorted(glob.glob(search_string)))
        data = np.concatenate((lidar_list[:, None], box_list[:, None],), axis=1,)
        data = [";".join(x) for x in data]
        return data

    def filter_data(self, data_list):
        filtered_data_list = []
        for i in data_list:
            box = i.split(';')[1]
            if os.stat(box).st_size != 0:
                filtered_data_list.append(i)
        return filtered_data_list
    def create_datasets_file(self):
        """
        Creates  train.dataset  and val.dataset file
        """
        data_list = self.get_data()
        data_list = self.filter_data(data_list)
        num_data = len(data_list)
        split_num = int(num_data*0.75)
        print(f"split_num {split_num}")
        # Save train and validation dataset
        filename = os.path.join(self.dataset_dir, "train.datatxt")
        save_dataset_list(filename, data_list[:split_num])
        print(
            f"The dataset of the size {len(data_list[:split_num])} saved in {filename}."
        )
        filename = os.path.join(self.dataset_dir, "val.datatxt")
        save_dataset_list(filename, data_list[split_num:])
        print(
            f"The dataset of the size {len(data_list[split_num:])} saved in {filename}."
        )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create kitti dataset file.")
    parser.add_argument("--dataset_dir", default="dataset")
    args = parser.parse_args()
    dataset_creator = PandaDetectionDataset(args.dataset_dir)
    dataset_creator.create_datasets_file()
