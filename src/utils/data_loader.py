# -*- coding: utf-8 -*-
import random

import torch
import torch.utils.data as Data
import numpy as np
import os
from collections import Counter

def load_data(input_file, features_selected, output_file,sample_ratio=1,preprocess=True):

    if os.path.exists(output_file):
        os.remove(output_file)
    if not os.path.exists(output_file):
        select_features_data(input_file, features_selected, output_file)  # save the selectted data to output_file

    return load_data_from_files(output_file,sample_ratio, preprocess)


def select_features_data(input_file, features_selected, output_file):
    with open(output_file, 'w') as out_file:
        with open(input_file, 'r') as in_file:
            line = in_file.readline()
            while line:
                line_arr = line.split(',')
                if line.startswith('Flow'):
                    index_arr = []
                    for i in range(len(features_selected)):
                        index_arr.append(line_arr.index(features_selected[i]))

                    line = in_file.readline()
                    continue

                line_tmp = ''
                for i in range(len(index_arr)):
                    line_tmp += line_arr[index_arr[i]] + ','
                # if line_arr[-1] =='0':
                #     line_arr[-1]=='1'
                # elif line_arr[-1] =='1':
                #     line_arr[-1] == '0'
                # else:
                #     print('unknow label:',line)
                line_tmp += line_arr[-1]
                out_file.write(line_tmp)

                line = in_file.readline()

    return output_file




def load_data_from_files(input_selected_features_file,sample_ratio=0.1, preprocess=True):

    benigin_data = []
    attack_data = []
    invalid_data = []
    X=[]
    Y=[]
    with open(input_selected_features_file, 'r') as in_file:
        line = in_file.readline()
        while line:
            line_arr = line.split(',')
            if line.startswith('Fl'):
                print(line)
                line = in_file.readline()
                continue
            if '-' in line or 'Nan' in line or 'Infinity' in line:  # ignore negative values
                line_arr[-1] = line_arr[-1].split('\n')[0]  # remove '\n'
                invalid_data.append(line_arr)
                # print('invalid data:',line,end='')
                line = in_file.readline()
                continue

            if line_arr[-1] == '0\n':
                line_arr[-1] = '1'
                benigin_data.append(line_arr[:-1])
            # elif line_arr[-1] == '1\n':
            #     line_arr[-1] = '0'
            #     attack_data.append(line_arr[:-1])
            #     # X.append(line_arr[:-1])
            #     # Y.append(line_arr[-1])
            else:
                line_arr[-1] = '0'    # others all reset to '0'
                attack_data.append(line_arr[:-1])
                # print('unknow label:',line,end='')

            line = in_file.readline()

    print('original_label: 1: %d, 0: %d'%(len(benigin_data), len(attack_data)))
    sample_size= int(len(benigin_data)*sample_ratio)
    print('sample size = ',sample_size,', int(len(benigin_data)*sample_ratio)')
    benigin_data=[ benigin_data[i] for i in sorted(random.sample(range(len(benigin_data)), sample_size)) ]
    attack_data=[ attack_data[i] for i in sorted(random.sample(range(len(attack_data)), sample_size)) ]

    if len(benigin_data) > len(attack_data):
        benigin_data=benigin_data[:len(attack_data)]
    else:
        attack_data=attack_data[:len(benigin_data)]

    X=benigin_data+attack_data
    for i in range(len(X)):
        X[i]= list(map(lambda x:float(x), X[i]))
        if i < len(benigin_data):
            Y.append(1)   # begin_data == 1
        else:
            Y.append(0)   # attack == 0

    return X, Y

    # return benigin_data,attack_data

def change_class_label(input_file, invalid_file,label=[1,0]):
    benigin_data = []
    attack_data = []
    invalid_data = []
    X = []
    Y = []
    all_data_count = 0
    output_file=input_file+'_label_changed_file.csv'
    with open(output_file, 'w') as out_file:
        with open(input_file, 'r') as in_file:
            line = in_file.readline()
            while line:
                line_arr = line.split(',')
                if line.startswith('Fl'):
                    print(line)
                    out_file.write(line)
                    line = in_file.readline()
                    continue

                all_data_count +=1
                line_arr[0]=str(all_data_count)   # instead of the first column(192.168.10.14-209.48.71.168-49459) with int_value
                line = ''
                for t in range(len(line_arr)):
                    line += line_arr[t]+','

                if '-' in line or 'NaN' in line or 'Infinity' in line:  # ignore negative values
                    if len(invalid_data) % 1000 ==0:
                        print(line_arr)
                    line_arr[-1] = line_arr[-1].split('\n')[0]  # remove '\n'
                    invalid_data.append(line_arr)
                    # print('invalid data:',line,end='')
                    line = in_file.readline()
                    continue


                if line_arr[-1] == 'BENIGN\n':
                    line_arr[-1] = label[0]
                    benigin_data.append(line_arr[:-1])
                # elif line_arr[-1] == '1\n':
                #     line_arr[-1] = '0'
                #     attack_data.append(line_arr[:-1])
                #     # X.append(line_arr[:-1])
                #     # Y.append(line_arr[-1])
                else:
                    line_arr[-1] = label[1]  # others all reset to '0'
                    attack_data.append(line_arr[:-1])
                    # print('unknow label:',line,end='')

                line_str = ''
                for j in range(len(line_arr)-1):
                    line_str += str(line_arr[j]) + ','
                out_file.write(line_str+str(line_arr[-1])+'\n')
                line = in_file.readline()

    if os.path.exists(invalid_file):
        os.remove(invalid_file)
    print('all data is %d, the number of invalid data is %d'%(all_data_count,len(invalid_data)))
    with open(invalid_file,'w') as out_file:
        for i in range(len(invalid_data)):
            line_str =''
            for j in range(len(invalid_data[i])):
                line_str +=str(invalid_data[i][j])+','
            out_file.write(line_str+'\n')
            out_file.flush()


    return output_file


def one_hot(ids, out_tensor):
    """
    ids: (list, ndarray) shape:[batch_size]
    out_tensor:FloatTensor shape:[batch_size, depth]
    """
    if not isinstance(ids, (list, np.ndarray)):
        raise ValueError("ids must be 1-D list or array")
    ids = torch.LongTensor(ids)
    out_tensor.zero_()
    return out_tensor.scatter_(dim=1, index=ids, value=1)
    # out_tensor.scatter_(1, ids, 1.0)

def normalize_data(data_no_label, axis=0, low=-1, high=1, eps=1e-5):

    data_tmp=np.copy(data_no_label)
    data_tmp=np.asarray(data_tmp,dtype='float')
    mins = np.min(data_tmp, axis)  # axis=0 comlumn
    maxs = np.max(data_tmp, axis)
    rng = maxs - mins
    for i in range(rng.shape[0]):
        if rng[i] == 0.0:
            rng[i] += eps

    # (x1-a)/(b-a)=(x2-mins)/(maxs-mins)
    # ==> x1= (x2-mins)/(maxs-mins)*(b-a)+a
    # data=high - (((high - low) * (maxs - data)) / (rng))
    data_tmp = (data_tmp - mins) / rng * (high - low) + low

    return data_tmp

def normalize(data, axis=0, low=-1, high=1, eps=1e-5):

    data_tmp=np.copy(data)
    data_tmp=data_tmp[:,:-1]
    label = []
    for i in range(len(data_tmp)):
        label.append(data[i][-1])
        # data[i] = np.asarray(list(map(lambda x: float(x), data[i][0:-1])))
        # data_tmp[i] = np.asarray(list(map(lambda x: float(x), data_tmp[i][0:].tolist())))  # lambda is useful to np.asarray
    data_tmp=np.asarray(data_tmp,dtype='float')
    mins = np.min(data_tmp, axis)  # axis=0 comlumn
    maxs = np.max(data_tmp, axis)
    rng = maxs - mins
    for i in range(rng.shape[0]):
        if rng[i] == 0.0:
            rng[i] += eps

    # (x1-a)/(b-a)=(x2-mins)/(maxs-mins)
    # ==> x1= (x2-mins)/(maxs-mins)*(b-a)+a
    # data=high - (((high - low) * (maxs - data)) / (rng))
    data_tmp = (data_tmp - mins) / rng * (high - low) + low
    # data=(data-mins)/rng
    new_data = []
    for i in range(len(label)):
        tmp = list(map(lambda x: str(x), data_tmp[i].tolist()))
        tmp.append(label[i])
        new_data.append(tmp)

    return new_data

# import os
# import pandas as pd
# from torch.utils.data import DataLoader, Dataset
#
#
# class MyDataset(Dataset):
#     """Face Landmarks dataset."""
#
#     def __init__(self, csv_file, root_dir, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.landmarks_frame)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample


# torch.manual_seed(1)  # reproducible
#
# BATCH_SIZE = 5  # 批训练的数据个数
#
# x = torch.linspace(1, 10, 10)  # x data (torch tensor)
# y = torch.linspace(10, 1, 10)  # y data (torch tensor)
#
# # 先转换成 torch 能识别的 Dataset
# torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
#
# # 把 dataset 放入 DataLoader
# loader = Data.DataLoader(
#     dataset=torch_dataset,  # torch TensorDataset format
#     batch_size=BATCH_SIZE,  # mini batch size
#     shuffle=True,  # 要不要打乱数据 (打乱比较好)
#     num_workers=2,  # 多线程来读数据
# )
#
# for epoch in range(3):  # 训练所有!整套!数据 3 次
#     for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
#         # 假设这里就是你训练的地方...
#
#         # 打出来一些数据
#         print(\'Epoch: \', epoch, \'| Step: \', step, \'| batch x: \',
#         batch_x.numpy(), \'| batch y: \', batch_y.numpy())
#
#                           """
#                           Epoch:  0 | Step:  0 | batch x:  [ 6.  7.  2.  3.  1.] | batch y:  [  5.   4.   9.   8.  10.]
#                           Epoch:  0 | Step:  1 | batch x:  [  9.  10.   4.   8.   5.] | batch y:  [ 2.  1.  7.  3.  6.]
#                           Epoch:  1 | Step:  0 | batch x:  [  3.   4.   2.   9.  10.] | batch y:  [ 8.  7.  9.  2.  1.]
#                           Epoch:  1 | Step:  1 | batch x:  [ 1.  7.  8.  5.  6.] | batch y:  [ 10.   4.   3.   6.   5.]
#                           Epoch:  2 | Step:  0 | batch x:  [ 3.  9.  2.  6.  7.] | batch y:  [ 8.  2.  9.  5.  4.]
#                           Epoch:  2 | Step:  1 | batch x:  [ 10.   4.   8.   1.   5.] | batch y:  [  1.   7.   3.  10.   6.]
#                           """


if __name__ == '__main__':

    res=one_hot([[1],[2],[3]],out_tensor=torch.LongTensor(2,5))
    print(res)
    torch.manual_seed(1)  # reproducible

    input_file = '../../data/ids2017_sampled.csv'
    output_file = '../../data/ids_selected_features_data.csv'

    features_selected = [' Source Port', ' Destination Port', ' Flow Duration', 'Total Length of Fwd Packets',
                         ' Total Length of Bwd Packets', 'Bwd Packet Length Max', ' Bwd Packet Length Min',
                         'Flow Bytes/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
                         'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Min', 'Bwd IAT Total',
                         ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
                         ' Fwd URG Flags', ' Fwd Header Length', ' Bwd Packets/s', ' Packet Length Mean',
                         ' ACK Flag Count', ' Down/Up Ratio', ' Avg Fwd Segment Size', ' Fwd Header Length.1',
                         'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Bwd Avg Bytes/Bulk', 'Bwd Avg Bulk Rate',
                         'Subflow Fwd Packets', ' Subflow Fwd Bytes', 'Init_Win_bytes_forward', ' act_data_pkt_fwd',
                         ' Active Std', ' Active Min', ' Idle Max']

    BATCH_SIZE = 1
    X, Y = load_data(input_file, features_selected,output_file, BATCH_SIZE)

    # X = torch.Tensor(np.asarray(X))
    # Y = torch.Tensor(np.asarray(Y))
    X=torch.Tensor(X)
    Y=torch.Tensor(Y)
    dataset = Data.TensorDataset(X, Y)
    data_loader = Data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,
        num_workers=2,
    )

    for step, (batch_x, batch_y) in enumerate(data_loader):
        # print(step, batch_x, batch_y)
        print('step:',step, batch_x.shape, batch_y.shape, Counter(batch_y.tolist()))


