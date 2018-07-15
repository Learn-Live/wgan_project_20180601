# -*- coding: utf-8 -*-
import time
from collections import Counter
import os

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.autograd import Variable, grad
from torchvision import datasets, transforms

from utils.data_loader import load_data, one_hot, normalize_data, load_data_from_files
from utils.show_save import *

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class WGAN(nn.Module):

    def __init__(self, *args, **kwargs):
        # super(WGAN,self).__init__() # python 2.x
        super().__init__()  # python 3.x

        # divide data
        (X, Y) = args[0]
        self.training_set=(X, Y)
        # get nn size
        self.nn_size_lst = args[1]
        self.in_size = self.nn_size_lst[0]
        self.h_size = self.nn_size_lst[1]
        self.out_size = self.nn_size_lst[2]
        self.g_in_size= self.in_size//10

        self.batch_size = args[2]
        self.epochs = args[3]

        self.show_flg = args[4]
        self.data_flg=args[5]

        self.map1 = nn.Linear(self.in_size, self.h_size * 2)
        self.map2 = nn.Linear(self.h_size * 2, self.h_size)
        self.map3 = nn.Linear(self.h_size, self.out_size)

        #
        #
        # self.model = nn.Sequential(self.map1, nn.Tanh(),
        #                            self.map2,nn.Tanh(),
        #                            self.map3
        #                            )

        self.D = nn.Sequential(nn.Linear(self.in_size, self.h_size * 2), nn.Tanh(),
                               nn.Linear(self.h_size * 2, self.h_size), nn.Tanh(),
                               self.map3
                               )

        self.G = nn.Sequential(nn.Linear(self.g_in_size, self.h_size * 2), nn.Tanh(),
                               nn.Linear(self.h_size * 2, self.h_size), nn.Tanh(),
                               nn.Linear(self.h_size, self.in_size), nn.Tanh()
                               )
        print('---------- Networks architecture -------------')
        print_network(self.D)
        print_network(self.G)
        print('-----------------------------------------------')

        self.criterion = nn.MSELoss(size_average=False)
        self.d_learning_rate = 1e-4
        self.g_learning_rate = 1e-4
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.d_learning_rate, betas=(0.5,0.9))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.g_learning_rate, betas=(0.5,0.9))

    def forward(self, x, mini_batch_size=10):
        y_preds = self.model(x)

        return y_preds

        # x = F.elu(self.map1(x))
        # x = F.elu(self.map2(x))
        # x = F.elu(self.map2(x))
        # # x = F.tanh(self.map1(x))
        # # x = F.tanh(self.map2(x))
        # # x = F.tanh(self.map2(x))
        # return F.tanh(self.map3(x))
        # # return self.map3(x)

    # def train(self):
    #
    #     for epoch in range(self.epochs):
    #
    #         dataset = Data.TensorDataset(self.training_set[0], self.training_set[1])  # X, Y
    #         training_set_data_loader = Data.DataLoader(
    #             dataset=dataset,  # torch TensorDataset format
    #             batch_size=1,  # mini batch size
    #             shuffle=True,
    #             num_workers=2,
    #         )
    #
    #         for mini_batch_index, (batch_x, batch_y) in enumerate(training_set_data_loader):
    #             y_preds = self.model(batch_x)
    #             # print(y_preds)
    #             # y_preds=self.model.forward(batch_x)
    #             batch_y_one_hot = one_hot(np.reshape(batch_y.numpy(), [len(batch_y.numpy()), 1]),
    #                                       out_tensor=torch.FloatTensor(y_preds.shape[0], y_preds.shape[1]))
    #             loss = self.criterion(y_preds, batch_y_one_hot)
    #             print(mini_batch_index, 'loss:', loss)
    #
    #             self.model.zero_grad()
    #             loss.backward()
    #
    #             # # Update the weights using gradient descent. Each parameter is a Tensor, so
    #             # # we can access and gradients like we did before.
    #             # with torch.no_grad():
    #             #     for param in self.model.parameters():
    #             #         param -= self.learning_rate * param.grad
    #
    #             self.optimizer.step()
    #
    #             # Weight Clipping
    #             for p in self.model.parameters():
    #                 p.data.clamp_(-0.01, 0.01)

    def train_gp(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.train_hist['D_decision'] = []

        self.gpu_mode = False
        self.lambda_ = 8.0
        self.n_critic = 3

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(
                torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(
                torch.zeros(self.batch_size, 1))

        # self.D.train()  # only useful for dropout, batch_normalizati
        print('training start!!')
        training_x_tmp=[]
        training_y_tmp=[]
        for i in range(len(self.training_set[1].tolist())):
            if self.data_flg=='benigin_data':
                if self.training_set[1].tolist()[i]==1:    # benigin_data
                    training_x_tmp.append(self.training_set[0].tolist()[i])
                    training_y_tmp.append(self.training_set[1].tolist()[i])
            elif self.data_flg=='attack_data':
                if self.training_set[1].tolist()[i]==0:   # attack_data
                    training_x_tmp.append(self.training_set[0].tolist()[i])
                    training_y_tmp.append(self.training_set[1].tolist()[i])
            else:
                print('self.data_flg ',self.data_flg)
                pass
        print(self.data_flg+' training set is', len(training_y_tmp),', counter y:',Counter(training_y_tmp))
        dataset = Data.TensorDataset(torch.Tensor(training_x_tmp), torch.Tensor(training_y_tmp)) # X, Y

        start_time = time.time()
        # self.D_loss_lst=[]
        # self.G_loss_lst=[]
        for epoch in range(self.epochs):
            # self.G.train()
            epoch_start_time = time.time()

            ### re divide dataset
            self.training_set_data_loader = Data.DataLoader(
                dataset=dataset,  # torch TensorDataset format
                batch_size=self.batch_size,  # mini batch size
                shuffle=True,
                num_workers=2,
            )
            for iter, (x_, y_) in enumerate(self.training_set_data_loader):
                if iter == self.training_set_data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.randn((self.batch_size, self.g_in_size))   # random normal 0-1

                if self.gpu_mode:
                    x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
                else:
                    x_, z_ = Variable(x_), Variable(z_)

                # update D network
                self.d_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = torch.mean(D_real)

                G_ = self.G(z_)  # detach to avoid training G on these labels
                # G_=self.G(z_)
                D_fake = self.D(G_.detach())
                D_fake_loss = torch.mean(D_fake)

                # gradient penalty
                if self.gpu_mode:
                    alpha = torch.rand(x_.size()).cuda()
                else:
                    alpha = torch.rand(x_.size())

                x_hat = Variable(alpha * x_.data + (1 - alpha) * G_.data, requires_grad=True)

                pred_hat = self.D(x_hat)
                if self.gpu_mode:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]
                else:
                    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

                gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

                D_loss = -(D_real_loss - D_fake_loss) + gradient_penalty
                wgan_distance=(D_real_loss - D_fake_loss)
                D_loss.backward()
                self.d_optimizer.step()

                if ((iter + 1) % self.n_critic) == 0:
                    # update G network
                    self.g_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = -torch.mean(D_fake)
                    self.train_hist['G_loss'].append(-G_loss.data[0])

                    G_loss.backward()
                    self.g_optimizer.step()

                    # self.train_hist['D_loss'].append(D_loss.data[0])
                    # print(wgan_distance.data[0], [D_real_loss.data[0],D_fake_loss.data[0], -G_loss.data[0]])
                    self.train_hist['D_loss'].append(wgan_distance.data[0])
                    self.train_hist['D_decision'].append([D_real_loss.data[0],D_fake_loss.data[0], -G_loss.data[0]])
                if ((iter + 1) % 5) == 0:
                    print("Epoch: [%2d] [%4d/%4d] wgan_distance: %.8f real:%.8f/fake:%.8f, -G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.training_set_data_loader.dataset.__len__() // self.batch_size,
                           wgan_distance.data[0], D_real_loss.data[0], D_fake_loss.data[0], -G_loss.data[0]))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            # self.visualize_results((epoch+1))
        show_figures(self.train_hist['D_loss'], self.train_hist['G_loss'])
        show_figures_2(self.train_hist['D_decision'])

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epochs, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        # self.save()
        # utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
        #                          self.epoch)
        # utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)


# def show_figures(D_loss, G_loss):
#     import matplotlib.pyplot as plt
#     plt.figure()
#
#     plt.plot(D_loss, 'r', label='D_loss')
#     plt.plot(G_loss, 'g', label='G_loss')
#     plt.legend(loc='upper right')
#     plt.show()
#
# def show_figures_2(decision_data):
#     import matplotlib.pyplot as plt
#     plt.figure()
#     new_decision_data=np.copy(decision_data)
#     plt.plot(new_decision_data[:,0], 'r', alpha =0.5, label='D_real')
#     plt.plot(new_decision_data[:,1], 'b', alpha =0.5,label='D_fake')
#     plt.plot(new_decision_data[:,2], 'g', alpha =0.5,label='D_G_fake')
#     plt.legend(loc='upper right')
#     plt.show()
#
# def save_tensor_data(tensor_data,output_file = 'testing_set.csv'):  # tensor_data: (X,Y)
#     with open(output_file, 'w') as out_file:
#         for i in range(len(tensor_data[0])):
#             # print('i',i)
#             line_str =''
#             row_value=tensor_data[0].tolist()[i]
#             for j in range(len(row_value)):
#                 line_str += str(row_value[j]) + ','
#             out_file.write(line_str + str(int(tensor_data[1].tolist()[i])) + '\n')
#             out_file.flush()
#
# def save_data(gen_data, data_type='0', output_file='./gen_data.csv'):
#     if os.path.exists(output_file):
#         os.remove(output_file)
#     with open(output_file,'w') as out_file:
#         for i in range(gen_data.shape[0]):
#             line_str =''
#             for j in range(len(gen_data[i].tolist())):
#                 line_str +=str(gen_data[i].tolist()[j])+','
#             out_file.write(line_str+data_type+'\n')
#             out_file.flush()
#
#
# def arff_header(output_file, feature_lst=''):
#    """
#    features_selected = [' Source Port', ' Destination Port', ' Flow Duration', 'Total Length of Fwd Packets',
#                          ' Total Length of Bwd Packets', 'Bwd Packet Length Max', ' Bwd Packet Length Min',
#                          'Flow Bytes/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
#                          'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Min', 'Bwd IAT Total',
#                          ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
#                          ' Fwd URG Flags', ' Fwd Header Length', ' Bwd Packets/s', ' Packet Length Mean',
#                          ' ACK Flag Count', ' Down/Up Ratio', ' Avg Fwd Segment Size', ' Fwd Header Length.1',
#                          'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Bwd Avg Bytes/Bulk', 'Bwd Avg Bulk Rate',
#                          'Subflow Fwd Packets', ' Subflow Fwd Bytes', 'Init_Win_bytes_forward', ' act_data_pkt_fwd',
#                          ' Active Std', ' Active Min', ' Idle Max']
#    :param output_file:
#    :param feature_lst:
#    :return:
#
#
#    for examples:
#         @relation 'DDoS FC-weka.filters.unsupervised.attribute.Remove-R1'
#
#         @attribute srcIPs string
#         @attribute nPkts numeric
#         @attribute uPktsSize numeric
#         @attribute stdPktsSize numeric
#         @attribute uArrivalTime numeric
#         @attribute stdArrivalTime numeric
#         @attribute Class {DDoS_Flow,FC_Flow}
#
#         @data
#
#    """
#    with open(output_file, 'w') as out_file:
#
#        out_file.write('@relation gen_data\n\n')
#        for i in range(len(feature_lst)):
#            line = '@attribute \'' +str(feature_lst[i]) +'\' numeric'
#            out_file.write(line+'\n')
#
#        out_file.write('@attribute Class {1,0}\n\n')
#        out_file.write('@data\n')
#
#
# def merge_files(file_lst,header='header.txt' , feature_lst=''):
#
#     if os.path.exists(header):
#         os.remove(header)
#     # if not os.path.exists(header):
#     arff_header(header,feature_lst)
#
#     if os.path.exists(file_list[0]):
#         os.remove(file_list[0])
#     with open(file_lst[0], 'w') as out_file:
#         with open(header,'r') as header_file:
#             line = header_file.readline()
#             while line:
#                 out_file.write(line)
#                 line =header_file.readline()
#
#         for i in range(len(file_lst) - 1):
#             with open(file_lst[i + 1], 'r') as in_file:
#                 line = in_file.readline()
#                 while line:
#                     # if line[-2]=='.0':  # change the label value
#                     #     line[-2]=''
#                     out_file.write(line)
#                     line = in_file.readline()
#
# def split_data(X, Y, percent=0.7):
#     dataset = Data.TensorDataset(torch.Tensor(X), torch.Tensor(Y))
#     dataset_loader = Data.DataLoader(
#         dataset=dataset,  # torch TensorDataset format
#         batch_size=int(percent * len(X)),  # mini batch size
#         shuffle=True,
#         num_workers=2,
#     )
#     if percent >= 0.5:
#         for step, (batch_x, batch_y) in enumerate(dataset_loader):
#             if step == 0:
#                 training_set = (batch_x, batch_y)
#                 continue
#             else:  # percent <=0.5:
#                 testing_set = (batch_x, batch_y)
#                 break
#     else:
#         testing_set_X = []
#         testing_set_Y = []
#         for step, (batch_x, batch_y) in enumerate(dataset_loader):
#             if step == 0:
#                 training_set = (batch_x, batch_y)
#                 continue
#             else:
#                 for i in range(len(batch_x)):
#                     testing_set_X.append(batch_x.tolist()[i])
#                     testing_set_Y.append(batch_y.tolist()[i])
#
#         testing_set = (torch.Tensor(testing_set_X), torch.Tensor(testing_set_Y))
#
#     return training_set, testing_set
#
# def evaluate(predict_label, real_label,threshold=0.5,show_flg=True):
#
#     print('len(predict)= %d, len(real_label) = %d, threshold =%f'%(len(predict_label), len(real_label),threshold))
#     real_cnt = 0
#     real_error_cnt = 0
#     real_all_cnt = 0
#
#     fake_cnt = 0
#     fake_error_cnt = 0
#     fake_all_cnt = 0
#     i = 0
#     for i in range(len(predict_label)):
#         label=str(int(real_label[i]))
#         decision_value=predict_label[i]
#         if decision_value > threshold and label == '1':
#             real_cnt += 1
#         elif decision_value <= threshold and label == '1':
#             real_error_cnt += 1
#         elif decision_value >= threshold and label == '0':
#             fake_error_cnt += 1
#         elif decision_value < threshold and label == '0':
#             fake_cnt += 1
#         else:
#             pass
#
#         if label == '1':
#             real_all_cnt += 1
#         elif label == '0':
#             fake_all_cnt += 1
#         else:
#             print(predict_label[i])
#             pass
#
#     if (real_all_cnt + fake_all_cnt) == 0:
#         accuracy = 0.0
#     else:
#         accuracy = (real_cnt + fake_cnt) / (real_all_cnt + fake_all_cnt)
#
#     if real_all_cnt == 0:
#         real_accuracy = 0.0
#         real_error_accuracy = 0.0
#     else:
#         real_accuracy = real_cnt / real_all_cnt
#         real_error_accuracy = real_error_cnt / real_all_cnt
#         i += 1
#     if fake_all_cnt == 0:
#         fake_accuracy = 0.0
#         fake_error_accuracy = 0.0
#     else:
#         fake_accuracy = fake_cnt / fake_all_cnt
#         fake_error_accuracy = fake_error_cnt / fake_all_cnt
#
#     if show_flg:
#         print(
#             '---data :%d (benigin data: %d + attack data: %d)\n'
#             '*****Accuracy :%.3f\n'
#             '\t confusion matrix:\n'
#             '\t\t(real_data_accuracy:%.3f, real_error_accuracy:%.3f\n'
#             '\t\t attack_error_accuracy:%.3f,   attack_data_accuracy:%.3f)\n' %
#             (real_all_cnt + fake_all_cnt,
#              real_all_cnt, fake_all_cnt,
#              accuracy,
#              real_accuracy, real_error_accuracy,
#              fake_error_accuracy, fake_accuracy))
#
#     return [real_all_cnt + fake_all_cnt,
#             real_all_cnt, fake_all_cnt,
#             accuracy,
#             real_accuracy, real_error_accuracy,
#             fake_error_accuracy, fake_accuracy]
#

if __name__ == '__main__':
    torch.manual_seed(1)  # reproducible

    input_file = '../data/ids2017_sampled.csv'
    output_file = '../data/ids_selected_features_data.csv'

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
    #
    X, Y = load_data(input_file, features_selected, output_file,sample_ratio=1)
    # output_file = '../original_data_no_sample/Wednesday-workingHours.pcap_ISCX_feature_selected.csv'
    # X, Y= load_data_from_files(output_file, sample_ratio=0.05, preprocess=True)
    print('label:', Counter(Y))
    X = normalize_data(X, axis=0, low=-1, high=1, eps=1e-5)
    print('X.shape:', X.shape, ' Y.shape:', np.asarray(Y).shape)
    print('X[0]:', X[0])
    show_flg = True
    save_flg=True
    in_size = 41
    h_size = 15
    out_size = 1
    dtype = torch.float
    percent=0.3
    root_dir='./wgan_data_'+str(percent)+'percent'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    training_set, testing_set=split_data(X, Y, percent)
    print('training percent',percent,':training_set', training_set[0].shape,', testing_set', testing_set[0].shape)

    if save_flg:
        test_percent='%.1f'%(1-percent)
        save_tensor_data(training_set, output_file=os.path.join(root_dir,str(percent)+'_training_set.csv'))
        save_tensor_data(testing_set, output_file=os.path.join(root_dir,str(test_percent)+'_testing_set.csv'))

        file_list = [os.path.join(root_dir,str(percent)+'_origin_training_set.arff'), os.path.join(root_dir,str(percent)+'_training_set.csv')]
        merge_files(file_list, header='header.txt', feature_lst=features_selected)  # add_arff_header
        file_list = [os.path.join(root_dir,str(test_percent)+'_origin_testing_set.arff'), os.path.join(root_dir,str(test_percent)+'_testing_set.csv')]
        merge_files(file_list, header='header.txt', feature_lst=features_selected)  # add_arff_header


    # N = 100

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    # N, D_in, H, D_out = 64, 1000, 100, 10

    mini_batch_size = 20
    epochs = 1
    nn_size_lst = [in_size, h_size, out_size]

    for i in range(2):
        if i ==0:
            data_flg = 'benigin_data'
        else:
            data_flg='attack_data'
        wgan = WGAN((training_set[0], training_set[1]), nn_size_lst, mini_batch_size, epochs, show_flg, data_flg)

        # wgan.train()
        wgan.train_gp()

        input = torch.randn((1000, in_size//10))
        gen_data=wgan.G(input)

        output_file = os.path.join(root_dir,str(percent)+'_gen_' + data_flg + '.csv')
        if data_flg=='benigin_data':
            data_type = '1'
            save_data(gen_data,data_type,output_file)
        else:
            data_type='0'
            save_data(gen_data, data_type, output_file)

        ### test
        normal_data=[]
        for j in range(len(training_set[1])):
            if int(training_set[1][j].tolist())==1: # normal data
                normal_data.append(training_set[0][j])
            if len(normal_data) > mini_batch_size:
                break
        normal_data= torch.stack(normal_data)
        threshold = wgan.D(normal_data).mean()  # get threshold on average

        #test on training set
        preds_res = wgan.D(training_set[0])
        preds_res = list(map(lambda x: x[0].tolist(), preds_res))
        evaluate(preds_res, training_set[1].tolist(), threshold[0].tolist())
        # test on testing set
        preds_res = wgan.D(testing_set[0])
        preds_res= list(map(lambda x:x[0].tolist(),preds_res))
        evaluate(preds_res,testing_set[1].tolist(),threshold[0].tolist())

    ## merge files
    file_list=[os.path.join(root_dir,str(percent)+'_all_in_one_file.csv.arff'), os.path.join(root_dir,str(percent)+'_training_set.csv'),
               os.path.join(root_dir,str(percent)+'_gen_benigin_data.csv'),os.path.join(root_dir,str(percent)+'_gen_attack_data.csv')]
    merge_files(file_list, header='header.txt', feature_lst=features_selected)



