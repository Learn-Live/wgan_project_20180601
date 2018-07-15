import os
import numpy as np
import torch
import torch.utils.data as Data

def show_figures(D_loss, G_loss):
    import matplotlib.pyplot as plt
    plt.figure()

    plt.plot(D_loss, 'r', alpha=0.5, label='Wgan_distance(D_loss-G_loss)')
    # plt.plot(G_loss, 'g', alpha=0.5, label='G_loss')
    plt.legend(loc='upper right')
    plt.show()

def show_figures_2(decision_data):
    import matplotlib.pyplot as plt
    plt.figure()
    new_decision_data=np.copy(decision_data)
    plt.plot(new_decision_data[:,0], 'r', alpha =0.5, label='D_real')
    # plt.plot(new_decision_data[:,1], 'b', alpha =0.5,label='D_fake')
    plt.plot(new_decision_data[:,2], 'g', alpha =0.5,label='D_G_fake')
    plt.legend(loc='upper right')
    plt.show()

def save_tensor_data(tensor_data,output_file = 'testing_set.csv'):  # tensor_data: (X,Y)
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, 'w') as out_file:
        for i in range(len(tensor_data[0])):
            # print('i',i)
            line_str =''
            row_value=tensor_data[0].tolist()[i]
            for j in range(len(row_value)):
                line_str += str(row_value[j]) + ','
            out_file.write(line_str + str(int(tensor_data[1].tolist()[i])) + '\n')
            out_file.flush()

def save_data(gen_data, data_type='0', output_file='./gen_data.csv'):
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file,'w') as out_file:
        for i in range(gen_data.shape[0]):
            line_str =''
            for j in range(len(gen_data[i].tolist())):
                line_str +=str(gen_data[i].tolist()[j])+','
            out_file.write(line_str+data_type+'\n')
            out_file.flush()


def arff_header(output_file, feature_lst=''):
   """
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
   :param output_file:
   :param feature_lst:
   :return:


   for examples:
        @relation 'DDoS FC-weka.filters.unsupervised.attribute.Remove-R1'

        @attribute srcIPs string
        @attribute nPkts numeric
        @attribute uPktsSize numeric
        @attribute stdPktsSize numeric
        @attribute uArrivalTime numeric
        @attribute stdArrivalTime numeric
        @attribute Class {DDoS_Flow,FC_Flow}

        @data

   """
   if os.path.exists(output_file):
       os.remove(output_file)
   with open(output_file, 'w') as out_file:

       out_file.write('@relation gen_data\n\n')
       for i in range(len(feature_lst)):
           line = '@attribute \'' +str(feature_lst[i]) +'\' numeric'
           out_file.write(line+'\n')

       out_file.write('@attribute Class {1,0}\n\n')
       out_file.write('@data\n')


def merge_files(file_lst=[],header='header.txt' , feature_lst=''):

    if os.path.exists(header):
        os.remove(header)
    # if not os.path.exists(header):
    arff_header(header,feature_lst)

    if os.path.exists(file_lst[0]):
        os.remove(file_lst[0])
    with open(file_lst[0], 'w') as out_file:
        with open(header,'r') as header_file:
            line = header_file.readline()
            while line:
                out_file.write(line)
                line =header_file.readline()

        for i in range(len(file_lst) - 1):
            with open(file_lst[i + 1], 'r') as in_file:
                line = in_file.readline()
                while line:
                    # if line[-2]=='.0':  # change the label value
                    #     line[-2]=''
                    out_file.write(line)
                    line = in_file.readline()

def split_data(X, Y, percent=0.7):
    dataset = Data.TensorDataset(torch.Tensor(X), torch.Tensor(Y))
    dataset_loader = Data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=int(percent * len(X)),  # mini batch size
        shuffle=True,
        num_workers=2,
    )
    if percent >= 0.5:
        for step, (batch_x, batch_y) in enumerate(dataset_loader):
            if step == 0:
                training_set = (batch_x, batch_y)
                continue
            else:  # percent <=0.5:
                testing_set = (batch_x, batch_y)
                break
    else:
        testing_set_X = []
        testing_set_Y = []
        for step, (batch_x, batch_y) in enumerate(dataset_loader):
            if step == 0:
                training_set = (batch_x, batch_y)
                continue
            else:
                for i in range(len(batch_x)):
                    testing_set_X.append(batch_x.tolist()[i])
                    testing_set_Y.append(batch_y.tolist()[i])

        testing_set = (torch.Tensor(testing_set_X), torch.Tensor(testing_set_Y))

    return training_set, testing_set



def evaluate(predict_label, real_label,threshold=0.5,show_flg=True):

    print('len(predict)= %d, len(real_label) = %d, threshold =%f'%(len(predict_label), len(real_label),threshold))
    real_cnt = 0
    real_error_cnt = 0
    real_all_cnt = 0

    fake_cnt = 0
    fake_error_cnt = 0
    fake_all_cnt = 0
    i = 0
    for i in range(len(predict_label)):
        label=str(int(real_label[i]))
        decision_value=predict_label[i]
        if decision_value > threshold and label == '1':
            real_cnt += 1
        elif decision_value <= threshold and label == '1':
            real_error_cnt += 1
        elif decision_value >= threshold and label == '0':
            fake_error_cnt += 1
        elif decision_value < threshold and label == '0':
            fake_cnt += 1
        else:
            pass

        if label == '1':
            real_all_cnt += 1
        elif label == '0':
            fake_all_cnt += 1
        else:
            print(predict_label[i])
            pass

    if (real_all_cnt + fake_all_cnt) == 0:
        accuracy = 0.0
    else:
        accuracy = (real_cnt + fake_cnt) / (real_all_cnt + fake_all_cnt)

    if real_all_cnt == 0:
        real_accuracy = 0.0
        real_error_accuracy = 0.0
    else:
        real_accuracy = real_cnt / real_all_cnt
        real_error_accuracy = real_error_cnt / real_all_cnt
        i += 1
    if fake_all_cnt == 0:
        fake_accuracy = 0.0
        fake_error_accuracy = 0.0
    else:
        fake_accuracy = fake_cnt / fake_all_cnt
        fake_error_accuracy = fake_error_cnt / fake_all_cnt

    if show_flg:
        print(
            '---data :%d (benigin data: %d + attack data: %d)\n'
            '*****Accuracy :%.3f\n'
            '\t confusion matrix:\n'
            '\t\t(real_cnt(TP):%d, real_error_cnt(FN):%d\n'
            '\t\t fake_error_cnt(FP):%d,   fake_cnt(TN):%d)\n' %
            (real_all_cnt + fake_all_cnt,
             real_all_cnt, fake_all_cnt,
             accuracy,
             real_cnt, real_error_cnt,
             fake_error_cnt, fake_cnt))

    return [real_all_cnt + fake_all_cnt,
            real_all_cnt, fake_all_cnt,
            accuracy,
            real_cnt, real_error_cnt,
            fake_error_cnt, fake_cnt]
