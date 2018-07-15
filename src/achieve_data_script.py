from utils.data_loader import load_data, change_class_label, select_features_data, load_data_from_files

if __name__ == '__main__':


    input_file = '../original_data_no_sample/Wednesday-workingHours.pcap_ISCX.csv'
    output_file = '../original_data_no_sample/Wednesday-workingHours.pcap_ISCX_feature_selected.csv'
    invalid_file='../original_data_no_sample/Wednesday-workingHours.pcap_ISCX_invalid_data_file.csv'

    label_changed_output_file=change_class_label(input_file,invalid_file, label=[1,0])  # normal=1, attack=0

    input_file=label_changed_output_file
    output_file = '../original_data_no_sample/Wednesday-workingHours.pcap_ISCX_feature_selected.csv'
    features_selected = [' Source Port', ' Destination Port', ' Flow Duration', 'Total Length of Fwd Packets',
                         ' Total Length of Bwd Packets', 'Bwd Packet Length Max', ' Bwd Packet Length Min',
                         'Flow Bytes/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
                         'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Min', 'Bwd IAT Total',
                         ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
                         ' Fwd URG Flags', ' Fwd Header Length', ' Bwd Packets/s', ' Packet Length Mean',
                         ' ACK Flag Count', ' Down/Up Ratio', ' Avg Fwd Segment Size', ' Fwd Header Length',
                         'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Bwd Avg Bytes/Bulk', 'Bwd Avg Bulk Rate',
                         'Subflow Fwd Packets', ' Subflow Fwd Bytes', 'Init_Win_bytes_forward', ' act_data_pkt_fwd',
                         ' Active Std', ' Active Min', ' Idle Max']
    select_features_data(input_file,features_selected,output_file)

    # #X, Y = load_data(input_file, features_selected, output_file)
    input_file = output_file
    X,Y=load_data_from_files(input_file)

