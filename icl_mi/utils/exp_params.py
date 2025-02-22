import argparse
import re

def get_default_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-seeds',
                        '-s', dest="seeds", default= 1,
                        type=int,
                        help='System seed for reproducibility')

    # model
    parser.add_argument('-model',
                        '-m', dest="model", default='gpt2-xl',
                        type= str,
                        help="Model's name")

    parser.add_argument('-num_of_bins',
                        '-nbins', dest="num_of_bins", default=100, type=int,
                        help='The number of bins that we divide the output of the neurons')

    # parser.add_argument('-start_samples',
    #                     '-ss', dest="start_samples", default=1,
    #                     type=int, help='The number of the first sample that we calculate the information')

    # parser.add_argument('-batch_size',
    #                     '-b', dest="batch_size", default=512,
    #                     type=int, help='The size of the batch')

    # parser.add_argument('-learning_rate',
    #                     '-l', dest="learning_rate", default=0.0004,
    #                     type=float,
    #                     help='The learning rate of the network')

    # parser.add_argument('-num_repeat',
    #                     '-r', dest="num_of_repeats", default=1,
    #                     type=int, help='The number of times to run the network')

    # parser.add_argument('-num_epochs',
    #                     '-e', dest="num_ephocs", default=10, #8000
    #                     type=int, help='max number of epochs')

    # parser.add_argument('-net',
    #                     '-n', dest="net_type", default='1',
    #                     help='The architecture of the networks')

    # parser.add_argument('-inds',
    #                     '-i', dest="inds", default='[80]',
    #                     help='The percent of the training data')

    # parser.add_argument('-name',
    #                     '-na', dest="name", default='net',
    #                     help='The name to save the results')

    # parser.add_argument('-d_name',
    #                     '-dna', dest="data_name", default='var_u',
    #                     help='The dataset that we want to run ')

    # parser.add_argument('-num_samples',
    #                     '-ns', dest="num_of_samples", default=400,
    #                     type=int,
    #                     help='The max number of indexes for calculate information')

    # parser.add_argument('-nDistSmpls',
    #                     '-nds', dest="nDistSmpls", default=1,
    #                     type=int, help='S')

    # parser.add_argument('-save_ws',
    #                     '-sws', dest="save_ws", type=str2bool, nargs='?', const=False, default=False,
    #                     help='if we want to save the output of the layers')

    # parser.add_argument('-calc_information',
    #                     '-cinf', dest="calc_information", type=str2bool, nargs='?', const=True, default=True,
    #                     help='if we want to calculate the MI in the network for all the epochs')

    # parser.add_argument('-calc_information_last',
    #                     '-cinfl', dest="calc_information_last", type=str2bool, nargs='?', const=False, default=False,
    #                     help='if we want to calculate the MI in the network only for the last epoch')

    # parser.add_argument('-save_grads',
    #                     '-sgrad', dest="save_grads", type=str2bool, nargs='?', const=False, default=False,
    #                     help='if we want to save the gradients in the network')

    # parser.add_argument('-run_in_parallel',
    #                     '-par', dest="run_in_parallel", type=str2bool, nargs='?', const=False, default=False,
    #                     help='If we want to run all the networks in parallel mode')

    # parser.add_argument('-activation_function',
    #                     '-af', dest="activation_function", default=0, type=int,
    #                     help='The activation function of the model 0 for thnh 1 for RelU')

    # parser.add_argument('-iad', dest="interval_accuracy_display", default=499, type=int,
    #                     help='The interval for display accuracy')

    # parser.add_argument('-interval_information_display',
    #                     '-iid', dest="interval_information_display", default=30, type=int,
    #                     help='The interval for display the information calculation')

    # parser.add_argument('-cov_net',
    #                     '-cov', dest="cov_net", type=int, default=0,
    #                     help='True if we want covnet')

    # parser.add_argument('-rl',
    #                     '-rand_labels', dest="random_labels", type=str2bool, nargs='?', const=False, default=False,
    #                     help='True if we want to set random labels')

    # parser.add_argument('-data_dir',
    #                     '-dd', dest="data_dir", default='data/',
    #                     help='The directory for finding the data')

    # args, = parser.parse_args()
    args, unkonwn = parser.parse_known_args()

    # args.inds = [map(int, inner.split(',')) for inner in re.findall("\[(.*?)\]", args.inds)]
    # if num_of_samples != None:
    #     args.inds = [[num_of_samples]]
    
    return args