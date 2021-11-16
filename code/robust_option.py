import argparse
import template

parser = argparse.ArgumentParser(description='RCDNet')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=0,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../data',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='RainHeavy', #'DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default= 'RainHeavyTest', #'DIV2K',
                    help='test dataset name')
parser.add_argument('--apath', type=str, default='../data/test/small/',
                    help='dataset directory')
parser.add_argument('--dir_hr', type=str, default='../data/test/small/norain',
                    help='dataset directory')
parser.add_argument('--dir_lr', type=str, default='../data/test/small/rain',
                    help='dataset directory')
parser.add_argument('--data_range', type=str, default='1-200/1-100',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='2',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=64,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--input_downsize', type=int, default=1,
                    help='resize the input data')
parser.add_argument('--gt_downsize', type=int, default=1,
                    help='resize the gt data')

# Model specifications
parser.add_argument('--model', default='RCDNet',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# RCDNet settings
parser.add_argument('--rcd_num_M', type=int, default=32,
                    help='the number of rain maps')
parser.add_argument('--rcd_num_Z', type=int, default=32,
                    help='the number of dual channles')
parser.add_argument('--T', type=int, default=4,
                    help='Resblocks number in each proxNet')
parser.add_argument('--rcd_stage', type=int, default=17,
                    help='Stage number S')
parser.add_argument('--rcd_width', type=float, default=1,
                    help='multiplier of width')
parser.add_argument('--rcd_depth', type=float, default=1,
                    help='multiplier of depth')
parser.add_argument('--rcd_branch', type=float, default=1,
                    help='number of branch')
parser.add_argument('--rcd_dilation', type=int, default=1,
                    help='dilation')

# MPRNet_R_SEADD_MB settings
parser.add_argument('--branch_reduction', type=int, default=1,
                    help='feature map width reduction for convolution path with dilations 2 or 4')

# RESCAN settings
parser.add_argument('--res_channel', type=int, default=24,
                    help='')
parser.add_argument('--res_stage_num', type=int, default=4,
                    help='')
parser.add_argument('--res_depth', type=int, default=7,
                    help='') #  >=3
parser.add_argument('--res_use_se', type=str, default='True',
                    help='')
parser.add_argument('--res_unit', type=str, default='GRU',
                    help='') # ['Conv', 'RNN', 'GRU', 'LSTM']
parser.add_argument('--res_frame', type=str, default='Full',
                    help='') # ['Conv', 'RNN', 'GRU', 'LSTM']

# JORDER_e settings
parser.add_argument('--jorder_e_channels', type=int, default=64,
                    help='the number of channels')
parser.add_argument('--jorder_e_width', type=float, default=1,
                    help='multiplier of width')
parser.add_argument('--jorder_e_tunnels', type=int, default=3,
                    help='the number of tunnels')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=25,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.2,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*MSE',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='RCDNet_syn',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save gt')

# robust
parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'ifgsm', 'free', 'none'])
parser.add_argument('--attack_iters', default=50, type=int)
parser.add_argument('--robust_epsilon', default=8, type=float)
parser.add_argument('--robust_alpha', default=2, type=float)
parser.add_argument('--restarts', default=1, type=int)
#parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
parser.add_argument('--save_attack', action='store_true',
                    help='save attacked images together')
parser.add_argument('--target', default='output', type=str, choices=['output', 'input','down_stream','down_stream_v2','residual'])
parser.add_argument('--attack_loss', default='l_2', type=str, choices=['l_1', 'l_2', 'lpips'])
parser.add_argument('--mse_weight', default=25, type=int, help='mse weight for down_stream attack')
parser.add_argument('--with_mask', action='store_true',
                    help='if the attack is with mask')
parser.add_argument('--rain_threshold', default=10, type=int)
parser.add_argument('--mask_norain', action='store_true',
                    help='if mask is on rain region')
parser.add_argument('--attack_gt', action='store_true',
                    help='attack gt')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False