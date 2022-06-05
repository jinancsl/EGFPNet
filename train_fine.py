import argparse
from collections import OrderedDict
import random
from numpy import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from metrics import dice_coef, batch_iou, mean_iou, iou_score, dice_coef_guiyi, precision, teyixing, minganxing, calculate_aucs, TP, TN, FP, FN
import losses
from utils import str2bool, count_params
from skimage.measure import label


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='EGFPNet_fine',
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Unet', )
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default="None",
                        help='dataset name')
    parser.add_argument('--input-channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss')
    parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

size1 = 224
size2 = 224

def train(args, model, criterion, criterion_edge, optimizer, train_image_people):
    losses = AverageMeter()
    ious = AverageMeter()
    model.train()
    train_image = np.zeros(shape=(args.batch_size, 1, size1, size2))
    train_label = np.zeros(shape=(args.batch_size, 1, size1, size2))
    edge_label_nd = np.zeros(shape=(args.batch_size, 1, size1, size2))

    for j in range(int(len(train_image_people) // args.batch_size)):
        for k in range(args.batch_size):
            train_image[k, :, :, :] = train_image_people[j * args.batch_size + k][0]
            train_label[k, :, :, :] = train_image_people[j * args.batch_size + k][1]
            edge_label_nd[k, :, :, :] = train_image_people[j * args.batch_size + k][2]
        input = train_image.copy()
        target = train_label.copy()
        input = torch.from_numpy(input).cuda().float()
        target = torch.from_numpy(target).cuda().float()
        edge_label_tor = torch.from_numpy(edge_label_nd).cuda().float()

        final_out, out_edge, e5_and_edge_to_out, d5_and_edge_to_out, d4_and_edge_to_out, d3_and_edge_to_out = model(input)
        loss = criterion(final_out, target) + criterion(e5_and_edge_to_out, target) + criterion(d5_and_edge_to_out, target) + criterion(d4_and_edge_to_out, target) + criterion(d3_and_edge_to_out, target) + \
               criterion_edge(out_edge, edge_label_tor)
        iou = dice_coef(final_out, target)
        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def validate(args, model, criterion, test_image):
    losses = AverageMeter()
    ious = AverageMeter()

    # switch to evaluate mode
    model.eval()
    result_all = np.zeros(shape=(len(test_image), size1, size2))
    label_all = np.zeros(shape=(len(test_image), size1, size2))
    with torch.no_grad():
        image = np.zeros(shape=(args.batch_size, 1, size1, size2))
        label = np.zeros(shape=(args.batch_size, 1, size1, size2))

        for j in range(int(len(test_image) // args.batch_size)):
            for k in range(args.batch_size):
                if j * args.batch_size + k <= len(test_image) - 1:
                    image[k, :, :, :] = test_image[j * args.batch_size + k][0]
                    label[k, :, :, :] = test_image[j * args.batch_size + k][1]
                else:
                    break
            input = torch.from_numpy(image).cuda().float()
            target = torch.from_numpy(label).cuda().float()

            final_out, out_edge, e5_to_edge_final_out, d5_to_edge_final_out, d4_to_edge_final_out, d3_to_edge_final_out = model(input)
            loss = criterion(final_out, target)
            iou = dice_coef(final_out, target)
            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            final_out_nd = torch.sigmoid(final_out).data.cpu().numpy()
            for k1 in range(args.batch_size):
                if j * args.batch_size + k1 <= len(test_image) - 1:
                    result_all[j * args.batch_size + k1, :, :] = final_out_nd[k1, 0, :, :]
                    label_all[j * args.batch_size + k1, :, :] = label[k1, 0, :, :]
                else:
                    break
    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log, result_all, label_all

def superposition(image, groups):
    for i in range(len(groups)):
        image.append(groups[i])

def main_label(args, groups, label):
    five = 5
    for repeats in range(five):
        train_image_people = []
        validate_image_people = []
        test_image_people = []
        if repeats % 5 == 0:
            groups_list = [0, 1, 2, 3]
            random.shuffle(groups_list)
            superposition(train_image_people, groups[groups_list[0]])
            superposition(train_image_people, groups[groups_list[1]])
            superposition(train_image_people, groups[groups_list[2]])
            superposition(validate_image_people, groups[groups_list[3]])
            superposition(test_image_people, groups[4])
        elif repeats % 5 == 1:
            groups_list = [4, 1, 2, 3]
            random.shuffle(groups_list)
            superposition(train_image_people, groups[groups_list[0]])
            superposition(train_image_people, groups[groups_list[1]])
            superposition(train_image_people, groups[groups_list[2]])
            superposition(validate_image_people, groups[groups_list[3]])
            superposition(test_image_people, groups[0])
        elif repeats % 5 == 2:
            groups_list = [0, 4, 2, 3]
            random.shuffle(groups_list)
            superposition(train_image_people, groups[groups_list[0]])
            superposition(train_image_people, groups[groups_list[1]])
            superposition(train_image_people, groups[groups_list[2]])
            superposition(validate_image_people, groups[groups_list[3]])
            superposition(test_image_people, groups[1])
        elif repeats % 5 == 3:
            groups_list = [0, 1, 4, 3]
            random.shuffle(groups_list)
            superposition(train_image_people, groups[groups_list[0]])
            superposition(train_image_people, groups[groups_list[1]])
            superposition(train_image_people, groups[groups_list[2]])
            superposition(validate_image_people, groups[groups_list[3]])
            superposition(test_image_people, groups[2])
        elif repeats % 5 == 4:
            groups_list = [0, 1, 2, 4]
            random.shuffle(groups_list)
            superposition(train_image_people, groups[groups_list[0]])
            superposition(train_image_people, groups[groups_list[1]])
            superposition(train_image_people, groups[groups_list[2]])
            superposition(validate_image_people, groups[groups_list[3]])
            superposition(test_image_people, groups[3])

        random.shuffle(train_image_people)
        random.shuffle(validate_image_people)
        random.shuffle(test_image_people)
        # create model
        model = EGFPNet.EGFPNet()

        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        model = model.cuda()
        model = nn.DataParallel(model)
        # define loss function (criterion)
        if args.loss == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss().cuda()
        else:
            criterion = losses.BCEDiceLoss()
            criterion_edge = losses.BCE()
        cudnn.benchmark = True

        model = model.cuda()

        print(count_params(model))

        if args.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                   weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

        best_iou = 0
        trigger = 0
        tex_path = '/media/csl/5b58b9b5-0162-4611-b882-6351023584d1/csl/methods_data/edge_pyramid_net/result/acdc/' + label + '/' + args.name + str(repeats) + '.txt'
        results = open(tex_path, 'w')
        for epoch in range(args.epochs):
            # train for one epoch
            train_log = train(args, model, criterion, criterion_edge, optimizer, train_image_people)
            # evaluate on validation set
            val_log, val_result_all, val_label_all_val = validate(args, model, criterion, validate_image_people)

            print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
                  % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

            results.write('lr %.4f - epoch %.4f - loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f' % (
            optimizer.param_groups[0]['lr'],
            epoch, train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'],))
            results.write('\n')

            trigger += 1

            if val_log['iou'] > best_iou:
                torch.save(model.state_dict(), '/media/csl/5b58b9b5-0162-4611-b882-6351023584d1/csl/methods_data/edge_pyramid_net/models/acdc/' + label + '/' + args.name + str(repeats) + '.pth')
                best_iou = val_log['iou']
                test, result_all, label_all = validate(args, model, criterion, test_image_people)
                trigger = 0

            # early stopping
            if not args.early_stop is None:
                if trigger >= args.early_stop:
                    np.savez('/media/csl/5b58b9b5-0162-4611-b882-6351023584d1/csl/methods_data/edge_pyramid_net/result/acdc/' + label + '/' + args.name + str(repeats), result_all=result_all, label_all=label_all)  # 保存数据集
                    print("=> early stopping")
                    break
            torch.cuda.empty_cache()

if __name__ == '__main__':
    args = parse_args()
    #label1
    ######groups 1
    datasets_train = np.load(
        '/media/csl/5b58b9b5-0162-4611-b882-6351023584d1/csl/methods_data/edge_pyramid_net/datasets/acdc/label1/acdc_dataset_label1_fine_groups_1.npz',
        allow_pickle=True)
    groups_1 = datasets_train['people']

    #groups 2
    datasets_validate = np.load(
        '/media/csl/5b58b9b5-0162-4611-b882-6351023584d1/csl/methods_data/edge_pyramid_net/datasets/acdc/label1/acdc_dataset_label1_fine_groups_2.npz',
        allow_pickle=True)
    groups_2 = datasets_validate['people']

    #groups 3
    datasets_test = np.load(
        '/media/csl/5b58b9b5-0162-4611-b882-6351023584d1/csl/methods_data/edge_pyramid_net/datasets/acdc/label1/acdc_dataset_label1_fine_groups_3.npz',
        allow_pickle=True)
    groups_3 = datasets_test['people']

    #groups 4
    datasets_train = np.load(
        '/media/csl/5b58b9b5-0162-4611-b882-6351023584d1/csl/methods_data/edge_pyramid_net/datasets/acdc/label1/acdc_dataset_label1_fine_groups_4.npz',
        allow_pickle=True)
    groups_4 = datasets_train['people']

    #groups 5
    datasets_validate = np.load(
        '/media/csl/5b58b9b5-0162-4611-b882-6351023584d1/csl/methods_data/edge_pyramid_net/datasets/acdc/label1/acdc_dataset_label1_fine_groups_5.npz',
        allow_pickle=True)
    groups_5 = datasets_validate['people']
    groups = []
    groups.append(groups_1)
    groups.append(groups_2)
    groups.append(groups_3)
    groups.append(groups_4)
    groups.append(groups_5)
    main_label(args, groups, 'label1')

