import numpy as np
import sys, os
import time
import matplotlib.pyplot as plt
import textdistance
sys.path.append(os.getcwd())


# crnn packages
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn
import alphabets
str1 = alphabets.chinese_3000

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, default='test_images/test5.jpg', help='the path to your images')
opt = parser.parse_args()
validation_pth = 'to_lmdb/test_width'
# crnn params
crnn_model_path = 'trained_models/crnn_Rec_done_69_5000.pth'
# crnn_model_path = 'trained_models/TrainedModuel_from_smaller_dataset/crnn_Rec_done_299_650.pth'
alphabet = str1
nclass = len(alphabet)+1
Batch_Test_Flag = True

path_prefix = 'to_lmdb/test_width/'
#path_prefix = 'C:\\Users\\Nuo Xu\\Desktop\\WORK\\Intern\\Solution1\\TestDataset2\\'

f = open('to_lmdb/test_width.txt', "r", encoding='utf-8')
#f = open('C:\\Users\\Nuo Xu\\Desktop\\WORK\\Intern\\Solution1\\TestDataset2.txt', "r", encoding='utf-8')
lines = f.readlines()
f.close()




def get_image_path_and_labels(dir, path_prefix):
    label_dict = {}
    for i in range(len(lines)):
        line_text_1 = lines[i]
        imgPth = os.path.join(path_prefix,line_text_1.split()[0])
        label  = line_text_1.split()[-1]
        label_dict.update({imgPth:label})
    
    return label_dict

tesing_dataset = get_image_path_and_labels(validation_pth, path_prefix)
# crnn文本信息识别
def crnn_recognition(imgpth, model, tesing_dataset, total_correct_num, total_string_length):
    cropped_image = Image.open(imgpth)

    converter = utils.strLabelConverter(alphabet)
  
    image = cropped_image.convert('L')

    ## 
    w  = int(image.size[0] / (280 * 1.0 / 200))
    # w = int(image.size[0] / (32 * 1.0 / image.size[1]))
    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    ground_truth = tesing_dataset.get(imgpth)
    correct_num = int(len(ground_truth) * textdistance.levenshtein.normalized_similarity(ground_truth, sim_pred))
    string_length = len(ground_truth)
    #check = ground_truth == sim_pred
    print('results: {0},  gt: {1}'.format(sim_pred, ground_truth))
    return correct_num, string_length

def crnn_single_test(cropped_image, model):

    converter = utils.strLabelConverter(alphabet)
  
    image = cropped_image.convert('L')

    ## 
    # w  = int(image.size[0] / (280 * 1.0 / 180))
    # w = image.size[0]
    w = int(image.size[0] / (32 * 1.0 / image.size[1]))
    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('results: {0}'.format(sim_pred))


if __name__ == '__main__':

	# crnn network
    model = crnn.CRNN(32, 1, nclass, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path))
    
    started = time.time()
    ## read an image
    # image = Image.open(opt.images_path)
    if Batch_Test_Flag:
        total_correct_num   = 0
        total_string_length = 0
        for stp in range(len(tesing_dataset)):
            imgpth = list(tesing_dataset.keys())[stp]
            # image = Image.open(imgpth)
            correct_num, string_length = crnn_recognition(imgpth, model, tesing_dataset, total_correct_num, total_string_length)
            total_correct_num   += correct_num
            total_string_length += string_length
        finished = time.time()
        print('elapsed time: {0}'.format(finished-started))
        print('Correct Rate: {0}'.format(total_correct_num / total_string_length))
    else:
        image = Image.open(opt.images_path)

        crnn_single_test(image, model)
        finished = time.time()
        print('elapsed time: {0}'.format(finished-started))
    