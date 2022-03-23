import os
import numpy as np
import cv2
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils.data.dataloader as DataLoader
from torchsummary import summary

# height, width and frames
image_rows, image_columns, image_depth = 128, 128, 24
VIDEO_LENGTH = 24


CORRECT_NUMBER = 0


training_list = []

# data path
surprisepath = '../data_3/surprise/'
positivepath = '../data_3/positive/'
negativepath = '../data_3/negative/'
video_list = [surprisepath, positivepath, negativepath]


class videoDataset(Dataset):
    def __init__(self, video_path):
        self.training_list = []
        self.traininglabels = []
        for i in video_path:
            self.load_class_video(i)
        self.training_list = np.asarray(self.training_list)
        self.trainingsamples = len(self.training_list)
        # print(self.traininglabels)
        # self.traininglabels = np.zeros((self.trainingsamples,), dtype=int)

        self.traininglabels = self.to_categorical(self.traininglabels, num_classes=3)

        self.training_data = [self.training_list, self.traininglabels]
        (self.trainingframes, self.traininglabels) = (self.training_data[0], self.training_data[1])

        # unsqueeze a dimension for Conv
        self.training_set = np.zeros((self.trainingsamples, 1, image_rows, image_columns, image_depth))
        for h in range(self.trainingsamples):
            self.training_set[h][0][:][:][:] = self.trainingframes[h, :, :, :]

        self.training_set = self.training_set.astype('float32')
        
        # std.
        self.training_set -= np.mean(self.training_set)
        self.training_set /= np.max(self.training_set)

    def __len__(self):
        return len(self.training_list)

    def __getitem__(self, item):
        return self.training_set[item], self.traininglabels[item]

    # label -> one-hot encode
    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]

    # load a video
    def load_video(self, video_path, framelist):
        video_len = len(framelist)
        if video_len < 24:
            print(video_path + " is < 24 frames!")
            return -1

        # video length is set to 24 via sample stride
        sample_time = video_len // VIDEO_LENGTH
        frames = []
        for i in range(VIDEO_LENGTH):
            image = cv2.imread(video_path + '/' + framelist[i * sample_time])
            # resize
            imageresize = cv2.resize(image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
            # ToGrey
            grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
            frames.append(grayimage)
        frames = np.asarray(frames)

        videoarray = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
        return videoarray

    # load video of a class
    def load_class_video(self, video_path):
        directorylisting = os.listdir(video_path)
        for video in directorylisting:
            videopath = video_path + video

            framelist = os.listdir(videopath)
            if "EP" in videopath:
                framelist.sort(key=lambda x:int(x.split('reg_img')[1].split('.jpg')[0]))
            else:
                framelist.sort(key=lambda x:int(x.split('img_')[1].split('.jpg')[0]))
            # framelist.sort()
            # load a video
            videoarray = self.load_video(videopath, framelist)
            if videoarray.all() == -1:
                print("video valid!")
                continue
            self.training_list.append(videoarray)
            # add labels
            if video_path == surprisepath:
                self.traininglabels.append(0);
            elif video_path == positivepath:
                self.traininglabels.append(1);
            elif video_path == negativepath:
                self.traininglabels.append(2);
                # elif video_path == positivepath:
            else:
                continue
                # self.traininglabels.append(3);


class MicroExpSTCNN(nn.Module):
    def __init__(self):
        super().__init__()  
        # self.conv3d = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 4))
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(4, 4, 7), stride=(2, 2, 1))    # 
        self.maxpooling3d = nn.MaxPool3d(kernel_size=(3, 3, 3))
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(in_features=32 * 42 * 42 * 7, out_features=128)
        self.fc1 = nn.Linear(in_features=32 * 21 * 21 * 6, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=3)
       
    def forward(self, x):
        
        x = self.conv3d(x)
        x = self.maxpooling3d(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU(inplace=False)(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train(model, dataset, epoch, optimizer, log_file=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    train_dataloader = DataLoader.DataLoader(dataset, batch_size=8, shuffle=True)
    totalsamples = len(dataset)
    
    for i, item in enumerate(train_dataloader):
        video, label = item

        optimizer.zero_grad()
        
        output = F.log_softmax(model(video.to(device)), dim=1)
        _, label = torch.max(label, dim=1)
        
        loss = F.nll_loss(output, label.to(device))
        _, pred = torch.max(output, dim=1)

        loss.backward()
        optimizer.step()

        if i % 10 == 0: 

            pred = np.array(pred.cpu())
            label = np.array(label)
            print('Epoch:', epoch)
            '''
            print("pred:[" + ','.join('%s' % j for j in pred.tolist()) + ']   label:[' + ','.join(
                '%s' % j for j in label.tolist()) + ']')
            '''
            print('[' + '{:5}'.format(i * 8) + '/' + '{:5}'.format(totalsamples) +
                  ' (' + '{:3.0f}'.format(100 * i * 8 / totalsamples) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))


def evaluate(model, epoch, dataset, version, log_file=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    correct_samples = 0
    total_loss = 0

    global CORRECT_NUMBER

    test_dataloader = DataLoader.DataLoader(dataset, batch_size=1)
    totalsamples = len(dataset)
    with torch.no_grad():
        for i, item in enumerate(test_dataloader):
            video, label = item
            output = F.log_softmax(model(video.to(device)), dim=1)
            _, label = torch.max(label, dim=1)
            loss = F.nll_loss(output, label.to(device))
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()

            correct_samples += pred.cpu().eq(label).sum()
            if i % 1 == 0:
                pred = np.array(pred.cpu())
                label = np.array(label)
                avg_loss = total_loss / totalsamples
                '''
                print("Epoch:", epoch)
                print("pred:[" + ','.join('%s' % j for j in pred.tolist()) + ']   label:[' + ','.join(
                    '%s' % j for j in label.tolist()) + ']')
                '''
        loss_info = 'Epoch:{}'.format(epoch) + '\nAverage test loss: ' + '{:.4f}'.format(avg_loss) \
                    + '  Accuracy:' + '{:5}'.format(correct_samples) + '/' \
                    + '{:5}'.format(totalsamples) + ' (' \
                    + '{:4.2f}'.format(100.0 * correct_samples / totalsamples) + '%)\n' # 正确率
        log_file.writelines(loss_info)
        print(loss_info)

        # save model
        if correct_samples > CORRECT_NUMBER:
            CORRECT_NUMBER = correct_samples
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,},
                       'saved_models_3/{}_{}.pth'.format(version, correct_samples / totalsamples))


if __name__ == "__main__":
    version = '03221558_(4,4,7,2)_pic128' # model version

    NEPOCH = 100
    dataset = videoDataset(video_list)
    dataset_size = dataset.__len__()
    # print(dataset_size)

    # train:test = 8:2
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # np.save('numpy_training_datasets/microexpstcnn_torch.npy', train_dataset)
    test_log_file = open(version + '_test_log.txt', 'w')
    # test_log_file.writelines('conv1(x, y)...') # model info

    MER = MicroExpSTCNN()
    # model summary
    summary(MER, (1, 128, 128, 24), batch_size=16, device="cpu") #gpu


    optimizer = torch.optim.SGD(params=MER.parameters(), lr=0.006, momentum=0.9, weight_decay=0.0005, nesterov=True)

    
    for i in range(1, NEPOCH + 1):
        train(model=MER, epoch=i, dataset=train_dataset, optimizer=optimizer, log_file=None)
        evaluate(model=MER, epoch=i, dataset=test_dataset, log_file=test_log_file, version=version)







