import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
# from model import Model
from model_hook import Model
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


class iCaRLmodel:

    def __init__(self, numclass, backbone, batch_size, task_size, memory_size, epochs, learning_rate):

        super(iCaRLmodel, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = Model(numclass, backbone)
        self.exemplar_set = []
        self.class_mean_set = []
        self.numclass = numclass
        self.transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.old_model = None

        self.train_transform = transforms.Compose([  # transforms.Resize(img_size),
                                                  transforms.RandomCrop(
                                                      (32, 32), padding=4),
                                                  transforms.RandomHorizontalFlip(
                                                      p=0.5),
                                                  transforms.ColorJitter(
                                                      brightness=0.24705882352941178),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.test_transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.classify_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                      # transforms.Resize(img_size),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.train_dataset = iCIFAR100(
            'dataset', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100(
            'dataset', test_transform=self.test_transform, train=False, download=True)

        self.batchsize = batch_size
        self.memory_size = memory_size
        self.task_size = task_size

        self.train_loader = None
        self.test_loader = None
        self.old_importance = None
    # get incremental train data
    # incremental
    def beforeTrain(self):
        self.model.eval()
        classes = [self.numclass-self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(
            classes)
        if self.numclass > self.task_size:
            self.model.Incremental_learning(self.numclass)
        self.model.train()
        self.model.to(device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader

    '''
    def _get_old_model_output(self, dataloader):
        x = {}
        for step, (indexs, imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                old_model_output = torch.sigmoid(self.old_model(imgs))
            for i in range(len(indexs)):
                x[indexs[i].item()] = old_model_output[i].cpu().numpy()
        return x
    '''

    # train model
    # compute loss
    # evaluate model
    def train(self):
        accuracy = 0
        opt = optim.SGD(self.model.parameters(),
                        lr=self.learning_rate, weight_decay=0.00001)

        self.model.feature.start_cal_importance()
        for epoch in range(self.epochs):
            if epoch == 48:
                if self.numclass == self.task_size:
                    print(1)
                    opt = optim.SGD(self.model.parameters(),
                                    lr=1.0/5, weight_decay=0.00001)
                else:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 5
                    #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 5,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                print("change learning rate:%.3f" % (self.learning_rate / 5))
            elif epoch == 62:
                if self.numclass > self.task_size:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 25
                    #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 25,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                else:
                    opt = optim.SGD(self.model.parameters(),
                                    lr=1.0/25, weight_decay=0.00001)
                print("change learning rate:%.3f" % (self.learning_rate / 25))
            elif epoch == 80:
                if self.numclass == self.task_size:
                    opt = optim.SGD(self.model.parameters(),
                                    lr=1.0 / 125, weight_decay=0.00001)
                else:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 125
                    #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                print("change learning rate:%.3f" % (self.learning_rate / 100))


            self.model.feature.reset_importance()
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(device), target.to(device)

                ce_loss, dist_loss = self._compute_loss_with_hook(indexs, images, target)
                loss_value = ce_loss + dist_loss / 10000.0

                # loss_value = self._compute_loss(indexs, images, target)
                # ce_loss = torch.zeros(1)
                # dist_loss = torch.zeros(1)

                opt.zero_grad()
                loss_value.backward()
                opt.step()

                print(f"epoch: {epoch:3d}, step {step:3d}, ce_loss {ce_loss.item():6.3f}, dist_loss {dist_loss.item():6.3f}, total_loss {loss_value.item():6.3f}")

            accuracy = self._test(self.test_loader, 1)
            print('epoch:%d,accuracy:%.3f' % (epoch, accuracy))

        self.old_importance = self.model.feature.get_importance()
        self.old_importance = [(x / x.mean()).detach() for x in self.old_importance]
        self.model.feature.stop_cal_importance()
        return accuracy

    def _test(self, testloader, mode):
        if mode == 0:
            print("compute NMS")
        self.model.eval()
        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(
                    imgs) if mode == 1 else self.classify(imgs)
            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        self.model.train()
        return accuracy

    def _compute_loss(self, indexs, imgs, target):
        output = self.model(imgs)
        target = get_one_hot(target, self.numclass)
        output, target = output.to(device), target.to(device)
        if self.old_model == None:
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            old_target = torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return F.binary_cross_entropy_with_logits(output, target)

    def _compute_loss_with_hook(self, indexs, imgs, target, importance=None):
        one_hot_label = get_one_hot(target, self.numclass)
        if self.old_model is None:
            output, _, _ = self.model.forward_with_hook(imgs)
            return F.binary_cross_entropy_with_logits(output, one_hot_label), torch.zeros(1).to(device)

        output_old, features_old, _ = self.old_model.forward_with_hook(imgs)
        features_old = [x.detach() for x in features_old]

        output, features, _ = self.model.forward_with_hook(imgs)
        old_target = torch.sigmoid(output_old.detach())
        old_task_size = old_target.shape[1]
        one_hot_label[..., :old_task_size] = old_target
        ce_loss = F.binary_cross_entropy_with_logits(output, one_hot_label)

        B = output.size(0)
        dist_loss = [z.view(1, -1) * torch.norm(x.view(B, x.size(1), -1) - y.view(B, y.size(1), -1), p='fro', dim=(-1))
            for x, y, z in zip(features, features_old, self.old_importance)]
        dist_loss = sum([x.mean(dim=0).sum() for x in dist_loss])
        return ce_loss, dist_loss



    # change the size of examplar
    def afterTrain(self, accuracy):
        self.model.eval()
        m = int(self.memory_size/self.numclass)
        self._reduce_exemplar_sets(m)
        for i in range(self.numclass-self.task_size, self.numclass):
            print('construct class %s examplar:' % (i), end='')
            images = self.train_dataset.get_image_class(i)
            self._construct_exemplar_set(images, m)
        self.numclass += self.task_size
        self.compute_exemplar_class_mean()
        self.model.train()
        KNN_accuracy = self._test(self.test_loader, 0)
        print("NMS accuracy："+str(KNN_accuracy.item()))
        filename = 'model_with_hook/accuracy:%.3f_KNN_accuracy:%.3f_increment:%d_net.pkl' % (
            accuracy, KNN_accuracy, i + 10)
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(device)
        self.old_model.eval()

    def _construct_exemplar_set(self, images, m):
        class_mean, feature_extractor_output = self.compute_class_mean(
            images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))

        for i in range(m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean +
                              feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        print("the size of exemplar :%s" % (str(len(exemplar))))
        self.exemplar_set.append(exemplar)
        # self.exemplar_set.append(images)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('Size of class %d examplar: %s' %
                  (index, str(len(self.exemplar_set[index]))))

    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(
                Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):
        x = self.Image_transform(images, transform).to(device)
        feature_extractor_output = F.normalize(
            self.model.feature_extractor(x).detach()).cpu().numpy()
        #feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            print("compute the class mean of %s" % (str(index)))
            exemplar = self.exemplar_set[index]
            # exemplar=self.train_dataset.get_image_class(index)
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_, _ = self.compute_class_mean(
                exemplar, self.classify_transform)
            class_mean = (class_mean/np.linalg.norm(class_mean) +
                          class_mean_/np.linalg.norm(class_mean_))/2
            self.class_mean_set.append(class_mean)

    def classify(self, test):
        result = []
        test = F.normalize(self.model.feature_extractor(
            test).detach()).cpu().numpy()
        #test = self.model.feature_extractor(test).detach().cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        for target in test:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)
