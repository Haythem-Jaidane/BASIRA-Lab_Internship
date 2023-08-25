import numpy as np

import pandas as pd

from imblearn.over_sampling import RandomOverSampler

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

import copy
import os

import torch

import torch.utils.data as data
import torchvision.transforms as transforms

from medmnist import INFO

class DataProcessing():
    
    def __init__(self,MedMNIST_object,DEVICE):

        self.device = DEVICE
        self.dataset_name = MedMNIST_object.flag

        os.makedirs("../plots/"+self.dataset_name+"/raw",exist_ok=True)
        os.makedirs("../data/"+self.dataset_name+"/raw",exist_ok=True)
        
        self.train_object = MedMNIST_object(root='../data/'+self.dataset_name+'/raw/', transform=transforms.ToTensor(), download=True, split='train')
        self.test_object = MedMNIST_object(root='../data/'+self.dataset_name+'/raw/', transform=transforms.ToTensor(), download=True, split='test')
        
        train_label = []
        test_label = []
        
        for label in self.train_object.labels:
            train_label.append(label[0])
            
        for label in self.test_object.labels:
            test_label.append(label[0])
            
        self.num_channels = INFO[self.dataset_name]["n_channels"]
        self.classes_name = INFO[self.dataset_name]["label"]
        self.num_classes = len(self.classes_name)
        self.dataset = {
            "original":{
                "train":{
                    "image": torch.from_numpy(self.train_object.imgs),
                    "labels" : torch.Tensor(train_label),
                    "labels_distribution" : torch.zeros(self.num_classes)
                },
                "test":{  
                    "image": torch.from_numpy(self.test_object.imgs) , 
                    "labels" : torch.Tensor(test_label) , 
                    "labels_distribution" : torch.zeros(self.num_classes)
                }
            }
        }
        self.calculateLabelsDistribution("original")

    def __del__(self):
        del self.dataset_name
        del self.classes_name
        self.dataset = {}
        del self.dataset
        del self.num_classes
        del self.train_object
        del self.test_object
        del self
        
    def __str__(self):
        return f"==============================\n" \
               f"dataset name : {self.dataset_name}\n" \
               f"==============================\n" \
               f"Train Dataset description\n" \
               f"==============================\n" \
               f"{self.train_object}\n" \
               f"==============================\n" \
               f"Test Dataset description\n" \
               f"==============================\n" \
               f"{self.test_object}\n"
    
    def __len__(self):
        print(f"you have {len(self.dataset)} dataset")
        return len(self.dataset)
    
    def splitData_non_IID(self,KFold,type_data,batch_size):
        
        loader_list = []
        
        for train_index, val_index in KFold.split(self.dataset[type_data]["train"]["image"], self.dataset[type_data]["train"]["labels"]):

            X_train, X_val = self.dataset[type_data]["train"]["image"][train_index], self.dataset[type_data]["train"]["image"][val_index]
            y_train, y_val = self.dataset[type_data]["train"]["labels"][train_index], self.dataset[type_data]["train"]["labels"][val_index]

            X_train, y_train = torch.tensor(X_train).float().to(self.device), torch.tensor(y_train).long().to(self.device)
            X_val, y_val = torch.tensor(X_val).float().to(self.device), torch.tensor(y_val).long().to(self.device)


            train_dataset = data.TensorDataset(X_train, y_train)
            val_dataset = data.TensorDataset(X_val,y_val)

            loader_list.append(
                {
                    "train":data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                    "validation":data.DataLoader(val_dataset)
                }
            )
            
        return loader_list
            
            
    def splitData_IID(self,num_client,type_data,batch_size):
        f = 0
        loader_list = []
        
        for train_index, val_index in StratifiedKFold(n_splits=num_client+1, shuffle=True).split(self.dataset[type_data]["train"]["image"], self.dataset[type_data]["train"]["labels"]):
        
            f += 1

            X_val = np.array(self.dataset[type_data]["train"]["image"])[train_index]
            y_val = np.array(self.dataset[type_data]["train"]["labels"])[train_index]



            X_val, y_val = torch.tensor(X_val).float().to(self.device), torch.tensor(y_val).long().to(self.device)


            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

            if ( f != num_client+1):

                loader_list.append(
                    {
                        'train': data.DataLoader(val_dataset, batch_size=batch_size,shuffle=True),
                        'validation':"" 
                    }
                )

            else:
                for i in range(num_client):
                    loader_list[i]["validation"] = data.DataLoader(val_dataset)
                    
        return loader_list
    
    def viewDataset(self,save,filename):
        
        fig,ax = plt.subplots(1,2,figsize=(10,10))
        ax[0].imshow(self.train_object.montage(length=20))
        ax[1].imshow(self.test_object.montage(length=20))
        
        ax[0].set_title("train dataset")
        ax[1].set_title("test dataset")
        
        ax[0].axis('off')
        ax[1].axis('off')
        plt.show()

        if save:
            dir_plot = "../plots/"+self.dataset_name+"/raw/"
            if not os.path.exists(dir_plot):
                os.mkdir(dir_plot)
            fig.savefig(dir_plot+filename+".png")
        
    def viewNSimple(self,n,height,width,dataset_type,save,filename):
        
        if height*width<n:
            raise Exception("the figure size is too small for showing your simples")
            
        fig,ax = plt.subplots(height,width,figsize=(25,25))

        index_simple=0

        for i in range(height):
            for j in range(width):

                image = self.dataset[dataset_type]["train"]["image"][index_simple,:,:,:]
                
                class_name = self.classes_name[str(int(self.dataset[dataset_type]["train"]["labels"][index_simple].item()))]

                ax[i,j].imshow(image)
                ax[i,j].axis('off')
                ax[i,j].set_title(class_name)

                index_simple+=1

        fig.suptitle(str(n)+" simples of the train dataset")
        plt.show()

        if save:
            dir_plot = "../plots/"+self.dataset_name+"/"+dataset_type+"/"
            if not os.path.exists(dir_plot):
                os.mkdir(dir_plot)
            fig.savefig(dir_plot+filename+".png")
        
    def calculateLabelsDistribution(self,dataset_type):
        
        self.dataset[dataset_type]["train"]["labels_distribution"] = torch.zeros(self.num_classes)
        self.dataset[dataset_type]["test"]["labels_distribution"]  = torch.zeros(self.num_classes)
        
        for label in self.dataset[dataset_type]["train"]["labels"]:
            self.dataset[dataset_type]["train"]["labels_distribution"][int(label.item())]+=1
            
        for label in self.dataset[dataset_type]["test"]["labels"]:
            self.dataset[dataset_type]["test"]["labels_distribution"][int(label.item())]+=1
    
    def plotLabelsDistribution(self,dataset_type,save,filename):
        
        self.calculateLabelsDistribution(dataset_type)
        
        classes_list = list(self.classes_name.values())


        train_distribution = pd.DataFrame({'class':classes_list,'simples number':self.dataset[dataset_type]["train"]["labels_distribution"]})
        train_distribution.set_index('class', inplace=True)


        test_distribution = pd.DataFrame({'class':classes_list,'simples number':self.dataset[dataset_type]["test"]["labels_distribution"]})
        test_distribution.set_index('class', inplace=True)

        fig,ax = plt.subplots(1,2,figsize=(10,10),sharey=True)

        train_distribution.plot.bar(ax=ax[0])
        test_distribution.plot.bar(ax=ax[1])

        fig.suptitle("distribution of the dataset")
        plt.show()

        if save:
            dir_plot = "../plots/"+self.dataset_name+"/"+dataset_type+"/"
            if not os.path.exists(dir_plot):
                os.mkdir(dir_plot)
            fig.savefig(dir_plot+filename+".png")
        
    def cleanImblanceData(self,from_data,new_dataset_type):
        
        self.dataset[new_dataset_type] = copy.deepcopy(self.dataset[from_data])
        
        X_train = self.dataset[new_dataset_type]["train"]["image"].reshape(self.dataset[new_dataset_type]["train"]["image"].shape[0], -1)
        X_test = self.dataset[new_dataset_type]["test"]["image"].reshape(self.dataset[new_dataset_type]["test"]["image"].shape[0], -1)

        oversampler_train = RandomOverSampler()
        X_train,self.dataset[new_dataset_type]["train"]["labels"] = oversampler_train.fit_resample(X_train, self.dataset["original"]["train"]["labels"])

        oversampler_test = RandomOverSampler()
        X_test,self.dataset[new_dataset_type]["test"]["labels"] = oversampler_test.fit_resample(X_test,self.dataset["original"]["test"]["labels"])
        
        self.dataset[new_dataset_type]["train"]["image"] = X_train.reshape(X_train.shape[0],28,28,3)
        self.dataset[new_dataset_type]["test"]["image"] = X_test.reshape(X_test.shape[0],28,28,3)
        
    def plotColorDistribution(self,dataset_type,save,filename):
        
        image_reshaped_train = self.dataset[dataset_type]["train"]["image"].reshape(-1, 3)
        image_reshaped_test = self.dataset[dataset_type]["test"]["image"].reshape(-1, 3)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4),sharey=True)
        fig.suptitle('Color Distribution')
        
        colors = ["Red","Green","Blue"]
        
        for i in range(3):
            axes[i].hist(image_reshaped_train[:, i], bins=256,label="train")
            axes[i].hist(image_reshaped_test[:, i], bins=256,label="test")
            axes[i].legend(loc="upper left")
            axes[i].set_title(colors[i])
            axes[i].set_xlabel('Color Intensity')
            axes[i].set_ylabel('Frequency')


        plt.tight_layout()

        if save:
            dir_plot = "../plots/"+self.dataset_name+"/"+dataset_type+"/"
            if not os.path.exists(dir_plot):
                os.mkdir(dir_plot)
            fig.savefig(dir_plot+filename+".png")
    
    def makeDomainshift(self,from_dataset,filter_kernal,new_type):
        
        self.dataset[new_type] = copy.deepcopy(self.dataset[from_dataset])

        self.dataset[new_type]["test"]["image"][:,:, :, 0] = self.dataset[new_type]["test"]["image"][:,:, :, 0] * filter_kernal[0]
        self.dataset[new_type]["test"]["image"][:,:, :, 1] = self.dataset[new_type]["test"]["image"][:,:, :, 1] * filter_kernal[1]
        self.dataset[new_type]["test"]["image"][:,:, :, 2] = self.dataset[new_type]["test"]["image"][:,:, :, 2] * filter_kernal[2]
        
    def save_dataset(self):
        for dataset_type,dataset in self.dataset.items():
            dir_data = "../data/"+self.dataset_name+"/"+dataset_type+"/"
            os.makedirs(dir_data,exist_ok=True)
            np.savez(dir_data+"/"+dataset_type+'.npz', 
                     train_image = dataset["train"]["image"],
                     train_labels= dataset["train"]["labels"],
                     test_image  = dataset["test"]["image"],
                     test_labels = dataset["test"]["labels"])