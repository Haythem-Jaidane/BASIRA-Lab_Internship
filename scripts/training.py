import copy
from time import time

from sklearn.model_selection import StratifiedKFold

import torch

import torch.nn as nn

class Training():
    def __init__(self, num_fold, ModelType ,
                 Data, optimizer,type_data,
                 hyperparameter,algos):
        
        self.supported_algorithms = {
                                        "baseline": self.baseline,
                                        "base"    : self.base,
                                        "FedAVG"  : self.FedAVG,
                                        "FedALA"  : self.FedALA
                                    }
        
        if not all(item in self.supported_algorithms.keys() for item in algos):
            raise Exception("algorithms not supported")
        else:
            self.algos = algos
            
        self.KFold = StratifiedKFold(n_splits=num_fold, shuffle=True)
        self.data = Data
        self.model = ModelType
        self.dataset = Data.dataset[type_data]
        self.type_data = type_data
        self.Optimizer = optimizer
        self.num_client = num_fold
        self.models = {}
        self.loss = {}
        self.acc = {}
        self.hyperparameter = hyperparameter
        
        self.device = self.data.device
        
        for name in self.algos:
            self.models[name] = []
            self.loss[name] = {"train":{},"validation":{}}
            self.acc[name] = {"train":{},"validation":{}}
            
        for name,item in self.loss.items():
            for type_name,type in item.items():
                for i in range(num_fold):
                    type[f"Fold {i+1}"] = []

        for name,item in self.acc.items():
            for type_name,type in item.items():
                for i in range(num_fold):
                    type[f"Fold {i+1}"] = []
                    
    
    def trainLoop(self,model,algo,fold,round_,verbos_level,optimizer,criterion,train_loader,val_loader,traininng_type):
        if (traininng_type==0):
            e = self.hyperparameter["epoch_num"]
        elif(traininng_type==1):
            e = self.hyperparameter["epoch_in_round"]
        for epoch in range(e):
            start = time()
            model.train()
            train_loss = 0
            total_train=0
            correct_train = 0
            train_accuracy = 0
            for batch, (X_, y_) in enumerate(train_loader):

                optimizer.zero_grad()
                outputs = model(X_.permute(0, 3, 1, 2)) 
                _, predicted = torch.max(outputs.data, 1)
                total_train += y_.size(0)
                correct_train += (predicted == y_).sum().item()
                loss = criterion(outputs, y_)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_accuracy += 100 * correct_train / total_train
            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc =train_accuracy / len(train_loader)

            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                correct = 0
                total = 0
                for val_batch_X, val_batch_y in val_loader:
                    val_batch_X = val_batch_X.permute(0, 3, 1, 2)
                    val_outputs = model(val_batch_X)
                    _, predicted = torch.max(val_outputs.data, 1)
                    total += val_batch_y.size(0)
                    correct += (predicted == val_batch_y).sum().item()
                    val_loss += criterion(val_outputs, val_batch_y).item()

                val_accuracy = 100 * correct / total
                avg_val_loss = val_loss / len(val_loader)

            print("====================================")
            print(f"Epoch {epoch+1}:")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Train Accuracy: {avg_train_acc:.2f}%")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.2f}%")

            print()
            print(f"time spend : {int(time() - start)} s") 
            print()
            print("====================================")

            self.loss[algo]["train"][f"Fold {fold+1}"].append(avg_train_loss)
            self.loss[algo]["validation"][f"Fold {fold+1}"].append(avg_val_loss)

            self.acc[algo]["train"][f"Fold {fold+1}"].append(avg_train_acc)
            self.acc[algo]["validation"][f"Fold {fold+1}"].append(val_accuracy)
            
            if (traininng_type==0):
                model.save(algo,f"Fold {fold+1}",f"epoch_{epoch+1}")
            elif(traininng_type==1):
                model.save(algo,f"Hospital {fold+1}",f"round_{round_}_epoch_{epoch+1}")
            

    def trainPipline(self):
        for name,fun in self.supported_algorithms.items():
            print("==============================")
            print(name)
            print("==============================")
            fun()

    def baseline(self):
        
        fold_data = self.data.splitData_non_IID(self.KFold,self.type_data,self.hyperparameter["batch_size"])

        for fold in range(len(fold_data)):
            print(f"Fold {fold+1}:")

            model = self.model(self.data)
            model.to(self.device)
            
            model.clean("baseline",f"Fold {fold+1}")

            criterion = nn.CrossEntropyLoss()
            optimizer = self.Optimizer(model.parameters(), lr=self.hyperparameter["lr"], weight_decay=0.0001)

            self.trainLoop(model,"baseline",fold,0,2,optimizer,criterion,fold_data[fold]["train"],fold_data[fold]["validation"],0)


            self.models["baseline"].append(model.to("cpu"))
            
            torch.cuda.empty_cache()

    def base(self):
        
        server_model = self.model(self.data)
        self.models["base"] = [copy.deepcopy(server_model) for _ in range(self.num_client)]


        optimizers = [self.Optimizer(self.models["base"][i].parameters(), lr=self.hyperparameter["lr"],weight_decay=0.0001) for i in range(self.num_client)]

            
        fold_data = self.data.splitData_IID(self.num_client,self.type_data,self.hyperparameter["batch_size"])
            
        for round_ in range(self.hyperparameter["num_periode"]):
            print(f"round {round_+1} : ")
            for fold in range(len(fold_data)):


                print(f"Client {fold+1} : ")
                model = self.models["base"][fold]
                model.to(self.device)

                model.clean("base",f"Hospital {fold+1}")

                optimizer = optimizers[fold]
                criterion = nn.CrossEntropyLoss()
                
                
                self.trainLoop(model,"base",fold,round_,2,optimizer,criterion,fold_data[fold]["train"],fold_data[fold]["validation"],1)
                    
                self.models["base"].append(model.to("cpu"))
                
                optimizers[fold] = optimizer

                torch.cuda.empty_cache()

    

    def FedALA(self):
        pass


    def FedAVG(self):


       server_model = self.model(self.data)
       self.models["FedAVG"] = [copy.deepcopy(server_model) for _ in range(self.num_client)]


       optimizers = [self.Optimizer(self.models["base"][i].parameters(), lr=self.hyperparameter["lr"],weight_decay=0.0001) for i in range(self.num_client)]

       fold_data = fold_data = self.data.splitData_IID(self.num_client,self.type_data,self.hyperparameter["batch_size"])


       for round_ in range(self.hyperparameter["num_periode"]):
           print(f"round {round_+1} : ")
           for fold in range(len(fold_data)):


               print(f"Client {fold+1} : ")
                
               model = self.models["FedAVG"][fold]
               model.to(self.device)

               model.clean("FedAVG",f"Hospital {fold+1}")

               optimizer = optimizers[fold]
               criterion = nn.CrossEntropyLoss()
                
               self.trainLoop(model,"FedAVG",fold,round_,2,optimizer,criterion,fold_data[fold]["train"],fold_data[fold]["validation"],1)

               self.models["FedAVG"].append(model.to("cpu"))
               
               optimizers[fold] = optimizer

           for key in server_model.state_dict().keys():
               if 'num_batches_tracked' in key:
                   server_model.state_dict()[key].data.copy_(self.models["FedAVG"][0].state_dict()[key])
               else:
                   temp = torch.zeros_like(server_model.state_dict()[key])
                   for client_idx in range(self.num_client):
                       temp += (1/self.num_client) * self.models["FedAVG"][client_idx].state_dict()[key]
                   server_model.state_dict()[key].data.copy_(temp)
                   for client_idx in range(self.num_client):
                       self.models["FedAVG"][client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])


