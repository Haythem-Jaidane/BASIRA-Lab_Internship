import numpy as np

import torch

import torch.utils.data as data

from sklearn import metrics


class evaluation:
    
    def __init__(self,trainer):
        
        self.trainer = trainer
        self.acc = {}
        self.spec = {}
        self.sens = {}
        
        for name in self.trainer.algos:
            self.acc[name] = []
            self.spec[name] = []
            self.sens[name] = []
        
        
    def test(self):

        test_loader = data.DataLoader(data.TensorDataset(torch.tensor(self.trainer.dataset["test"]["image"]).float(), torch.tensor(self.trainer.dataset["test"]["labels"]).long()))


        for name,models in self.trainer.models.items():
            for model in models:

                model.eval()

                predicted_list = []

                for img, labels in test_loader:

                    img = img.permute(0, 3, 1, 2)

                    output = model(img)
                    _, predicted = torch.max(output.data, 1)

                    predicted_list.append(predicted.long().detach().numpy())

                predicted_list = np.array(predicted_list).reshape(len(predicted_list))
                target = torch.tensor(self.trainer.dataset["test"]["labels"]).long().detach().numpy()


                accuracy = metrics.accuracy_score(target, predicted_list)
                precision = metrics.precision_score(target, predicted_list, average=None, zero_division=1)
                recall = metrics.recall_score(target, predicted_list, average=None)

                self.spec[name].append(precision*100)
                self.acc[name].append(accuracy*100)
                self.sens[name].append(recall*100)

