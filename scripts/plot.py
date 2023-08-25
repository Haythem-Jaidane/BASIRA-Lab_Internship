import numpy as np

import matplotlib.pyplot as plt

class Results():

    def __init__(self,evaluations):
        self.evaluations = evaluations

    def plot_loss(self,algo,epoch,save,path):
        fig, ax = plt.subplots(1,self.evaluation.trainer.num_client,figsize=(10,5),sharey=True)
        t = np.arange(1, epoch+1, 1)
        
        for i in range(self.evaluation.trainer.num_client):

            ax[i].plot(t, self.evaluation.trainer.loss[algo]["train"][f"Fold {i+1}"],label="train")
            ax[i].plot(t, self.evaluation.trainer.loss[algo]["validation"][f"Fold {i+1}"],label="validation")
            ax[i].legend(loc="upper right")
            ax[i].set(xlabel='epoch', ylabel='loss value',title=f"Fold {i+1}")

        plt.show()

    def plot_acc(self,algos):
        x = np.arange(3)

        """
        avg = 0

        for i in acc_baseline:
        avg += i
        avg /= 3
        acc_baseline.append(avg)
        """

        fig, ax = plt.subplots(layout='constrained')
        
        for algo in algos:
            ax.bar(x,self.evaluation.acc[algo])

        ax.set_ylabel('accuracy')
        ax.set_title('accuracy on different Folds')
        ax.set_xticks(x, ("Fold 1","Fold 2","Fold 3"))
        ax.set_ylim(0, 100)

        plt.show()

    def plot_sens(self):

        fig, ax = plt.subplots(self.evaluation.trainer.num_client,1,figsize=(8,8),layout='constrained')
        
        x = np.arange(self.evaluation.trainer.datasetObject.num_classes)  # the label locations
        width = 0.4  # the width of the bars
        multiplier = 0


        #for attribute, measurement in penguin_means.items():
        
        for i in range(self.evaluation.trainer.num_client):
            offset = width * i
            ax[i].bar(x,self.evaluation.sens["baseline"][i] )#label=attribute)
            #ax[i].legend(loc='upper left', ncols=3)
            ax[i].set(xlabel='Classes', ylabel='sensitivity',title=f'sensitivity in Fold {i+1}')
            ax[i].set_xticks(x + width/2,(0,1,2,3,4,5,6))


        #multiplier += 1


        plt.show()


    def plot_spec(self):

        fig, ax = plt.subplots(self.evaluation.trainer.num_client,1,figsize=(8,8),layout='constrained')
        
        x = np.arange(self.evaluation.trainer.datasetObject.num_classes)  # the label locations
        width = 0.4  # the width of the bars
        multiplier = 0


        #for attribute, measurement in penguin_means.items():
        
        for i in range(self.evaluation.trainer.num_client):
            offset = width * i
            ax[i].bar(x,self.evaluation.spec["baseline"][i] )#label=attribute)
            #ax[i].legend(loc='upper left', ncols=3)
            ax[i].set(xlabel='Classes', ylabel='specificity',title=f'specificity in Fold {i+1}')
            ax[i].set_xticks(x + width/2,(0,1,2,3,4,5,6))


        #multiplier += 1


        plt.show()