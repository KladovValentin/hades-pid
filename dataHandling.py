import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import glob
import uproot
import pandas
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
import math 
import os


class My_dataset(Dataset):
    def __init__(self, dataTable):
        self.datasetX, self.datasetY = dataTable[0], dataTable[1]

    def __len__(self):
        return len(self.datasetY)

    def __getitem__(self, index):
        return torch.tensor(self.datasetX[index]), torch.tensor(self.datasetY[index])


def load_dataset(dataTable):
    # transform to numpy, assign types, split on features-labels
    df = dataTable
    dfn = df.to_numpy()

    x = dfn[:,:-1].astype(np.float32)
    y = dfn[:, -1].astype(np.float32)

    print('x shape = ' + str(x.shape))
    print('y shape = ' + str(y.shape))
    return (x, y)


class DataManager():
    def __init__(self,datasetType) -> None:
        self.dataSetType = datasetType
        self.poorColumnValues = [('tofdedx',-1)]
        #self.poorColumnValues = []
        self.pidsToSelect = [8,9,11,12,14]
        #self.features = ['momentum','charge','theta','phi','mdcdedx','tofdedx','tof','distmeta','beta','metamatch','mass2']
        #self.features = ['momentum','charge','theta','phi','tofdedx','tof','distmeta','beta','metamatch','mass2']
        #self.features = ['momentum','charge','theta','phi','mdcdedx','tof','distmeta','beta','metamatch','mass2']
        self.features = ['momentum','charge','theta','phi','mdcdedx','tofdedx','tof','distmeta','metamatch','mass2']
    

    def compareInitialDistributions(self):
        dftSim = pandas.read_parquet(os.path.join("nndata",'simu' + self.dataSetType + '.parquet'))
        dftExp = pandas.read_parquet(os.path.join("nndata",'expu' + self.dataSetType + '.parquet'))
        dftSim = dftSim.loc[dftSim['pid'] == 3]
        #dftExp = dftExp.loc[dftExp['pid'] == 4]

        inputsLength = len(dftSim.columns)-1
        class_histS = [dftSim[(list(dftSim.columns)[i])].to_numpy() for i in range(inputsLength)]
        class_histE = [dftExp[(list(dftExp.columns)[i])].to_numpy() for i in range(inputsLength)]

        lowBord = [np.mean(class_histS[i]) - 2 * np.std(class_histS[i]) for i in range(inputsLength)]
        upBord = [np.mean(class_histS[i]) + 2 * np.std(class_histS[i]) + 4*np.std(class_histS[i])/200 for i in range(inputsLength)]
        step = [4*np.std(class_histS[i])/200 for i in range(inputsLength)]
        print(lowBord[0])
        print(upBord[0])
        print(step[0])
        bins = [np.arange(-5, 5, 0.01) for i in range(inputsLength)]

        # Create a 3x3 grid of subplots
        #fig, axes = plt.subplots(3, 4, figsize=(10, 8))

        # Loop through the subplots and plot histograms
        for i in range(inputsLength):
            plt.hist(class_histS[i], bins[i], edgecolor='#0504aa', linewidth=2, fill=False, histtype='step',
                     alpha=0.7, rwidth=1.0, density=True, label = 'sim')
            plt.hist(class_histE[i], bins[i], edgecolor='#9e001a', linewidth=2, fill=False, histtype='step',
                     alpha=0.7, rwidth=1.0, density=True, label = 'exp')
            plt.title(list(dftSim.columns)[i])
            plt.legend()
            plt.show()
        
        #fig.suptitle('"Feature" distributions, 9/128, both train AND validation dataset', fontsize=16)

        #plt.show()


    def prepareTable(self, datF):
        # make datasets equal sizes for each class out of n
        lastName = list(datF.columns)[-1]
        nClasses = datF[lastName].nunique()
        x = [len(datF[(datF[lastName]==i)]) for i in range(nClasses)]
        print("classes lenghts = : " + str(x))
        #minimumCount = np.amin(np.array(x))
        #frames = [datF.loc[datF[lastName] == i].sample(minimumCount) for i in range(nClasses)]
        #print("new classes lenghts = : " + str([len(frames[i]) for i in range(nClasses)]))
        #return pandas.concat(frames).sort_index().reset_index(drop=True)
        return
        #print(datF)
        #return datF.sort_index().reset_index(drop=True)


    def meanAndStdTable(self, dataTable):
        # find mean and std values for each column of the dataset (used for train dataset)
        df = dataTable
        dfn = df.to_numpy()

        x = dfn[:,:].astype(np.float32)
        mean = np.array( [np.mean(x[:,j]) for j in range(x.shape[1]-1)] )
        std  = np.array( [np.std( x[:,j]) for j in range(x.shape[1]-1)] )
        
        #__ if you have bad data sometimes in one of the columns - 
        # - you can calculate mean and std without these bad entries
        #   and then make them = 0 -> no effect on the first layer
        for i in range(len(self.poorColumnValues)):
            cPoor = df.columns.get_loc(self.poorColumnValues[i][0])
            vPoor = self.poorColumnValues[i][1]
            mean[cPoor] = np.mean(x[(x[:,cPoor]!=vPoor),cPoor])
            std[cPoor] = np.std(x[(x[:,cPoor]!=vPoor),cPoor])

        return mean, std


    def normalizeDataset(self, df):
        meanValues, stdValues = self.readTrainData()
        columns = list(df.columns)
        masks = []
        for i in range(len(self.poorColumnValues)):
            masks.append(df[self.poorColumnValues[i][0]]==self.poorColumnValues[i][1])

        for i in range(len(columns)-1):
            if (stdValues[i] == 0):
                df[columns[i]] = 0
            else:
                df[columns[i]] = (df[columns[i]]-meanValues[i])/stdValues[i]
        
        for i in range(len(self.poorColumnValues)):
            df[self.poorColumnValues[i][0]].mask(masks[i], 0, inplace=True)
        return df


    def getDataset(self, rootPath,mod):
        # read data, select raws (pids) and columns (drop)
        #dataPath = rootPath + "data3/pid_data_ascii_random.root:pid"
        #simPath  = rootPath + "sim3/*sim*.root"

        if (mod == "simLabel"):
            rootPath = rootPath + "sim41Gen3/*sim*.root"
        else:
            #rootPath = rootPath + "data41/pid_data_ascii_random.root:pid"
            rootPath = rootPath + "data41/*exp*.root"
        fileC = 0
        batches = []
        for batch in uproot.iterate([rootPath],library="pd"):
            print(fileC)
            if mod == "simLabel":
                batches.append(batch.sample(frac=0.05*1.0).reset_index(drop=True))
            else:
                batches.append(batch.sample(frac=0.05).reset_index(drop=True))
            del batch
            fileC = fileC+1
        #
        setTable = pandas.concat(batches,ignore_index=True).reset_index(drop=True)
        del batches
        

        selection = (setTable['charge']>-10) & (setTable['mass2']>-1.5) & (setTable['mass2']<2.5) & (setTable['momentum']>0.05) & (setTable['momentum']<5) & (setTable['mdcdedx']>0.1) & (setTable['mdcdedx']<50)
        setTable = setTable.loc[selection].copy()
        del selection

        # selecting pids from sim mix and
        if mod == "simLabel":
    
            ttables = []
            for i in range(len(self.pidsToSelect)):
                ttables.append(setTable.loc[setTable['pid']==self.pidsToSelect[i]].copy())
                ttables[i]['pid'] = i
            ttables[2] = ttables[2].sample(frac=0.3).copy() #0.3
            ttables[3] = ttables[3].sample(frac=0.8).copy()
            ttables[4] = ttables[4].sample(frac=0.5).copy()
            try:
                fullSetTable = pandas.concat(ttables, verify_integrity=True).sort_index()
            except ValueError as e:
                print('ValueError', e)
            #fullSetTable = pandas.concat(ttables).sort_index()

            fullSetTableBad = setTable.drop(fullSetTable.index)
            fullSetTableBad['pid'] = len(self.pidsToSelect)
            setTable = pandas.concat([fullSetTable]).sort_index().reset_index(drop=True)

        #setTable['mass2'] = setTable['mass2'].abs()
        
        #dropColumns = ['event_id', 'theta', 'phi']
        dropColumns = ['event_id']
        for drop in dropColumns:
            setTable.drop(drop,axis=1,inplace=True)

        #setTable['mass2'] = math.sqrt(setTable['mass2'])
    
        print(setTable)
        return setTable

    def dropCols(self,table):
        columns = list(table.columns)
        #dropColumns = ['charge','beta','ringcorr']
        dropColumns = []
        for i in range(len(columns)):
            if (columns[i] not in self.features) and (columns[i]!='pid'):
                dropColumns.append(columns[i])
        for drop in dropColumns:
            table.drop(drop,axis=1,inplace=True)
        return table

    def manageDataset(self, mod):
        #self.prepareTable
        dftCorr = self.getDataset("", "simLabel")
        setTable = self.getDataset("", "data")#.sample(frac=1.0).sort_index().reset_index(drop=True)
        #setTable = setTable[[c for c in setTable if c not in ['pid']] + ['pid']]
        #dftCorr = dftCorr[[c for c in dftCorr if c not in ['pid']] + ['pid']]
        print(dftCorr)
        #self.prepareTable(dftCorr)
        dftCorr = self.dropCols(dftCorr)
        setTable = self.dropCols(setTable)
        pq.write_table(pa.Table.from_pandas(dftCorr), os.path.join("nndata",'simu' + self.dataSetType + '.parquet'))
        pq.write_table(pa.Table.from_pandas(setTable), os.path.join("nndata",'expu' + self.dataSetType + '.parquet'))
        print(dftCorr)

        # Find mean and average
        mean, std = 0, 0
        if (mod == "train_dann"):
            mean, std = self.meanAndStdTable(pandas.concat([dftCorr,setTable], ignore_index=True))
        elif (mod == "train_nn"):
            mean, std = self.meanAndStdTable(dftCorr,setTable)
        elif (mod.startswith("test")):
            mean, std = self.readTrainData()  
        if (mod.startswith("train")):
            self.writeTrainData(mean,std)


        # Check the mean and averages
        dftCorr = self.normalizeDataset(dftCorr).copy()
        setTable = self.normalizeDataset(setTable)
        mean1, std1 = self.meanAndStdTable(pandas.concat([dftCorr,setTable], ignore_index=True))
        print(mean1)
        print(std1)

        #pq.write_table(pa.Table.from_pandas(dftCorr), os.path.join("nndata",'simu1' + dataSetType + '.parquet'))
        #pq.write_table(pa.Table.from_pandas(setTable), os.path.join("nndata",'expu1' + dataSetType + '.parquet'))

    
    def writeTrainData(self, meanArr,stdArr):
        np.savetxt(os.path.join("nndata",'meanValues' + self.dataSetType + '.txt'), meanArr, fmt='%s')
        np.savetxt(os.path.join("nndata",'stdValues' + self.dataSetType + '.txt'), stdArr, fmt='%s')

    def readTrainData(self):
        meanValues = np.loadtxt(os.path.join("nndata",'meanValues' + self.dataSetType + '.txt'))
        stdValues = np.loadtxt(os.path.join("nndata",'stdValues' + self.dataSetType + '.txt'))
        return meanValues, stdValues
        




