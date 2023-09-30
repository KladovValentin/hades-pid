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
    def __init__(self) -> None:
        self.poorColumnValues = [('tofdedx',-1)]
        self.pidsToSelect = [8,9,11,12,14]
        self.features = ['momentum','charge','theta','mdcdedx','tofdedx','tof','distmeta','beta','metamatch','mass2']
    

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


    def normalizeDataset(self, df, meanValues, stdValues):
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
            rootPath = rootPath + "sim4/*sim*.root"
        else:
            rootPath = rootPath + "data4/pid_data_ascii_random.root:pid"
        fileC = 0
        batches = []
        for batch in uproot.iterate([rootPath],library="pd"):
            print(fileC)
            if mod == "simLabel":
                batches.append(batch.sample(frac=0.4).reset_index(drop=True))
            else:
                batches.append(batch)
            del batch
            fileC = fileC+1
        #
        setTable = pandas.concat(batches,ignore_index=True).reset_index(drop=True)
        del batches
        
        """
        tree = uproot.open(dataPath)
        features = tree.keys()
        sim = {key: [] for _, key in enumerate(features)}
        if mod == "simLabel":
            for feature in features:
                MAX_EVENTS = 0
                for f in glob.glob(simPath):
                    t = uproot.open(f+":pid")
                    if feature == "event_id":
                        sim[feature].append(t[feature].array().to_numpy()+MAX_EVENTS)
                        MAX_EVENTS += np.max(t[feature].array().to_numpy())
                    else:
                        sim[feature].append(t[feature].array().to_numpy())
        else:
            for feature in features:
                sim[feature].append(tree[feature].array().to_numpy())

        sim = {key: np.hstack(array) for key, array in sim.items()}
        setTable  = pandas.DataFrame(data=np.vstack(list(sim.values())).T, columns=features)
        del sim,tree
        """

        selection = (setTable['charge']>-10) & (setTable['mass2']>-1.5) & (setTable['mass2']<2.5) & (setTable['momentum']>0.05) & (setTable['momentum']<5) & (setTable['mdcdedx']>0.1) & (setTable['mdcdedx']<50)
        setTable = setTable.loc[selection].copy()
        del selection

        # selecting pids from sim mix and
        if mod == "simLabel":
    
            ttables = []
            for i in range(len(self.pidsToSelect)):
                ttables.append(setTable.loc[setTable['pid']==self.pidsToSelect[i]].copy())
                ttables[i]['pid'] = i
            ttables[2] = ttables[2].sample(frac=0.5).copy()
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

    def manageDataset(self, mod, dataSetType):
        #self.prepareTable
        dftCorr = self.getDataset("", "simLabel")
        setTable = self.getDataset("", "data").sample(frac=1.0).sort_index().reset_index(drop=True)
        setTable = setTable[[c for c in setTable if c not in ['pid']] + ['pid']]
        dftCorr = dftCorr[[c for c in dftCorr if c not in ['pid']] + ['pid']]
        print("ASDSADSADASDSADASDSADASDSADASDASDSA")
        print(dftCorr)
        #self.prepareTable(dftCorr)
        pq.write_table(pa.Table.from_pandas(dftCorr), os.path.join("nndata",'simu' + dataSetType + '.parquet'))
        pq.write_table(pa.Table.from_pandas(setTable), os.path.join("nndata",'expu' + dataSetType + '.parquet'))
        dftCorr = self.dropCols(dftCorr)
        setTable = self.dropCols(setTable)
        print(dftCorr)
        mean, std = 0, 0
        if (mod == "train_dann"):
            mean, std = self.meanAndStdTable(pandas.concat([dftCorr,setTable], ignore_index=True))
        elif (mod == "train_nn"):
            mean, std = self.meanAndStdTable(dftCorr,setTable)
        elif (mod.startswith("test")):
            mean, std = readTrainData(dataSetType)

        dftCorr = self.normalizeDataset(dftCorr,mean,std).copy()
        setTable = self.normalizeDataset(setTable,mean,std)

        mean1, std1 = self.meanAndStdTable(pandas.concat([dftCorr,setTable], ignore_index=True))
        print(mean1)
        print(std1)

        pq.write_table(pa.Table.from_pandas(dftCorr), os.path.join("nndata",'simu1' + dataSetType + '.parquet'))
        pq.write_table(pa.Table.from_pandas(setTable), os.path.join("nndata",'expu1' + dataSetType + '.parquet'))

        if (mod.startswith("train")):
            writeTrainData(mean,std, dataSetType)
        



def writeTrainData(meanArr,stdArr, dataSetType):
    np.savetxt(os.path.join("nndata",'meanValues' + dataSetType + '.txt'), meanArr, fmt='%s')
    np.savetxt(os.path.join("nndata",'stdValues' + dataSetType + '.txt'), stdArr, fmt='%s')

def readTrainData(dataSetType):
    meanValues = np.loadtxt(os.path.join("nndata",'meanValues' + dataSetType + '.txt'))
    stdValues = np.loadtxt(os.path.join("nndata",'stdValues' + dataSetType + '.txt'))
    return meanValues, stdValues

