
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from models.model import DANN
from models.model import Model
from dataHandling import My_dataset, DataManager, load_dataset
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from tqdm.auto import tqdm
from tqdm import trange
import math
import os


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

dataSetType = 'NewKIsUsed'
dataManager = DataManager(dataSetType)


def loadModel(input_dim, nClasses):
    nn_model = DANN(input_dim=input_dim, output_dim=nClasses).type(torch.FloatTensor)
    nn_model.type(torch.FloatTensor)
    nn_model.load_state_dict(torch.load(os.path.join('nndata','tempModel' + dataSetType + '.pt')))
    #nn_model.eval()
    return nn_model.to(device)

def constructPredictionFrame(index,nparray):
    # return new DataFrame with 4 columns corresponding to probabilities of resulting classes and 5th column to chi2 
    df2 = pandas.DataFrame(index=index)
    for i in range(nparray.shape[1]):
        df2[str(i)] = nparray[:i].tolist()
    return df2


def checkDistributions():
    dfm = pandas.read_table("testSet" + "Mod" + ".txt",sep='	',header=None)
    dfe = pandas.read_table("testSet" + "Exp" + ".txt",sep='	',header=None)
    numberOfColumns = len(dfm.columns)
    columnNames = list(dfe.columns)
    dfe[(dfe[columnNames[10]]==1000.0) & (dfe[columnNames[0]]<0.35)][columnNames[2]].plot(kind='kde')
    dfm[(dfm[columnNames[10]]==1000.0) & (dfm[columnNames[0]]<0.35)][columnNames[2]].plot(kind='kde')
    plt.show()


def makePredicionList(experiment_path, savePath):
    dftCorrExp = pandas.read_parquet(os.path.join("nndata",experiment_path))

    print(dftCorrExp.iloc[0])
    
    dftCorrExp = dataManager.normalizeDataset(dftCorrExp)

    print(dftCorrExp.iloc[0])

    #class1p = dftCorrExp.loc[(dftCorrExp['pid']==1)].copy()
    #restt = dftCorrExp.drop(class1p.index)
    #class1p = class1p.sample(frac=0.05).copy()
    #print(class1p)
    #dftCorrExp = pandas.concat([class1p,restt]).sort_index()
    #print(dftCorrExp)

    exp_dataset = My_dataset(load_dataset(dftCorrExp))
    exp_dataLoader = DataLoader(exp_dataset, batch_size=512, drop_last=False)

    #nClasses = dftCorrExp[list(dftCorrExp.columns)[-1]].nunique()
    nClasses = 5
    input_dim = exp_dataset[0][0].shape[0]

    #load nn and predict
    nn_model = loadModel(input_dim, nClasses)
    nn_model.eval()
    
    #inputTens = torch.tensor(np.array([exp_dataset[0][0],exp_dataset[1][0]])).to(device)
    #print(inputTens)
    #print(nn_model(inputTens)[0].softmax(dim=1).detach().cpu().numpy())

    dat_list = []
    exp_iter = iter(exp_dataLoader)
    tepoch = tqdm(range(len(exp_dataLoader)), total=len(exp_dataLoader))
    #for i in range(len(exp_dataLoader)):
    for i_step in tepoch:
        tepoch.set_description(f"Epoch {1}")
        (ve_x, _) = next(exp_iter)
        ve_x = ve_x.to(device)
        #inputArr = np.array([[-0.424882, -1.417659, -1.180430,  0.007431, -0.053383,  0.000000,  0.086087,  0.376829, -0.008964,   0.091145, -0.346358],
        #                     [-0.169018, -1.417659,  2.259818,  1.890011,  0.408186,  0.125931, -0.047164,  0.560324, -0.004500,  -0.314674, -0.249043]]).astype(np.float32)
        #inputTens = torch.tensor(inputArr)
        #print(nn_model(inputTens)[0].softmax(dim=1).detach().cpu().numpy())
        e_class, e_domain = nn_model(ve_x)
        e_class = e_class.softmax(dim=1).detach().cpu().numpy()
        #e_feature = e_feature[:,0:-1].detach().cpu().numpy()
        #e_class = np.concatenate((e_class,e_feature),axis=1)
        dat_list.append(pandas.DataFrame(e_class))

    fullPredictionList = pandas.concat(list(dat_list),ignore_index=True)
    pq.write_table(pa.Table.from_pandas(fullPredictionList), os.path.join("nndata",savePath))
    return fullPredictionList


def draw_probabilities_spread(outputs,selected):
    class_hist = outputs.to_numpy()
    class_hist_selected = selected.to_numpy()

    bins = np.linspace(0, 1, 20)

    plt.hist(class_hist_selected[:,2], bins, color='#0504aa',
                            alpha=0.7, rwidth=0.95, label = '$\pi$')
    plt.hist(class_hist_selected[:,5], bins, color='#228B22',
                            alpha=0.7, rwidth=0.95, label = '$P$')
    plt.hist(class_hist_selected[:,4], bins, color='#aa0404',
                            alpha=0.7, rwidth=0.95, label = '$K$')
    plt.legend(loc=[0.6,0.8])
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('probability')
    plt.show()

def draw_probabilities_vs_parameter(probs, tables, column):
    #x =[np.sqrt(np.absolute(tables[0][column].to_numpy())),
    #    np.sqrt(np.absolute(tables[2][column].to_numpy())),
    #    np.sqrt(np.absolute(tables[4][column].to_numpy()))]
    
    x =[tables[0][column].to_numpy(),
        tables[2][column].to_numpy(),
        tables[4][column].to_numpy()]

    cN = list(probs[0].columns)
    y = [probs[0][cN[0]].to_numpy(),
         probs[2][cN[2]].to_numpy(),
         probs[4][cN[4]].to_numpy()]

    # Define bin edges and number of bins
    bin_edges = np.linspace(-1, 2, 151)
    num_bins = len(bin_edges)


    # Initialize arrays to accumulate sums and counts
    bin_averages = np.zeros((3,num_bins-1))
    bin_rms = np.zeros((3,num_bins-1))
    bin_errors = np.zeros((3,num_bins-1))

    # Loop through data to accumulate sums and counts within each bin
    for k in range(3):
        bin_indices = np.digitize(x[k], bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 1) 
        num_bins = len(bin_edges) - 1
        bin_sums = np.bincount(bin_indices, weights=y[k], minlength=num_bins)
        bin_sums2 = np.bincount(bin_indices, weights=y[k]*y[k], minlength=num_bins)
        bin_counts = np.bincount(bin_indices, minlength=num_bins)

        for j in range(num_bins):
            if (bin_counts[j] <= 0):
                bin_counts[j] = 1
        
        updated_array = bin_sums[:-1]
        updated_array1 = bin_counts[:-1]
        updated_array2 = bin_sums2[:-1]

        averages = updated_array / updated_array1
        squareAverages = updated_array2 / updated_array1

        # Calculate averages and errors
        bin_averages[k] = (averages)
        bin_rms[k] = (np.sqrt(squareAverages - averages**2))
        bin_errors[k] = (np.sqrt(updated_array) / updated_array1)  # Assuming Poisson errors


    print (((bin_edges[1:] + bin_edges[:-1]) / 2).shape)
    print(bin_averages[0].shape)

    fig, ax = plt.subplots()

    #plt.errorbar((bin_edges[1:] + bin_edges[:-1]) / 2, bin_averages[1], yerr=bin_rms[1], fmt="o-", color='#228B22', label = '$K$')
    #plt.errorbar((bin_edges[1:] + bin_edges[:-1]) / 2, bin_averages[0], yerr=bin_rms[0], fmt="o-", color='#0504aa', label = '$\pi$')
    #plt.errorbar((bin_edges[1:] + bin_edges[:-1]) / 2, bin_averages[2], yerr=bin_rms[2], fmt="o-", color='#cf4c00', label = '$p$')
    ax.plot((bin_edges[1:] + bin_edges[:-1]) / 2, bin_averages[1], color='#228B22', label = '$K$')
    ax.fill_between((bin_edges[1:] + bin_edges[:-1]) / 2, bin_averages[1] - bin_rms[1], bin_averages[1] + bin_rms[1], alpha=0.3, color='#228B22', label='')
    ax.plot((bin_edges[1:] + bin_edges[:-1]) / 2, bin_averages[0], color='#0504aa', label = '$\pi$')
    ax.fill_between((bin_edges[1:] + bin_edges[:-1]) / 2, bin_averages[0] - bin_rms[0], bin_averages[0] + bin_rms[0], alpha=0.3, color='#0504aa', label='')
    ax.plot((bin_edges[1:] + bin_edges[:-1]) / 2, bin_averages[2], color='#cf4c00', label = '$p$')
    ax.fill_between((bin_edges[1:] + bin_edges[:-1]) / 2, bin_averages[2] - bin_rms[2], bin_averages[2] + bin_rms[2], alpha=0.3, color='#cf4c00', label='')



    ax.set_xlabel("$M^{2} [GeV^{2}]$")
    ax.xaxis.set_label_coords(0.9, -0.07)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("probability [%]")
    ax.yaxis.set_label_coords(-0.07, 0.9)
    ax.set_ylim(-0.05, 1.05)
    #ax.set_title("probabilities for true particles' types (sim)")

    legend = plt.legend(loc=[0.6,0.5])

    font = {'family': 'serif', 'size': 12, 'weight': 'normal'}
    for text in legend.get_texts():
        text.set_fontproperties(font)  # Set font properties for legend labels
    plt.show()



def draw_parameter_spread(tables, column):
    #class_hist0 = np.sqrt(np.absolute(tables[0][column].to_numpy()))
    #class_hist1 = np.sqrt(np.absolute(tables[2][column].to_numpy()))
    #class_hist2 = np.sqrt(np.absolute(tables[4][column].to_numpy()))
    
    class_hist0 = tables[0][column].to_numpy()
    class_hist1 = tables[2][column].to_numpy()
    class_hist2 = tables[4][column].to_numpy()


    class_histAll = np.concatenate((class_hist0,class_hist1),axis=0)

    bins = np.linspace(-5,5,5000)

    #plt.hist(class_histAll[:], bins, color='#660404',
    #                        alpha=0.7, rwidth=1.0, label = '$Sum$')
    plt.hist(class_hist0[:], bins, color='#0504aa',
                            alpha=0.7, rwidth=1.0, label = '$\pi$')
    plt.hist(class_hist1[:], bins, color='#228B22',
                            alpha=0.7, rwidth=1.0, label = '$K$')
    plt.hist(class_hist2[:], bins, color='#cf4c00',
                            alpha=0.7, rwidth=1.0, label = '$p$')
    plt.xlabel('mass2')
    plt.show()


def draw_confusion_matrix(maskPrediction,maskTarget):
    confusionMatrix = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            confusionMatrix[i][j] = np.around(100*np.count_nonzero(np.multiply(maskPrediction[i],maskTarget[j]))/np.count_nonzero(maskTarget[j]),decimals=3)
    print(confusionMatrix)
    #confusionMatrix = np.array(confusionMatrix)
    h = plt.matshow(confusionMatrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confusionMatrix.shape[0]):
        for j in range(confusionMatrix.shape[1]):
            plt.text(x=j, y=i,s=confusionMatrix[i, j], va='center', ha='center', size='large')
    plt.xlabel('Target', fontsize=12)
    plt.ylabel('Prediction', fontsize=12)
    #h = plt.hist2d(np.arange(5),np.arange(5), bins = 5, cmin=1, cmap=plt.cm.jet)
    #plt.colorbar(h[3])
    plt.show()


def draw_2d_param_spread(tables, column1, column2):
    class_hist0 = [tables[1][column1].to_numpy(),tables[1][column2].to_numpy()]
    class_hist1 = [tables[4][column1].to_numpy(),tables[4][column2].to_numpy()]
    class_hist2 = [tables[3][column1].to_numpy(),tables[3][column2].to_numpy()]

    x = np.linspace(0,5,100)
    y = 1./np.sqrt(1.+(0.195**2)/(x**2)) * (1.-0.02*np.exp(-((x-0.55)*(x-0.55))/2./(0.3*0.3)))
    plt.plot(x,y,'r')
    y1 = 1./np.sqrt(1.+(0.685**2)/(x**2)) * (1.-0.02*np.exp(-((x-0.95)*(x-0.95))/2./(0.3*0.3)))
    plt.plot(x,y1,'r')

    h0 = plt.hist2d(class_hist1[0],class_hist1[1], bins = 300, cmin=5, cmap=plt.cm.jet)
    h1 = plt.hist2d(class_hist0[0],class_hist0[1], bins = 300, cmin=5, cmap=plt.cm.jet)
    h2 = plt.hist2d(class_hist2[0],class_hist2[1], bins = 100, cmin=1, cmap=plt.cm.jet)
    plt.colorbar(h2[3])
    print(len(class_hist2[0]))
    print("ASAAAAAAAAAAAAAAAAAAAAA")
    print(class_hist2)
    #plt.hist2d(class_hist2[0],class_hist2[1], bins = 300, cmap = "RdYlBu_r", norm = colors.LogNorm())
    #plt.ylim([0,1.5])
    plt.show()


def draw_feature_distribution(tablesS, tablesE):
    class_histS = [tablesS[0][(list(tablesS[0].columns)[-10+i])].to_numpy() for i in range(9)]
    class_histE = [tablesE[0][(list(tablesE[0].columns)[-10+i])].to_numpy() for i in range(9)]

    #bins = np.linspace(-0.25,0.25,3000)
    bins = [np.arange(np.mean(class_histS[i]) - 2 * np.std(class_histS[i]), np.mean(class_histS[i]) + 2 * np.std(class_histS[i]) + 4*np.std(class_histS[i])/200,  4*np.std(class_histS[i])/200) for i in range(9)]

    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))

    # Loop through the subplots and plot histograms
    for i, ax in enumerate(axes.flatten()):
        ax.hist(class_histS[i], bins[i], edgecolor='#0504aa', linewidth=2, fill=False, histtype='step',
            alpha=0.7, rwidth=1.0, density=True, label = 'sim')
        ax.hist(class_histE[i], bins[i], edgecolor='#9e001a', linewidth=2, fill=False, histtype='step',
            alpha=0.7, rwidth=1.0, density=True, label = 'exp')
        ax.legend()
    
    fig.suptitle('"Feature" distributions, 9/128, both train AND validation dataset', fontsize=16)

    plt.show()

def draw_initial_distribution(tablesS, tablesE):
    class_histS = [tablesS[0][(list(tablesS[0].columns)[i])].to_numpy() for i in range(10)]
    class_histE = [tablesE[0][(list(tablesE[0].columns)[i])].to_numpy() for i in range(10)]

    #bins = np.linspace(-0.25,0.25,3000)
    bins = [np.arange(np.mean(class_histS[i]) - 4 * np.std(class_histS[i]), np.mean(class_histS[i]) + 4 * np.std(class_histS[i]) + 4*np.std(class_histS[i])/200,  4*np.std(class_histS[i])/200) for i in range(10)]

    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(4, 4, figsize=(10, 8))

    # Loop through the subplots and plot histograms
    for i, ax in enumerate(axes.flatten()):
        if (i >= 10):
            continue
        ax.hist(class_histS[i], bins[i], edgecolor='#0504aa', linewidth=2, fill=False, histtype='step',
            alpha=0.7, rwidth=1.0, density=True, label = 'sim')
        ax.hist(class_histE[i], bins[i], edgecolor='#9e001a', linewidth=2, fill=False, histtype='step',
            alpha=0.7, rwidth=1.0, density=True, label = 'exp')
        ax.legend()
    
    fig.suptitle('"Feature" distributions, 9/128, both train AND validation dataset', fontsize=16)

    plt.show()


def write_output(outputs, mod, enlist):
    goodTableList = []
    badEventTableList = []
    badChi2TableList = []
    i = 0
    for dft in outputs:
        # split output table in 3 - good, bad by nn and bad by input (no energy point in train or 100 chi2)
        badChi2Table = dft.loc[(dft['chi2']==100) | ((dft['0']==0) & (dft['1']==0))].copy()
        possGTable = dft.drop(badChi2Table.index)
        goodTable0 = possGTable.loc[(possGTable['0']-possGTable['1']>0.2) & (possGTable['0']-possGTable['2']>0.4) & (possGTable['0']-possGTable['3']>0.2) & (possGTable['0']>0.2)].copy()
        goodTable1 = possGTable.loc[(possGTable['1']-possGTable['0']>0) & (possGTable['1']-possGTable['2']>0) & (possGTable['1']-possGTable['3']>0) & (possGTable['1']>0.2)].copy()
        goodTable = pandas.concat([goodTable0,goodTable1])
        badEventTable = possGTable.drop(goodTable0.index).copy()
        efficiency = 0
        if (possGTable.shape[0] > 0):
            efficiency = float(goodTable.shape[0])/possGTable.shape[0]
        print(str(enlist[i]) + " efficiency equals to " + str(efficiency*100))
        goodTableList.append(goodTable0)
        badEventTableList.append(badEventTable)
        badChi2TableList.append(badChi2Table)
        i=i+1

    goodTableFull = pandas.concat(list(goodTableList)).sort_index()
    badEventTableFull = pandas.concat(list(badEventTableList)).sort_index()
    badChi2TableFull = pandas.concat(list(badChi2TableList)).sort_index()

    draw_probabilities_spread(pandas.concat(outputs).sort_index(),goodTableFull)

    # assing final marks to each input row
    goodTableFull['0'] = 1
    badEventTableFull['0'] = 0
    badChi2TableFull['0'] = 3   
    
    resultingMarks = pandas.concat([goodTableFull,badEventTableFull,badChi2TableFull]).sort_index().drop(['1','2','3'], axis=1)
    resultingMarks.to_csv("testMarks" + mod + ".csv",sep='	',header=False,index=False)


def analyseOutput(predFileName, experiment_path,mod):
    pT = pandas.read_parquet(os.path.join("nndata",predFileName))
    dftCorrExp = pandas.read_parquet(os.path.join("nndata",experiment_path))
    print(pT)
    print(dftCorrExp)
    plt.show()
    tablesPClasses = []
    tablesClasses = []
    tablesClasses2 = []
    tablesClasses3 = []
    lrange = 5
    #(dftCorrExp['pid']==3) 
    mask = []
    mask2 = []
    for i in range(lrange):
        cN = list(pT.columns)
        #mask.append((dftCorrExp['beta']<1.2) & (dftCorrExp['beta']>0.25))
        #mask2.append((dftCorrExp['beta']<1.2) & (dftCorrExp['beta']>0.25))
        mask.append((dftCorrExp['mass2']<111.2))
        mask2.append((dftCorrExp['mass2']<111.2))
        for j in range(lrange-1):
            if (mod == "sim"):
                mask2[i] = mask2[i] & (dftCorrExp['pid'] == i)
            mask[i] = mask[i] & (pT[cN[i]]-pT[cN[(i+j+1)%lrange]]>0.4)
        tablesPClasses.append(pT.loc[mask2[i]].copy())
        tablesClasses.append(dftCorrExp.loc[mask[i]].copy())
        tablesClasses2.append(dftCorrExp.loc[mask2[i]].copy())
        #tablesClasses3.append(dftCorrExp)
    #print(tablesClasses[0])
    #print(str(tablesClasses[0].shape[0]) + "," + str(tablesClasses[1].shape[0]) + "," +str(tablesClasses[2].shape[0]) + "," +str(tablesClasses[3].shape[0]) + "," +str(tablesClasses[4].shape[0]))
    #sumEvs = (tablesClasses[0].shape[0] + tablesClasses[1].shape[0] + tablesClasses[2].shape[0] + tablesClasses[3].shape[0] + tablesClasses[4].shape[0])/5
    #print(str(sumEvs / tablesClasses[0].shape[0]) + "," + str(sumEvs /tablesClasses[1].shape[0]) + "," +str(sumEvs /tablesClasses[2].shape[0]) + "," +str(sumEvs /tablesClasses[3].shape[0]) + "," +str(sumEvs /tablesClasses[4].shape[0]))

    print(tablesClasses2)
    print(tablesPClasses)
    
    #draw_probabilities_spread(tablesPClasses[1],tablesPClasses[1])
    if (mod == "sim"):
        draw_probabilities_vs_parameter(tablesPClasses,tablesClasses2, 'mass2')
        draw_confusion_matrix(np.array(mask),np.array(mask2))
        draw_2d_param_spread(tablesClasses2,'momentum','mdcdedx')
        #draw_2d_param_spread(tablesClasses2,'momentum','beta')
    elif (mod == "exp"):
        draw_2d_param_spread(tablesClasses,'momentum','mdcdedx')
        #draw_2d_param_spread(tablesClasses,'momentum','beta')
        draw_parameter_spread(tablesClasses,'mass2')
    #draw_parameter_spread(tablesClasses,'tof')
    #draw_parameter_spread(tablesClasses2,'mass2')
    #plt.show()

    #0.7380741598515985,0.924485731694385,2.541151390667785,4.290668468491014,0.5163012252092372

def analyseExpAndSim(predFileNameSim, experiment_pathSim, predFileNameExp, experiment_pathExp):
    pTS = pandas.read_parquet(os.path.join("nndata",predFileNameSim))
    dftS = pandas.read_parquet(os.path.join("nndata",experiment_pathSim))
    pTE = pandas.read_parquet(os.path.join("nndata",predFileNameExp))
    dftE = pandas.read_parquet(os.path.join("nndata",experiment_pathExp))
    print(pTS)

    tablesPClassesS = []
    tablesPClassesE = []
    tablesClassesS = []
    tablesClassesE = []
    lrange = 5
    mask = []
    mask2 = []
    #for i in range(lrange):
    #mask.append((dftS['beta']<1.2) & (dftS['beta']>0.55))
    #mask2.append((dftE['beta']<1.2) & (dftE['beta']>0.55))
    mask.append((dftS['mass2']<111.2))
    mask2.append((dftE['mass2']<111.2))
    print(mask[0].shape[0])
    print(pTS.shape[0])
    tablesPClassesS.append(pTS.copy())
    tablesPClassesE.append(pTE.copy())

    tablesClassesS.append(dftS.copy())
    tablesClassesE.append(dftE.copy())
    
    #draw_feature_distribution(tablesPClassesS,tablesPClassesE)
    draw_initial_distribution(tablesClassesS,tablesClassesE)


def predict_nn(fName, oName):

    predictionList = makePredicionList(fName, oName)
    print(predictionList)

    #write_output(predictionList,mod,enlist)


def predict(fName, oName):
    predict_nn(fName, oName)

#dataManager = DataManager()
#dataManager.manageDataset("test")

#dataSetType = 'NewKIsUsed'

#print("start python predict")
#predict('expu' + dataSetType + '.parquet','predictedExp' + dataSetType + '.parquet')
#predict('simu' + dataSetType + '.parquet','predictedSim' + dataSetType + '.parquet')
analyseOutput('predictedExp' + dataSetType + '.parquet','expu' + dataSetType + '.parquet',"exp")
analyseOutput('predictedSim' + dataSetType + '.parquet','simu' + dataSetType + '.parquet',"sim")

#analyseExpAndSim('predictedSim' + dataSetType + '.parquet','simu' + dataSetType + '.parquet', 'predictedExp' + dataSetType + '.parquet','expu' + dataSetType + '.parquet')

#plt.show()
