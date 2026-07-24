
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import pandas
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from models.model import DANN, Encoder, Classifier, Discriminator
from models.model import Model
from dataHandling import My_dataset, DataManager, load_dataset
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from tqdm.auto import tqdm
from tqdm import trange
import math
import os

import ROOT


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

#dataSetType = 'NewKIsUsed'
dataSetType = 'gen4'
dataManager = DataManager(dataSetType)


def loadModel(input_dim, nClasses):
    nn_model = DANN(input_dim=input_dim, output_dim=nClasses).type(torch.FloatTensor)
    nn_model.type(torch.FloatTensor)
    nn_model.load_state_dict(torch.load(os.path.join('nndata','tempModel' + dataSetType + '.pt')))
    #nn_model.eval()
    return nn_model.to(device)

def loadModelSeparated(input_dim, nClasses):
    latentDim = 64
    encoder = Encoder(input_dim=input_dim, output_dim=latentDim).to(device).float()
    enc_sd = torch.load(os.path.join('nndata', 'encoder' + dataSetType + '.pt'),
                        map_location=device, weights_only=True)
    encoder.load_state_dict(enc_sd)

    classifier = Classifier(input_dim=latentDim, output_dim=nClasses).to(device).float()
    cls_sd = torch.load(os.path.join('nndata', 'classifier' + dataSetType + '.pt'),
                        map_location=device, weights_only=True)
    classifier.load_state_dict(cls_sd)

    discriminator = Discriminator(input_dim=latentDim, output_dim=2).to(device).float()
    disc_sd = torch.load(os.path.join('nndata', 'discriminator' + dataSetType + '.pt'),
                         map_location=device, weights_only=True)
    discriminator.load_state_dict(disc_sd)

    return encoder, classifier, discriminator

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

    #dftCorrExp['charge'] = dftCorrExp['charge'].replace(-1,1)

    #print(dftCorrExp.iloc[0])
    
    dftCorrExp = dataManager.normalizeDataset(dftCorrExp)

    print(dftCorrExp.iloc[0])

    #class1p = dftCorrExp.loc[(dftCorrExp['pid']==1)].copy()
    #restt = dftCorrExp.drop(class1p.index)
    #class1p = class1p.sample(frac=0.05).copy()
    #print(class1p)
    #dftCorrExp = pandas.concat([class1p,restt]).sort_index()
    #print(dftCorrExp)

    exp_dataset = My_dataset(load_dataset(dftCorrExp))
    exp_dataLoader = DataLoader(exp_dataset, batch_size=1024*32, drop_last=False)

    #nClasses = dftCorrExp[list(dftCorrExp.columns)[-1]].nunique()
    nClasses = 5
    input_dim = exp_dataset[0][0].shape[0]

    #load nn and predict
    #nn_model = loadModel(input_dim, nClasses)
    #nn_model.eval()
    encoder,classifier,discriminator = loadModelSeparated(input_dim, nClasses)
    encoder.eval()
    classifier.eval()
    discriminator.eval()
    
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
        #e_class, e_domain = nn_model(ve_x)
        e_feature = encoder(ve_x)
        e_class = classifier(e_feature)
        e_class = e_class.softmax(dim=1).detach().cpu().numpy()
        #e_feature = e_feature[:,0:-1].detach().cpu().numpy()
        e_class = np.concatenate((e_class,e_feature.detach().cpu().numpy()),axis=1)
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

    bins = np.linspace(-1,1,500)

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

    canvas = ROOT.TCanvas("canvas", "Canvas Title", 800, 600)
    hist0 = ROOT.TH1F("hist0", "p;#beta_{measured}-#beta{expected}", 300, -2, 2)
    hist1 = ROOT.TH1F("hist1", "K^{+};#beta_{measured}-#beta{expected}", 100, -2, 2)
    hist2 = ROOT.TH1F("hist2", "#pi^{+};#beta_{measured}-#beta{expected}", 300, -2, 2)
    for i in range(len(class_hist0)):
        hist0.Fill(class_hist0[i])
    for i in range(len(class_hist1)):
        hist1.Fill(class_hist1[i])
    for i in range(len(class_hist2)):
        hist2.Fill(class_hist2[i])
    hist0.Draw("colz")
    hist2.Draw("colzsame")
    hist1.Draw("colzsame")
    canvas.Update()
    input()


def draw_parameters_spread(tables, column,column1,column2):
    #class_hist0 = np.sqrt(np.absolute(tables[0][column].to_numpy()))
    #class_hist1 = np.sqrt(np.absolute(tables[2][column].to_numpy()))
    #class_hist2 = np.sqrt(np.absolute(tables[4][column].to_numpy()))
    
    class_hist0 = tables[0][column].to_numpy()
    class_hist1 = tables[2][column1].to_numpy()
    class_hist2 = tables[4][column2].to_numpy()


    class_histAll = np.concatenate((class_hist0,class_hist1),axis=0)

    bins = np.linspace(-1,1,500)

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

    canvas = ROOT.TCanvas("canvas", "Canvas Title", 800, 600)
    hist0 = ROOT.TH1F("hist0", "p;#beta_{measured}-#beta{expected}", 300, -2, 2)
    hist1 = ROOT.TH1F("hist1", "K^{+};#beta_{measured}-#beta{expected}", 100, -2, 2)
    hist2 = ROOT.TH1F("hist2", "#pi^{+};#beta_{measured}-#beta{expected}", 300, -2, 2)
    for i in range(len(class_hist0)):
        hist0.Fill(class_hist0[i])
    for i in range(len(class_hist1)):
        hist1.Fill(class_hist1[i])
    for i in range(len(class_hist2)):
        hist2.Fill(class_hist2[i])
    hist0.Draw("colz")
    hist2.Draw("colzsame")
    hist1.Draw("colzsame")
    canvas.Update()
    input()



def draw_confusion_matrix(maskPrediction,maskTarget):
    nClasses = 5
    confusionMatrix = np.zeros((nClasses,nClasses))
    for i in range(nClasses):
        for j in range(nClasses):
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
    class_hist0 = [tables[0][column1].to_numpy(),tables[0][column2].to_numpy()]
    class_hist1 = [tables[4][column1].to_numpy(),tables[4][column2].to_numpy()]
    class_hist2 = [tables[2][column1].to_numpy(),tables[2][column2].to_numpy()]

    #print(class_hist0[1])
    #print(class_hist2[1])
    print(len(class_hist2[0]))
    print(class_hist2)

    # Print class shares vs binned column1 (momentum)
    all_col1 = np.concatenate([t[column1].to_numpy() for t in tables if column1 in t.columns])
    if all_col1.size > 0:
        bin_edges = np.linspace(0, 4, 41)
        total_counts, _ = np.histogram(all_col1, bins=bin_edges)
        class0_counts, _ = np.histogram(class_hist0[0], bins=bin_edges)
        class2_counts, _ = np.histogram(class_hist2[0], bins=bin_edges)
        class4_counts, _ = np.histogram(class_hist1[0], bins=bin_edges)

        print(f"Shares by {column1} bins (class 0 and 2):")
        print("bin_low\tbin_high\tshare_class0\tshare_class2\tshare_class4\tcount_total")
        for i in range(len(bin_edges) - 1):
            total = total_counts[i]
            if total == 0:
                share0 = 0.0
                share2 = 0.0
                share4 = 0.0
            else:
                share0 = class0_counts[i] / total
                share2 = class2_counts[i] / total
                share4 = class4_counts[i] / total
            print(f"{bin_edges[i]:.6g}\t{bin_edges[i+1]:.6g}\t{share0:.6f}\t{share2:.6f}\t{share4:.6f}\t{total}")

    if (column2 == "beta"):
        x = np.linspace(0,1,100)
        y = 1./np.sqrt(1.+(0.13957**2)/(x**2))
        plt.xlabel("P [GeV/c]")
        plt.ylabel(r"$\beta$")
        plt.plot(x,y,'black')

        x = np.linspace(0,1,100)
        y = 1./np.sqrt(1.+(0.195**2)/(x**2)) * (1.-0.02*np.exp(-((x-0.55)*(x-0.55))/2./(0.3*0.3)))
        plt.plot(x,y,'r')

        xk = np.linspace(0,1,100)
        yk = 1./np.sqrt(1.+(0.493**2)/(xk**2))
        plt.plot(xk,yk,'black')

        y1 = 1./np.sqrt(1.+(0.685**2)/(x**2)) * (1.-0.02*np.exp(-((x-0.95)*(x-0.95))/2./(0.3*0.3)))
        plt.plot(x,y1,'r')

        xp = np.linspace(0,1,100)
        yp = 1./np.sqrt(1.+(0.938272**2)/(xp**2))
        plt.plot(xp,yp,'black')

    h1 = plt.hist2d(class_hist0[0],class_hist0[1], bins = 300, cmin=5, cmap=plt.cm.jet)
    h2 = plt.hist2d(class_hist2[0],class_hist2[1], bins = 100, cmin=5, cmap=plt.cm.jet)
    h0 = plt.hist2d(class_hist1[0],class_hist1[1], bins = 300, cmin=5, cmap=plt.cm.jet)
    plt.colorbar(h2[3])
    #plt.hist2d(class_hist2[0],class_hist2[1], bins = 300, cmap = "RdYlBu_r", norm = colors.LogNorm())
    if (column2 == "beta"):
        plt.ylim([0,1.5])
    plt.show() 

    canvas = ROOT.TCanvas("canvas", "Canvas Title", 1400, 1000)
    if (column2 == "beta"):
        hist0 = ROOT.TH2F("hist0", "p;P [GeV/c];#beta", 300, 0, 5, 300, 0.2, 1.2)
        hist1 = ROOT.TH2F("hist1", "K^{+};P [GeV/c];#beta", 100, 0, 5, 100, 0.2, 1.2)
        hist2 = ROOT.TH2F("hist2", "#pi^{+};P [GeV/c];#beta", 300, 0, 5, 300, 0.2, 1.2)
        for i in range(len(class_hist1[0])):
            hist0.Fill(class_hist1[0][i],class_hist1[1][i])
        for i in range(len(class_hist2[0])):
            hist1.Fill(class_hist2[0][i],class_hist2[1][i])
        for i in range(len(class_hist0[0])):
            hist2.Fill(class_hist0[0][i],class_hist0[1][i])
        hist0.Draw("colz")
        hist2.Draw("colzsame")
        hist1.Draw("colzsame")
        canvas.Update()
    elif ("newCol" in column2):
        hist0 = ROOT.TH2F("hist0", "p;P [GeV/c];#beta_{expected}-#beta_{measured}", 300, 0, 5, 300, -0.5, 0.5)
        hist1 = ROOT.TH2F("hist1", "K^{+};P [GeV/c];#beta_{expected}-#beta_{measured}", 100, 0, 5, 100, -0.5, 0.5)
        hist2 = ROOT.TH2F("hist2", "#pi^{+};P [GeV/c];#beta_{expected}-#beta_{measured}", 300, 0, 5, 300, -0.5, 0.5)
        for i in range(len(class_hist1[0])):
            hist0.Fill(class_hist1[0][i],class_hist1[1][i])
        for i in range(len(class_hist2[0])):
            hist1.Fill(class_hist2[0][i],class_hist2[1][i])
        for i in range(len(class_hist0[0])):
            hist2.Fill(class_hist0[0][i],class_hist0[1][i])
        hist0.Draw("colz")
        hist2.Draw("colzsame")
        hist1.Draw("colzsame")
        canvas.Update()

    input()


def plot_roc_curves(y_true, y_score, class_names=None, average=False, groups=None):
    n_classes = y_score.shape[1]

    # Binary indicators for each class
    y_true_bin = np.eye(n_classes)[y_true]

    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]

    plt.figure(figsize=(10, 8))

    if average:
        # Compute micro-average ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2,
                 label=f'micro-average ROC curve (area = {roc_auc:.3f})')

    if groups:
        for idx, group in enumerate(groups):
            for i in group:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2,
                         label=f'ROC curve of {class_names[i]} (group {idx+1}) (area = {roc_auc:.3f})')
    else:
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2,
                     label=f'ROC curve of {class_names[i]} (area = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
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
    dftCorrExp = pandas.read_parquet(os.path.join("nndata",experiment_path)).reset_index()
    #dftCorrExp['charge'] = dftCorrExp['charge'].replace(-1,1)
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
    for i in range(lrange):  #looping through particle classes
        cN = list(pT.columns)
        #mask.append((dftCorrExp['beta']<1.5) & (dftCorrExp['beta']>0.1))
        #mask2.append((dftCorrExp['beta']<1.5) & (dftCorrExp['beta']>0.1))
        mask.append((dftCorrExp['charge']<1.5) & (dftCorrExp['charge']>-1.5))
        mask2.append((dftCorrExp['charge']<1.5) & (dftCorrExp['charge']>-1.5))
        #mask.append((dftCorrExp['mass2']<111.2))
        #mask2.append((dftCorrExp['mass2']<111.2))
        if (mod == "sim"):
            mask2[i] = mask2[i] & (dftCorrExp['pid'] == i)

        for j in range(lrange-1):     #looping through other classes to check probability differences
            #mask[i] = mask[i] & (pT[cN[i]]-pT[cN[(i+j+1)%lrange]]>0.01 & pT[cN[i]]-pT[cN[(i+j+1)%lrange]]<0.7)
            mask[i] = mask[i] & (pT[cN[i]]-pT[cN[(i+j+1)%lrange]]>0.1)
        #pT.index = pandas.Int64Index(pT.index)
        print (pT.index)
        print (mask2[i].index)
        tablesPClasses.append(pT.loc[mask2[i]].copy())
        tablesClasses.append(dftCorrExp.loc[mask[i]].copy())
        tablesClasses2.append(dftCorrExp.loc[mask2[i]].copy())
        #tablesClasses3.append(dftCorrExp)
    #print(tablesClasses[0])
    print(str(tablesClasses[0].shape[0]) + "," + str(tablesClasses[1].shape[0]) + "," +str(tablesClasses[2].shape[0]) + "," +str(tablesClasses[3].shape[0]) + "," +str(tablesClasses[4].shape[0]))
    #sumEvs = (tablesClasses[0].shape[0] + tablesClasses[1].shape[0] + tablesClasses[2].shape[0] + tablesClasses[3].shape[0] + tablesClasses[4].shape[0])/5
    #print(str(sumEvs / tablesClasses[0].shape[0]) + "," + str(sumEvs /tablesClasses[1].shape[0]) + "," +str(sumEvs /tablesClasses[2].shape[0]) + "," +str(sumEvs /tablesClasses[3].shape[0]) + "," +str(sumEvs /tablesClasses[4].shape[0]))

    #print(tablesClasses2)
    #print(tablesPClasses)
    
    #draw_probabilities_spread(tablesPClasses[1],tablesPClasses[1])
    if (mod == "sim"):
        #draw_probabilities_vs_parameter(tablesPClasses,tablesClasses2, 'mass2')
        #    draw_confusion_matrix(np.array(mask),np.array(mask2))
        plot_roc_curves(dftCorrExp['pid'].to_numpy(),pT.to_numpy()[:,:5], class_names=['$\pi^{+}$','$\pi^{-}$','$K^{+}$','$K^{-}$','p'])    #for each true class from sim -> how they are identified.
        #draw_2d_param_spread(tablesClasses,'momentum','mdcdedx')
        #    draw_2d_param_spread(tablesClasses,'momentum','beta')
        #draw_2d_param_spread(tablesClasses,'momentum','newColK')
        #draw_parameter_spread(tablesClasses,'mass2')
        print("sim")
    elif (mod == "exp"):
        draw_2d_param_spread(tablesClasses,'momentum','mdcdedx')
        draw_2d_param_spread(tablesClasses,'momentum','beta')
        #draw_2d_param_spread(tablesClasses,'momentum','newColK')
        #draw_parameter_spread(tablesClasses,'mass2')
        #draw_parameter_spread(tablesClasses,'newColK')
        #draw_parameters_spread(tablesClasses,'newColPi1','newColK','newColp')
        print("exp")
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
    
    draw_feature_distribution(tablesPClassesS,tablesPClassesE)
    #draw_initial_distribution(tablesClassesS,tablesClassesE)


def predict_nn(fName, oName):

    predictionList = makePredicionList(fName, oName)
    print(predictionList)

    #write_output(predictionList,mod,enlist)




def plotSimpleComp():
    # Bin centers
    bin_centers = np.arange(0.05, 2.25, 0.1)

    # nn exp
    nn_exp = [
        0.000062, 0.001347, 0.058443, 0.246736, 0.369547, 0.431809,
        0.478176, 0.506656, 0.534461, 0.562519, 0.591011, 0.617339,
        0.645514, 0.673183, 0.697785, 0.724006, 0.757243, 0.780135,
        0.808918, 0.836201, 0.858410, 0.885601
    ]

    # nn sim
    nn_sim = [
        0.000000, 0.004278, 0.022886, 0.059118, 0.095918, 0.136483,
        0.201851, 0.291627, 0.399146, 0.506946, 0.606102, 0.689188,
        0.758846, 0.811606, 0.853431, 0.886965, 0.909095, 0.928865,
        0.942953, 0.955549, 0.961963, 0.971527
    ]

    # dann exp
    dann_exp = [
        0.000000, 0.001323, 0.055613, 0.244267, 0.367263, 0.429713,
        0.476758, 0.505761, 0.533645, 0.561214, 0.589737, 0.616373,
        0.644372, 0.671775, 0.696466, 0.722917, 0.755538, 0.779182,
        0.810122, 0.837669, 0.858944, 0.882542
    ]

    # dann sim
    dann_sim = [
        0.000000, 0.004264, 0.022870, 0.059105, 0.095906, 0.136459,
        0.201842, 0.291598, 0.399180, 0.506984, 0.606006, 0.689314,
        0.758719, 0.811171, 0.852527, 0.885640, 0.907682, 0.927348,
        0.942380, 0.955740, 0.961682, 0.970215
    ]

    plt.figure()
    plt.plot(bin_centers, nn_exp, label="nn exp")
    plt.plot(bin_centers, nn_sim, label="nn sim")
    plt.plot(bin_centers, dann_exp, label="dann exp")
    plt.plot(bin_centers, dann_sim, label="dann sim")

    plt.xlabel("bin center")
    plt.ylabel("share_class2")
    plt.legend()
    plt.show()




def predict(fName, oName):
    predict_nn(fName, oName)

#dataManager = DataManager()
#dataManager.manageDataset("test")

#dataSetType = 'NewKIsUsed'

#print("start python predict")
if os.environ.get("DANN_SWEEP") == "1":
    sim_file = 'predictedSim' + dataSetType + '.parquet'
    predict('simu' + dataSetType + '.parquet', sim_file)
    scores = pandas.read_parquet(os.path.join('nndata', sim_file)).to_numpy()[:, :5]
    labels = pandas.read_parquet(os.path.join('nndata', 'simuTest' + dataSetType + '.parquet'))['pid'].to_numpy()
    auc_k_plus = auc(*roc_curve(labels == 2, scores[:, 2])[:2])
    auc_k_minus = auc(*roc_curve(labels == 3, scores[:, 3])[:2])
    print(f"SWEEP_RESULT auc_k_plus={auc_k_plus:.9f} auc_k_minus={auc_k_minus:.9f} product={auc_k_plus * auc_k_minus:.9f}")
    sys.exit(0)

predict('expu' + dataSetType + '.parquet','predictedExp' + dataSetType + '.parquet')
predict('simu' + dataSetType + '.parquet','predictedSim' + dataSetType + '.parquet')
#analyseOutput('predictedExp' + dataSetType + '.parquet','expuTest' + dataSetType + '.parquet',"exp")
analyseOutput('predictedSim' + dataSetType + '.parquet','simuTest' + dataSetType + '.parquet',"sim")

#analyseExpAndSim('predictedSim' + dataSetType + '.parquet','simu' + dataSetType + '.parquet', 'predictedExp' + dataSetType + '.parquet','expu' + dataSetType + '.parquet')

#plt.show()


#plotSimpleComp()
