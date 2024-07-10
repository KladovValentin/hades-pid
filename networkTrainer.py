
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import uproot
import pandas
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm.auto import tqdm
from tqdm import trange
from pympler import asizeof
from models.model import DANN, Encoder, Classifier, Discriminator
from models.model import Model
from dataHandling import My_dataset, DataManager, load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(torch.__version__)  # Check PyTorch version
print(torch.cuda.get_device_name(0))
print(f"Using {device} device")
if device == "cuda:0":
    torch.cuda.empty_cache()
    cuda = torch.device('cuda:0')
#print(torch.cuda.memory_allocated(0))
#print(torch.cuda.memory_reserved(0))

dataSetType = 'NewKIsUsed'
dataManager = DataManager(dataSetType)


def train_DN_model(model, train_loader, loss, optimizer, num_epochs, valid_loader, scheduler=None):
    print("start model nn train")
    loss_history = []
    train_history = []
    validLoss_history = []

    for epoch in range(num_epochs):
        model.train()

        loss_train = 0
        accuracy_train = 0
        isteps = 0
        tepoch = tqdm(train_loader,unit=" batch")
        for i_step, (x, y) in enumerate(tepoch):
            x.to(device)
            y.to(device)
            tepoch.set_description(f"Epoch {epoch}")

            prediction = model(x)
            running_loss = loss(prediction, y)
            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()

            indices = torch.max(prediction, 1)[1]   
            running_acc = torch.sum(indices==y)/y.shape[0]
            if i_step > len(train_loader)*3./4.:
                accuracy_train += running_acc
                loss_train += float(running_loss)
                isteps += 1

            loss_history.append(float(running_loss))

            tepoch.set_postfix(loss=float(running_loss), accuracy=float(running_acc)*100)
            del indices, prediction, x, y, running_acc, running_loss

        accuracy_train = accuracy_train/isteps
        loss_train = loss_train/isteps

        #<<<< Validation >>>>#
        model.eval()
        loss_valid = 0
        accuracy_valid = 0
        validLosses = []
        validAccuracies = []
        with torch.no_grad():
            for v_step, (x, y) in enumerate(valid_loader):
                x.to(device)
                y.to(device)
                prediction = model(x)
                validLosses.append(float(loss(prediction, y)))
                indices = torch.max(prediction, 1)[1]
                validAccuracies.append(float(torch.sum(indices==y))/ y.shape[0])

            loss_valid = np.mean(np.array(validLosses))
            accuracy_valid = np.mean(np.array(validAccuracies))
        model.train() 


        if scheduler is not None:
            #scheduler.step(ave_valid_loss)
            scheduler.step()


        #<<<< Printing and drawing >>>>#
        #loss_history.append(loss_train)
        train_history.append(accuracy_train)
        validLoss_history.append(float(loss_valid))
        ep = np.arange(1,(epoch+1)*(i_step+1)+1,1)
        lv = np.array(validLoss_history)
        lt = np.array(loss_history)
        plt.clf()
        plt.plot(ep,lt,"blue",label="train")
        #plt.plot(ep,lv,"orange",label="validation")
        plt.legend(loc=[0.5,0.6])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if ((epoch+1)%1 == 0):
            plt.show()

        print("Average loss: %f, valid loss: %f, Train accuracy: %f, V acc: %f, epoch: %f" % (loss_train, loss_valid, accuracy_train*100, accuracy_valid*100, epoch+1))
    
    return 1


def train_DANN_model(model, sim_loader, exp_loader, val_exp_loader, val_sim_loader, lossClass, lossDomain, optimizer, num_epochs, scheduler=None):
    loss_history = []
    train_history = []
    validLoss_history = []
    validAccu_history = []
    loss_valid = 0
    accuracy_valid = 0
    validLosses = []
    validAccuracies = []

    len_dataloader = min(len(sim_loader), len(exp_loader))
    len_dataloader1 = min(len(val_sim_loader), len(val_exp_loader))
    for epoch in range(num_epochs):
        sim_iter = iter(sim_loader)
        exp_iter = iter(exp_loader)
        perc10p=0
        tepoch = tqdm(range(len_dataloader), total=len_dataloader)
        for i_step in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            (s_x, s_y) = next(sim_iter)
            (e_x, _)   = next(exp_iter)
            s_x = s_x.to(device)
            s_y = s_y.long().flatten().to(device)
            e_x = e_x.to(device)

            #model.zero_grad()
            batch_size = len(s_y)

            domain_label = torch.zeros(len(s_y)).long().to(device)
            s_class, s_domain = model(s_x)
            s_class = s_class.to(device)
            s_domain = s_domain.to(device)

            batch_size = len(e_x)

            domain_labele = torch.ones(len(e_x)).long().to(device)
            _, e_domain = model(e_x)
            e_domain = e_domain.to(device)

            running_loss_class = lossClass(s_class, s_y)
            #print(s_domain.shape)
            #print(domain_label.shape)
            running_loss_domain = lossDomain(s_domain, domain_label) + lossDomain(e_domain, domain_labele)
            running_loss = 0.1*running_loss_domain + running_loss_class

            optimizer.zero_grad()
            #if (i_step//2 == 0):
            #    running_loss_class.backward()
            #else:
            #    running_loss_domain.backward()
            running_loss.backward()
            optimizer.step()

            e_domain_copy = e_domain.clone().detach()
            s_domain_copy = s_domain.clone().detach()
            s_class_copy = s_class.clone().detach()
            
            #print(indicesD.cpu(), domain_labele.cpu())

            e_domain_copy1 = e_domain.clone().detach()
            if not torch.equal(e_domain_copy1, e_domain_copy):
                print("prediction changed somehow even without grads")
            
            #print(s_domain)
            indicesD = torch.argmax(s_domain_copy, 1)
            running_accD = torch.sum(indicesD == 0)/(len(s_x))

            indicesD1 = torch.argmax(e_domain_copy, 1)
            running_accD1 = torch.sum(indicesD1 == 1)/(len(e_x))

            #print(indicesD.cpu(), domain_labele.cpu())

            indices = torch.max(s_class, 1)[1]
            running_acc = torch.sum(indices==s_y)/len(s_y)

            #print(e_domain.cpu())

            train_history.append(1-running_acc)
            loss_history.append(float(running_loss))

            tepoch.set_postfix(loss=float(running_loss), acc=float(running_acc)*100, accD=float(running_accD)*100, accD1=float(running_accD1)*100)

            #perc10 = (((len(loss_history)+1)*1)//len_dataloader)
            #if ( perc10>perc10p):
                #perc10p = perc10
                # validation step
                

        validFeaturesSim = []
        validFeaturesExp = []
        with torch.no_grad():
            model.eval()
            val_sim_iter = iter(val_sim_loader)
            val_exp_iter = iter(val_exp_loader)
            for j_step in range(len_dataloader1):
                            
                (vs_x, vs_y) = next(val_sim_iter)
                (ve_x, _)   = next(val_exp_iter)
                vs_x = vs_x.to(device)
                vs_y = vs_y.long().to(device)
                ve_x = ve_x.to(device)

                vs_y = vs_y.long().flatten()
                batch_size = len(vs_y)

                vdomain_label = torch.zeros(len(vs_y)).long().to(device)
                vs_class, vs_domain = model(vs_x)
                vs_class = vs_class.to(device)
                vs_domain = vs_domain.to(device)
                #print(s_class)
                vs_loss = lossClass(vs_class, vs_y) + 0.1*lossDomain(vs_domain, vdomain_label)

                batch_size = len(ve_x)

                vdomain_label = torch.ones(len(ve_x)).long().to(device)
                _, ve_domain = model(ve_x)
                ve_domain = ve_domain.to(device)
                ve_loss = 0.1*lossDomain(ve_domain, vdomain_label)

                vrunning_loss = vs_loss + ve_loss

                vindices = torch.max(vs_class, 1)[1]
                vrunning_acc = torch.sum(vindices==vs_y)/vs_y.shape[0]

                validLosses.append(float(vrunning_loss))
                validAccuracies.append(vrunning_acc.cpu())

                #validFeaturesSim.append(vs_feature[:,10].detach().cpu().numpy())
                #validFeaturesExp.append(ve_feature[:,10].detach().cpu().numpy())


            #bins = np.linspace(-0.1,0.1,2000)
            #validFeaturesSim = np.concatenate(validFeaturesSim)
            #validFeaturesExp = np.concatenate(validFeaturesExp)
            #plt.hist(validFeaturesSim, bins, color='#0504aa',
            #        alpha=0.7, rwidth=1.0, density=True, label = 'sim')
            #plt.hist(validFeaturesExp, bins, color='#9e001a',
            #        alpha=0.7, rwidth=1.0, density=True, label = 'exp')
            #plt.show()

            loss_valid = np.mean(np.array(validLosses))
            accuracy_valid = np.mean(np.array(validAccuracies))
            validAccu_history.append(accuracy_valid)
            validLoss_history.append(loss_valid)


            print("Valid loss: %f, V acc: %f, epoch: %f" % (loss_valid, accuracy_valid*100, epoch+1))

            #model.eval()
            model.cpu()
            torch.save(model.state_dict(), os.path.join('nndata','tempModel' + dataSetType + '.pt'))
            model.to(device)

            model.train()

        if scheduler is not None:
            scheduler.step()

    # drawing step
    ep = np.arange(1,len(loss_history)+1,1)
    lt = np.array(loss_history)
    at = np.array(train_history)
    vx = np.arange(len(loss_history)/len(validLoss_history),len(loss_history)+1,len(loss_history)/len(validLoss_history))
    vyl= np.array(validLoss_history)
    vya= np.array(validAccu_history)
    plt.clf()
    #plt.plot(ep,at,"orange",label="1-acc")
    plt.plot(ep,lt,"blue",label="lossTrain")
    plt.plot(vx, vyl,"red",label="lossValid")
    plt.legend(loc=[0.5,0.6])
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.show()
    
    model.eval()

    return 1


def train_Proper_DANN_model(encoder, classifier, discriminator, sim_loader, exp_loader, val_exp_loader, val_sim_loader, lossClass, lossDomain, optimizer, num_epochs, scheduler):
    loss_history = []
    train_history = []
    validLoss_history = []
    validAccu_history = []
    loss_valid = 0
    accuracy_valid = 0
    validLosses = []
    validAccuracies = []

    len_dataloader = min(len(sim_loader), len(exp_loader))
    len_dataloader1 = min(len(val_sim_loader), len(val_exp_loader))
    for epoch in range(num_epochs):
        sim_iter = iter(sim_loader)
        exp_iter = iter(exp_loader)

        start_steps = epoch * len_dataloader
        total_steps = num_epochs * len_dataloader

        #if (epoch // 2 * 2 != epoch):
        #    encoder.eval()
        #    classifier.eval()
        #else:
        #    encoder.train()
        #    classifier.train()

        tepoch = tqdm(range(len_dataloader), total=len_dataloader)
        for i_step in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            p = float(i_step + start_steps) / total_steps
            #alpha = 2. / (1. + np.exp(-10 * p)) - 1
            alpha = 0.5

            (s_x, s_y) = next(sim_iter)
            (e_x, _)   = next(exp_iter)
            s_x = s_x.to(device)
            s_y = s_y.long().flatten().to(device)
            e_x = e_x.to(device)

            combined_x = torch.cat((s_x,e_x),0).to(device)

            domain_label = torch.zeros(len(s_y)).long().to(device)
            domain_labele = torch.ones(len(e_x)).long().to(device)
            combined_domain_label = torch.cat((domain_label, domain_labele), 0).to(device)


            optimizer.zero_grad()

            combined_feature = encoder(combined_x)
            sim_feature = encoder(s_x)
            exp_feature = encoder(e_x)

            # 1.Classification loss
            s_class = classifier(sim_feature)
            s_class = s_class.to(device)
            class_loss = lossClass(s_class, s_y)

            # 2. Domain loss
            domain_pred = discriminator(combined_feature, alpha)
            domain_preds = discriminator(sim_feature, alpha)
            domain_prede = discriminator(exp_feature, alpha)
            #print(s_class)
            #print(domain_prede)
            domain_loss = lossDomain(domain_preds,domain_label) + lossDomain(domain_prede,domain_labele)
            #domain_loss = lossDomain(domain_pred, combined_domain_label)
            
            #if (epoch // 2 * 2 != epoch):
            #encoder.eval()
            #classifier.eval()
            total_loss = class_loss + 2*domain_loss
            #total_loss = domain_loss

            total_loss.backward()
            optimizer.step()
            
            #indicesD = torch.argmax(domain_pred, 1)
            #running_accD = torch.sum(indicesD == combined_domain_label)/(len(combined_domain_label))
            indicesDs = torch.max(domain_preds, 1)[1]
            running_accDs = torch.sum(indicesDs == domain_label)/(len(domain_label))
            indicesDe = torch.max(domain_prede, 1)[1]
            running_accDe = torch.sum(indicesDe == domain_labele)/(len(domain_labele))

            indices = torch.max(s_class, 1)[1]
            running_acc = torch.sum(indices==s_y)/len(s_y)

            train_history.append(1-running_acc.cpu())
            loss_history.append(float(total_loss))

            tepoch.set_postfix(loss=float(class_loss), lossDom=float(domain_loss), acc=float(running_acc)*100, accD=float(running_accDs)*100, accD1=float(running_accDe)*100)

        if scheduler is not None:
            scheduler.step()
        
        encoder.eval()
        encoder.cpu()
        torch.save(encoder.state_dict(), os.path.join('nndata','encoder' + dataSetType + '.pt'))
        encoder.to(device)
        encoder.train()

        classifier.eval()
        classifier.cpu()
        torch.save(classifier.state_dict(), os.path.join('nndata','classifier' + dataSetType + '.pt'))
        classifier.to(device)
        classifier.train()

        discriminator.eval()
        discriminator.cpu()
        torch.save(discriminator.state_dict(), os.path.join('nndata','discriminator' + dataSetType + '.pt'))
        discriminator.to(device)
        discriminator.train()

    # drawing step
    ep = np.arange(1,len(loss_history)+1,1)
    lt = np.array(loss_history)
    at = np.array(train_history)
    #vx = np.arange(len(loss_history)/len(validLoss_history),len(loss_history)+1,len(loss_history)/len(validLoss_history))
    vyl= np.array(validLoss_history)
    vya= np.array(validAccu_history)
    plt.clf()
    #plt.plot(ep,at,"orange",label="1-acc")
    plt.plot(ep,lt,"blue",label="lossTrain")
    #plt.plot(vx, vyl,"red",label="lossValid")
    plt.legend(loc=[0.5,0.6])
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.show()
    
    encoder.eval()
    classifier.eval()
    discriminator.eval()

    return 1


def train_NN(simulation_path, experiment_path):
    print("start nn training")
    
    batch_size = 1024*4

    dftCorr = pandas.read_parquet(os.path.join("nndata",simulation_path))
    dftCorr = dataManager.normalizeDataset(dftCorr).sample(frac=1.0).reset_index(drop=True) # with shuffling
    dataTable = dftCorr.sample(frac=0.8).sort_index()
    validTable = dftCorr.drop(dataTable.index)

    dftCorrExp = pandas.read_parquet(os.path.join("nndata",experiment_path))
    dftCorrExp = dataManager.normalizeDataset(dftCorrExp).sample(frac=1.0).reset_index(drop=True) # with shuffling
    dataTableExp = dftCorrExp.sample(frac=0.8).sort_index()
    validTableExp = dftCorrExp.drop(dataTableExp.index)
    
    train_dataset = My_dataset(load_dataset(dataTable))
    valid_dataset = My_dataset(load_dataset(validTable))

    exp_dataset = My_dataset(load_dataset(dataTableExp))
    exp_valset = My_dataset(load_dataset(validTableExp))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False)

    exp_dataLoader = DataLoader(exp_dataset, batch_size=int(batch_size), drop_last=False)
    exp_valLoader = DataLoader(exp_valset, batch_size=int(batch_size), drop_last=False)

    nClasses = dftCorr[list(dftCorr.columns)[-1]].nunique()
    print(nClasses)

    valueCounts = dftCorr[list(dftCorr.columns)[-1]].value_counts(normalize=True)
    print(dftCorr[list(dftCorr.columns)[-1]].value_counts(normalize=False))
    indicesWeights = valueCounts.index.to_numpy()
    valuesWeights = valueCounts.values
    weights = np.zeros(nClasses).astype(np.float32)
    for i in range(nClasses):
        weights[indicesWeights[i]] = math.sqrt(1./nClasses * 1./valuesWeights[i])
    weights[3] = weights[3]/2
    weights[2] = weights[2]/2.5
    print(weights)
    
    print(dftCorr.isnull().sum())
    input_dim = train_dataset[0][0].shape[0]
    print("input dim is   ", input_dim)

    del validTable, valid_dataset, dftCorr, batch_size
    del exp_dataset, dftCorrExp, exp_valset, dataTableExp, validTableExp

    #nn_model = Model(input_dim=input_dim, output_dim=nClasses)
    nn_model = DANN(input_dim=input_dim, output_dim=nClasses).type(torch.FloatTensor).to(device)

    encoder = Encoder(input_dim=input_dim, output_dim=64).type(torch.FloatTensor).to(device)
    classifier = Classifier(input_dim=64, output_dim=nClasses).type(torch.FloatTensor).to(device)
    discriminator = Discriminator(input_dim=64, output_dim=2).type(torch.FloatTensor).to(device)

    #exported_program = torch.export.export(DANN(input_dim=input_dim, output_dim=nClasses), (torch.randn(2,input_dim),))
    #torch.export.save(exported_program, 'exported_program.pt2')

    #weights = np.array([0.7195969431474389,0.8441318995839315,1.9450211571985525,8.533961043270507,0.5572980154693978]).astype(np.float32)
    #weights = np.array([0.6738684703630756,0.49427438150839936,2.2488611656428765,24.62766199493874,0.9810510497144537]).astype(np.float32)
    #weights = np.array([1.33, 1.64, 2.98, 60, 1]).astype(np.float32)
    #weights = np.array([1.776, 1.7275, 1.699, 33.7245, 0.582]).astype(np.float32)

    #weights = np.array([0.6738684703630756, 10 ,2.2488611656428765, 60 ,0.9810510497144537]).astype(np.float32)
    #weights = np.array([5.13, 1.54 ,0.41, 31.3 ,1]).astype(np.float32)
    loss = nn.CrossEntropyLoss(torch.tensor(weights)).to(device)
    #loss = nn.CrossEntropyLoss().to(device)
    #loss_domain = nn.CrossEntropyLoss().to(device)
    #loss = nn.MSELoss()
    #loss_domain = nn.NLLLoss()
    loss_domain = nn.CrossEntropyLoss().to(device)
    #loss_domain = nn.BCELoss()

    #optimizer = optim.SGD(nn_model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0)
    optimizer = optim.AdamW(list(encoder.parameters())+list(classifier.parameters())+list(discriminator.parameters()), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.000)
    #optimizer = optim.AdamW(discriminator.parameters(), lr=0.003, betas=(0.9, 0.999), weight_decay=0.0000)
    #optimizer = optim.AdamW(nn_model.parameters(), lr=0.00003, betas=(0.5, 0.9), weight_decay=0.0001)
    #optimizer = optim.Adam(list(encoder.parameters())+list(classifier.parameters())+list(discriminator.parameters()), lr=0.00003, weight_decay=0.0)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.2, factor=0.2)

    print("prepared to train nn")
    #train_DN_model(nn_model, train_loader, loss, optimizer, 10, valid_loader, scheduler = scheduler)
    train_Proper_DANN_model(encoder,classifier,discriminator, train_loader, exp_dataLoader, exp_valLoader, valid_loader, loss, loss_domain, optimizer, 5, scheduler)
    #train_DANN_model(nn_model, train_loader, exp_dataLoader, exp_valLoader, valid_loader, loss, loss_domain, optimizer, 3, scheduler=scheduler)

    torch.onnx.export(nn_model.cpu(),                                # model being run
                  torch.randn(1, input_dim),    # model input (or a tuple for multiple inputs)
                  os.path.join("nndata",'tempModel' + dataSetType + '.onnx'),           # where to save the model (can be a file or file-like object)
                  input_names = ["input"],              # the model's input names
                  output_names = ["class","domain"])            # the model's output names
    
    torch.onnx.export(encoder.cpu(),                                # model being run
                  torch.randn(1, input_dim),    # model input (or a tuple for multiple inputs)
                  os.path.join("nndata",'encoder' + dataSetType + '.onnx'),           # where to save the model (can be a file or file-like object)
                  input_names = ["input"],              # the model's input names
                  output_names = ["features"])            # the model's output names
    torch.onnx.export(classifier.cpu(),                                # model being run
                  torch.randn(1, 64),    # model input (or a tuple for multiple inputs)
                  os.path.join("nndata",'classifier' + dataSetType + '.onnx'),           # where to save the model (can be a file or file-like object)
                  input_names = ["features"],              # the model's input names
                  output_names = ["class"])            # the model's output names

    print("trained nn")
    #model_scripted = torch.jit.trace(nn_model,torch.tensor(np.array([load_dataset(dataTable)[0][0],load_dataset(dataTable)[0][1]])))
    #model_scripted.save('modelScr.pt')


print("start_train_python")

dataManager.manageDataset("train_dann")

#dataManager.compareInitialDistributions()

train_NN('simu' + dataSetType + '.parquet','expu' + dataSetType + '.parquet')

