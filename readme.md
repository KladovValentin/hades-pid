# How to use

## Setup

### 1. If you are using both header and source files:

In your `.h` file:
```
#include "/lustre/hades/user/vkladov/sub/expKKpiBatchFarm/HNetworkPID.h"
HNetworkPID* networkPID;
```
In your .C or .cc file with the analysis code:
```
#include "/lustre/hades/user/vkladov/sub/expKKpiBatchFarm/HNetworkPID.C"
networkPID = new HNetworkPID();
```
### 2. If you are using only the source file:
```
#include "/lustre/hades/user/vkladov/sub/expKKpiBatchFarm/HNetworkPID.C"
HNetworkPID* networkPID = new HNetworkPID();
```
### 3. In the Makefile:
Make sure to link the necessary library (or copy it to any of your directories first):
```
LIBS+=/lustre/hades/user/vkladov/packages/KinFit/lib/libonnxruntime.so
```
### 4. Export the environment variable MYHADDIR when launching the job:
```
export MYHADDIR=/lustre/hades/user/vkladov/packages/KinFit
```
Alternatively, if you use your own third-party libraries, you can copy all ONNX-related files to your own directory that you export as MYHADDIR (include/ and source/).

## Usage details
* The network uses normalized PID indices for particles:
pi+, pi-, K+, K-, p correspond to the network output indices: (0,1,2,3,4).
* For each HParticleCand input, the network returns 5 probabilities that sum up to 1. Here's how you can use it:

```
for (HParticleCand* x : particles){
    // Returns a vector of PIDs with probabilities exceeding 35% (max 2 per particle)
    vector<int> pClass = networkPID->getPredictionLooseFull(x);    

    // Returns the PID with the largest probability (if it exceeds the others by 30%)
    int pClassStrict = networkPID->getPredictionFull(x); 
}
```
## Notes:
* The network will return -1 if the input data is invalid (too off: e.g. beta > 1.5, dedx > 50 etc).
* You can manually check the validity of the input, retrieve the input parameters for the neural network, and get the network's output probabilities:
```
bool nnInputIsGood(HParticleCand *x); // Check if input data is good
vector<float> get_NN_Input_Pars(HParticleCand *x); // Get input parameters to NN
vector<float> getPredictionProbability(vector<float> inputTensorValues); // Get vector of 5 probabilities
```

* Example of how I use:
```
for (HParticleCand* x : negativeHades){
    vector<int> pClass = networkPID->getPredictionLooseFull(x);
    int pClassStrict = networkPID->getPredictionFull(x);  // only with a highest 
    for (int pC: pClass){
        if (pC == 3){
            kaonsNIdent.push_back(x);
        }
    }
    if (pClassStrict == 3){
        hMKN->Fill(x->getMass2()* 1e-6);
    }
}
```

* You can copy the #include files into your own directory and modify them as needed.
* Potentially problematic region: beta > 1. Now mostly fixed (the issue was in mass^2 which is now removed from the list of input parameters). If it is still the issue for you, treat all particles with beta > 1 as pions.


# Plots and purity-efficiency estimations