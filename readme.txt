To use my network:
___SETUP:

1. If you use header and source files:

a) In your .h file:
#include "/lustre/hades/user/vkladov/sub/expKKpiBatchFarm/HNetworkPID.h"
Inside analysis class:
HNetworkPID* networkPID;

b) In your .C or .cc file with all the analysis code:
#include "/lustre/hades/user/vkladov/sub/expKKpiBatchFarm/HNetworkPID.C"
networkPID = new HNetworkPID();

2. If you use only source:
#include "/lustre/hades/user/vkladov/sub/expKKpiBatchFarm/HNetworkPID.C"
networkPID = new HNetworkPID();

In the Makefile:

LIBS+=/lustre/hades/user/vkladov/packages/KinFit/lib/libonnxruntime.so

When launching the job, export myhaddir:
export MYHADDIR=/lustre/hades/user/vkladov/packages/KinFit
(or copy all onnx stuff to your lib if you use your own third-party libs)

___USAGE example:
// I use normalized pids: pi+, pi-, K+, K-, p respectively for (0,1,2,3,4) network output indices
// For each HParticleCand input, initially network returns 5 numbers which sum up to 1 (kind of probabilities)
    
    for (HParticleCand* x : particles){
        vector<int> pClass = networkPID->getPredictionLooseFull(x);    //will return a vector of PIDs which probabilities exceed 35% or something
        int pClassStrict = networkPID->getPredictionFull(x);                    //will return the PID with largest probability, if it exceed all other probabilities by 30%
    }

// In any case it will return -1 if input data is bad (I use some wide cuts, e.g. on momentum, beta < 1 etc)
// You can also separately check if data is good or bad, get vector of inputs to NN and get vector (5) of probabilities with these member functions:

    bool nnInputIsGood(HParticleCand *x);
    vector<float> get_NN_Input_Pars(HParticleCand *x);
    vector<float> getPredictionProbability(vector<float> inputTensorValues);


// You can copy my #include files to your dir and edit

!!!!!!! For now, there are problems at beta > 1, so if you want to identify pions, treat all particles with beta > 1 as them



# Network Usage Guide

## Setup

### 1. If you are using both header and source files:

#### In your `.h` file:
```
#include "/lustre/hades/user/vkladov/sub/expKKpiBatchFarm/HNetworkPID.h"
Inside the analysis class:
```
HNetworkPID* networkPID;
```
In your .C or .cc file with the analysis code:
```
#include "/lustre/hades/user/vkladov/sub/expKKpiBatchFarm/HNetworkPID.C"
networkPID = new HNetworkPID();
```
2. If you are using only the source file:
```
#include "/lustre/hades/user/vkladov/sub/expKKpiBatchFarm/HNetworkPID.C"
networkPID = new HNetworkPID();
```
3. In the Makefile:
Make sure to link the necessary library:
```
LIBS+=/lustre/hades/user/vkladov/packages/KinFit/lib/libonnxruntime.so
```
4. Export the environment variable MYHADDIR when launching the job:
```
export MYHADDIR=/lustre/hades/user/vkladov/packages/KinFit
```
Alternatively, if you use your own third-party libraries, you can copy all ONNX-related files to your own lib.

# Usage Example
The network uses normalized PID indices for particles:

pi+, pi-, K+, K-, p correspond to the network output indices: (0,1,2,3,4).
For each HParticleCand input, the network returns 5 probabilities that sum up to 1. Here's how you can use it:

```
for (HParticleCand* x : particles){
    // Returns a vector of PIDs with probabilities exceeding 35%
    vector<int> pClass = networkPID->getPredictionLooseFull(x);    

    // Returns the PID with the largest probability (if it exceeds the others by 30%)
    int pClassStrict = networkPID->getPredictionFull(x); 
}
```
## Error Handling:
The network will return -1 if the input data is invalid (for example, when values are out of expected bounds like momentum or beta > 1).
Additional Functions:
You can manually check the validity of the input, retrieve the input parameters for the neural network, and get the network's output probabilities:

```
bool nnInputIsGood(HParticleCand *x); // Check if input data is good
vector<float> get_NN_Input_Pars(HParticleCand *x); // Get input parameters to NN
vector<float> getPredictionProbability(vector<float> inputTensorValues); // Get vector of 5 probabilities


# Notes
You can copy the #include files into your own directory and modify them as needed.
Potentially problematic region: beta > 1. Now mostly fixed (the issue was in mass^2 which is now removed from the list of input parameters). 
If it is still the issue for you, treat all particles with beta > 1 as pions.