To use my network:
___SETUP:

if you use header and source files:
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
//    bool nnInputIsGood(HParticleCand *x);
//    vector<float> get_NN_Input_Pars(HParticleCand *x);
//    vector<float> getPredictionProbability(vector<float> inputTensorValues);
// You can copy my #include files to your dir and edit