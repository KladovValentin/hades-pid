#include "/home/localadmin_jmesschendorp/onnxruntime-linux-x64-1.14.1/include/onnxruntime_cxx_api.h"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <bits/stdc++.h>

using namespace std;

void softmax(vector<float>& input) {
  auto rowmax = *max_element(begin(input), end(input));
  std::vector<float> y(input.size());
  float sum = 0.0f;
  for (size_t i = 0; i != input.size(); ++i) {
    sum += y[i] = exp(input[i] - rowmax);
  }
  for (size_t i = 0; i != input.size(); ++i) {
    input[i] = y[i] / sum;
  }
}

void dosmth(){
    Ort::Env env;
    std::string model_path = "tempModel.onnx";
    Ort::Session* mSession = new Ort::Session(env, model_path.c_str(), Ort::SessionOptions{ nullptr });
    
    const char* mInputName[] = {"input"};
    Ort::TypeInfo inputTypeInfo = mSession->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::vector<int64_t> mInputDims = inputTensorInfo.GetShape();

    const char* mOutputName[] = {"output"};
    Ort::TypeInfo outputTypeInfo = mSession->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::vector<int64_t> mOutputDims = outputTensorInfo.GetShape();
    cout << mOutputDims[0] << "  " << mOutputDims[1] << endl;

    size_t inputTensorSize = mInputDims[1];
    std::vector<float> inputTensorValues = {-0.493264, 0.756992, -0.381234, -0.609464, -0.356375, 0.000000, -0.473017, 0.354888, 0.014933, 0.316937, -0.620801}; // 0 (pi+)
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),inputTensorSize,mInputDims.data(),mInputDims.size());

    size_t os = 0;
    for (size_t i = 0; i < mOutputDims.size(); i++)
        os = os + mOutputDims[i];
    size_t outputTensorSize = os;
    vector<float> outputTensorValues(outputTensorSize);
    // Create output tensors of ORT::Value
    Ort::Value outputTensor = Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(), outputTensorSize,mOutputDims.data(), mOutputDims.size());

    // 1 means number of inputs and outputs
    mSession->Run(Ort::RunOptions{nullptr}, mInputName,&inputTensor, 1, mOutputName,&outputTensor, 1);
    vector<float> outputProbs;
    for (size_t i = 0; i < outputTensorSize-1; i++){
      outputProbs.push_back(outputTensorValues[i]);
    }
    softmax(outputProbs);
    for (size_t i = 0; i < outputProbs.size(); i++){
        cout << outputProbs[i] << endl;
    }
}