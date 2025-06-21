## Introduction

## Experiment Setup

### Hardware environment

> [!IMPORTANT]
> 
> **General test-bed requirements**
> - **OS**: Ubuntu >= 20.04;
> - **CPU**: X86/X64 CPU;
> - **GPU**: CUDA architecture (V100, A6000, A100, etc.);
> - **Memory**: 128GB GPU Memory available (if you use 72B model with vLLM);
> - **Storage**: 500GB Storage available;
> - **Network**: Good Network to GitHub and Docker Hub;



### Software requirement

You need a DeepSeek API key to invoke the DeepSeek API service (of course you can modify configuration in [./config/model.yaml](./config/model.yaml))



## Get started
#### download the repository
Please download our repository and go to the first tier directory.

#### Install dependencies

Firstly, we should install some necessary python dependencies.
We strongly recommend users use `conda` to manage the python environments.
Please follow the below commands.

```shell
conda create -n fuel python=3.12
pip install requirements.txt
```

#### Install PyTorch nightly version
When fuzzing the systems under tests (SUTs), we use the nightly version, in order to detect new bugs.

Here we use CUDA 12.6 as an example. Please install the nightly version based on your CUDA version. You can get the corresponding commands from https://pytorch.org/
```shell
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
```

#### create Openai API key
In our experiment, we use DeepSeek API to invoke the LLM service. DeepSeek API service is compatible with openai interfaces.

For the below command, you should replace `[YOUR_API_KEY]` with your own DeepSeek API key.
```shell
key="[YOUR_API_KEY]"
echo "$key" > ./config/llm-key.txt
```

#### Start fuzzing
```shell
python fuel/fuzz.py --lib pytorch run_fuzz \
                    --max_round 1000 \
                    --heuristic SA \
                    --validate compiler
```
Note that the fuzzing experiment is really time-consuming. Maybe you should check the results after about ~20hours.

#### Check results
Please check the generated models in [results/fuel/pytorch](results/fuel/pytorch).
If you want to get the detected bugs, please check [outputs/bug_reports.txt](outputs/bug_reports.txt).


#### Bug finding

So far, FUEL has detected previously **84** unknown new bugs, with **73** already confirmed and **36** already fixed. Five detected bugs were labeled as *high-priority*, and one was labeled as *good first issue*.

The evidence is here: