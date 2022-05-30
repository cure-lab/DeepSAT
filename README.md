# DeepSAT: An EDA-Driven Learning Framework for SAT
Code repository for the paper:  
**DeepSAT: An EDA-Driven Learning Framework for SAT**  
Anonymous Author(s)

## Abstract
We present DeepSAT, a novel end-to-end learning framework for the Boolean satisfiability (SAT) problem. Unlike existing solutions trained on random SAT instances with relatively weak supervisions, we propose applying the knowledge of the well-developed electronic design automation (EDA) field for SAT solving. Specifically, we first resort to advanced logic synthesis algorithms to pre-process SAT instances into optimized and-inverter graphs (AIGs). By doing so, our training and test sets have a unified distribution, thus the learned model can generalize well to test sets of various sources of SAT instances. Next, we regard the distribution of SAT solutions being a product of conditional Bernoulli distributions. Based on this observation, we approximate the SAT solving procedure with a conditional generative model, leveraging a directed acyclic graph neural network with two polarity prototypes for conditional SAT modeling. To effectively train the generative model,  with the help of logic simulation tools, we obtain the probabilities of nodes in the AIG being logic `1' as rich supervisions. 
We conduct extensive experiments on various SAT instances. DeepSAT achieves significant accuracy improvements over state-of-the-art learning-based SAT solutions, especially when generalized to SAT instances that are large or with diverse distributions. 

## Installation
The experiments are conducted on Linux, with Python version 3.7.4, PyTorch version 1.8.1, and [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) version 2.0.1.

To set up the environment:
```sh
cd deepsat
conda create -n deepsat python=3.7.4
conda activate deepsat
pip install -r requirements.txt
```

Please download the dataset from [DeepSAT_Dataset](https://drive.google.com/file/d/1oIszUt2dIdzcKPRLya-2hujOrhSmvLfp/view?usp=sharing) to `./data`

## Running training code
Before training the model, please check the dataset in directory `./data`
The training dataset should be `sr3_10`

To train the DeepSAT, run the script `train.sh` in directory `./experiments`.

## Run evaluation code
Before evaluating the model, please check the dataset in directory `./data`
The testing dataset should be `sr10`, `sr20`, `sr40`, `sr60`, `sr80`

Then, check the trained model. 
The trained model has been in `./exp/deepsat/deepsat_sr3to10_gru/model_trained.pth`

Run the script `test.sh` in directory `./experiments`.

## Acknowledgements
This code uses [DAGNN](https://github.com/vthost/DAGNN), [D-VAE](https://github.com/muhanzhang/D-VAE), [SAT-RL](https://github.com/emreyolcu/sat) and [neurosat](https://github.com/dselsam/neurosat)/[NeuroSAT](https://github.com/ryanzhangfan/NeuroSAT) as backbone. We gratefully appreciate the impact these libraries had on our work. If you use our code, please consider citing the original papers as well. The code organization follows the one from [CenterNet](https://github.com/xingyizhou/CenterNet). Thanks!
