# Dimensionality Reduction and Classification in High-Dimensional Data: A Hybrid Approach Using Generalized Matrix LVQ and Deep Learning Techniques

This is the code on which my master's thesis is based on and it is a continuation of ["Autoencoder-based Interpretable Classification using LVQ and Relevance Learning"](https://github.com/chadji15/LVQ_Autoencoder). This thesis addresses the challenge of explainability in high-dimensional data classification by proposing a hybrid approach that integrates Generalized Matrix Learning Vector Quantization (GMLVQ) with deep learning techniques, specifically autoencoders. The proposed method leverages the strengths of GMLVQ in providing interpretability through prototype-based classification, combined with the dimensionality reduction capabilities of autoencoders. By making use of the decoder part, the approach aims to map the reduced-dimensional space back to the original feature space, thus offering a transparent view of the features influencing the classification outcomes.

## Setup


For the experiments, Matlab 2024a was used.

### Required Toolboxes:
- Deep Learning Toolbox
- Deep Learning HDL Toolbox
- Parallel Processing Toolbox
- Image Processing Toolbox
- Optimization Toolbox
- Statistics and Machine Learning Toolbox
- No-nonsense GMLVQ (https://www.cs.rug.nl/~biehl/gmlvq)
  
The GMLVQ toolbox needs to be added to the Matlab path. Further instruction on that can be found in the provided link.

### Folder structure:
Create two new folders in the same level as "src" named "data" (for the datasets) and "models" (for the outputs).

### Datasets: 
Put these datasets in their own folders under the "data" directory. 
- MNIST dataset for matlab: https://lucidar.me/en/matlab/load-mnist-database-of-handwritten-digits-in-matlab/
- FashionMNIST: https://github.com/zalandoresearch/fashion-mnist
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html


Generally, the top-level directory should look like this:

        .
        ├── src                     # Source code
            └── ...
        ├── models                  # Output .mat files
        ├── data                    # Datasets
            ├── mnist
            ├── fashionMNIST
            └── cifar-10-batches-mat
        └── README.md
    


    

## Execution:

Details for execution and parameters are written inside the scripts themselves. Code in the _util_ and _architecture_ directories are building blocks and utilities that allow the core experiments / scripts to run. The _architecture_ directory contains the implementations for all the versions of autoencoders we used, while code in _util_ and _layers_ aids these implementations, provides easy access to the datasets and implements other auxiliary functions. 

The main logic and workflow is contained within the scripts directly under the _src_ directory and the _src/scripts_ directory. The name and comment description of each one provides more details for their utility.

## Attributions
These files were taken from Matlab's Train Variational Autoencoder (VAE) to Generate Images example:

- samplingLayer.m
- projectAndReshape.m
- processLabelsMNIST.m
- processImagesMNIST.m
- initializeZeros.m
- initializeGlorot.m
- modelLoss.m
- elboLoss.m
- modelPredictions.m
- preprocessMiniBatch.m

These files were taken from Matlab's Train Object Detector Using R-CNN Deep Learning example:

- helperCIFAR10Data.m