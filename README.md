# INFERENTIAL ANALYSIS OF THE CRITICAL TEMPERATURE IN SUPERCONDUCTORS ON DOCKER WITH SPARK

## ABSTRACT
This repository contains and analysis carried on for academic purposes, related to Big Data analysis infrastructures and techniques.
The main goal is to infer the relationship between the critical temperature of a [supercondutor](https://en.wikipedia.org/wiki/Superconductivity) material and its molecular structer,
in terms of both atomic characteristics and elements of composition. The analysis consists in the usage of machine learning models from
the pyspark library, simulating the computation on a fictitious computer cluster based on Docker containers.

## DATASET
Data source: ["Superconductivity Data"](https://archive.ics.uci.edu/dataset/464/superconductivty+data) from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/)

The dataset contains 21263 occurrences, representing different superconductor materials. It's divided in the following two csv files:
- **superconductivity.csv**: data about the atomic characteristics of the material (renamed from train.csv)
- **molec_structure.csv**: data about the quantity of every element from the periodic table used to build the material (renamed from unique.csv)
You can find both csv into the 'data' directory of this repository, but i suggest you to download them from the original source linked upward.

## INFRASTRUCTURE
The infrastructure is meant to simulate a computer cluster to compute the models with Spark and the pyspark library.
A masternode instance is created from the main Windows platform and 4 workernodes are connected to it, from a dedicated Docker containter each.

![cluster structure](https://github.com/Firefly55lm/superconductivity_lbd/blob/c7488811858e0c1c2109f0aab33e3128ef72c51c/pictures/simulated_cluster_architecture.png)

To run the masternode instance on Windows, locate from the terminal the directory '%SPARKHOME%/bin' and launch the command 'spark-class org.apache.spark.deploy.master.Master'.
You can access the Spark UI from your browser at the link given in the shell, usually at port 8080.
To run the workernodes from the linux container or from another hardware unit, run the command 'sudo ./start-worker.sh spark://<IP>:<port>' from your '%SPARKHOME%/sbin' directory,
where IP and port are the IP of your masternode on your LAN and its port. 
Then, you just need to link your pyspark session with the same IP and port and run the code.
Spark automatically parallelizes the computation over the cluster of workers. The same result can be accomplished with a traditional computer cluster with real different hardware.

## METHODOLOGY
In the notebook of the root directory of branch main you can find the implementation of the following techniques:
- Regularized Linear Regression: to better handle the collinearity between the features, keeping the interpretability of a simple model, I implemented
a simple regularized linear regression model, organizing a grid search to find the better hyperparameters. The latters are a reg_param, which is the lambda
coefficient to control the shrinking of the regression's coefficients, and an elasticnet_param, which controls the regularization type. With a value of 0
of the elasticnet parameter, an L2 regularization is performed (Ridge), with a value of 1 it's L1 regularization (Lasso) and with 0.5 is a combination of
the other two (Elasticnet). For the evaluation, the $R^2$ metric is used, estimated with a 5-fold Cross Validation. I could not implement a more sofisticated model,
such as a Kernel Ridge Regression, because of the lack of choise granted by pyspark;
- PCA - Principal Component Analysis: unsupervised learning technique useful to decrease the dimensionality of the data, performed on the predictors. The plots
are colorized by the critical temperature to observe if the PCA dimensions can track a pattern to explain the Y variable.

Both techniques are implemented in different notebooks for both the available csv files.

## RESULTS
The best hyperparameters for the regularized linear regression on superconductivity.csv are a lambda of 0.001 and a Ridge regularization, with a $R^2$ of 0.735.
On the other side, on molec_structure.csv data, the best combination is a lambda of 1 combined with a Lasso regularization, with a low $R^2$ of just 0.569.

You can find the details about the coefficients and the related features in the dedicated notebooks.

More interesting are the results of the PCA models. We can observe a few patterns: in the first and main dataset, the first two dimensions are more related to informazion
about the atomic structure of the material, then the other 3 are more related to thermal and energetic information.
We can also apreciate the fact that the main PCA's dimension has a significant discerning power of explaining the variability of the response variable.

![3D PCA superconductivity](https://github.com/Firefly55lm/superconductivity_lbd/blob/f0fed6c4fb482a7b868547aee09bc9a512c6ae09/pictures/3D_PCA_plot_superconductivity.png)
![2D PCA superconductivity](https://github.com/Firefly55lm/superconductivity_lbd/blob/f0fed6c4fb482a7b868547aee09bc9a512c6ae09/pictures/2D_PCA_plot_superconductivity.png)

The findings on the molec_structure.csv data are even more particular. In fact, looking at the first five most important features, sorted by absolute value of the coeffient,
we can observe that the first three dimensions correspond to [common categories of superconductors](https://it.wikipedia.org/wiki/Superconduttivit%C3%A0#Classificazione_chimica):
- Dimension 0: cuprates (O, Cu, Ba, Y)
- Dimension 1: metallic superconductors (As, Fe, PT)
- Dimension 2: fullerenes (C, Rb, K, H)
You can find more specific information about the elements [here](https://pubchem.ncbi.nlm.nih.gov/periodic-table/).

![3D PCA molecular structure](https://github.com/Firefly55lm/superconductivity_lbd/blob/f0fed6c4fb482a7b868547aee09bc9a512c6ae09/pictures/3D_PCA_plot_molec_structure.png)
![2D PCA molecular structure](https://github.com/Firefly55lm/superconductivity_lbd/blob/f0fed6c4fb482a7b868547aee09bc9a512c6ae09/pictures/2D_PCA_plot_molec_structure.png)
