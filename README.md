# Neural-networks-for-non-life-insurance-pricing
This page contains the code used for my thesis: Neural networks for non-life insurance pricing. The goal of my thesis is to compare neural networks and combined actuarial neural networks (CANNs) with generalized linear models (GLMs), generalized additive models (GAMs) and several tree-based machine learning methods being normal regression trees, random forests and gradient boosting machines (GBMs). Currently the text of the thesis is not yet available, a link to the text will be added later. We compare the different models  for a motor third party liability data set. The GLMs, GBMs and tree-based machine learning models were earlier constructed for both claim frequency and claim severity in Henckaerts, R., Antonio, K., Côté, M. P., & Verbelen, R. (2019): Boosting insights in insurance tariff plans with tree-based machine learning (https://arxiv.org/abs/1904.10890). 
# Overview of the included files 
This page contains four main files:
1. Neural_network_and_CANN_tuning: we tune a neural network and CANN using a sequential approach and cross-validation. 
2. Neural_networks: Compute the partial dependence plots, variable importance plots, predictions and test losses (from the cross-validation) for neural networks and bias regulated neural networks.
3. CANNs: Compute the partial dependence plots, variable importance plots, predictions and test losses (from the cross-validation) for fixed, flexible and bias regulated neural networks.
4. Plots_and_model_lifts: contains the construction of every plot and visualization used in the thesis (model lifts, ordered Lorenz curves, comparison of out-of-sample predictions and some extra partial dependence plots and variable importance plots).

The data needed used in these files is also included.
