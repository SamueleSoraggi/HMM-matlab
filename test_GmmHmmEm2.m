% test for the Em algorithm with observation conditionally distributed as
% mixtures of gaussians. 
% the tool to plot a dataset is not always working fine, sometimes the
% program fail because the data is crappy. Try plotting again data :D
%close all; clear all; clc;

Clustering = 'kmvar';   %variational EM + kmeans initialization of the chain
Starting = 'var';       %variational EM for  initialization of the mixtures 
normalize_int = []; 	%unnormalized data
soglia_em = 1e-3;       %threshold increase for the loglikelihood
max_loop = 100;         %max number of loops
Hidden = [];            %randomly initialized hidden parameters
info = 'yes';           %print info
reg = 1e-5;             %regularization parameter
data = [];              %paint your own data (move the mouse while you
                        %press the left button,then press the right button)
printOpt = 0;           %make plots

[new_trans_prob,new_start_prob,Param,gmm_obj,loglike,track,track2]=...
GmmHmmEm2(data,Clustering,Starting,normalize_int,soglia_em,max_loop,Hidden,info,printOpt,reg);