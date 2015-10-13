% test for the Em algorithm with multiple mixtures of gaussians
% (in this version each GMM is evaluated ONLY in the observations
% that lay in the cluster of the GMM. the result is a little less precise
% but the algorithm is fast)
close all; clear all; clc;

Clustering = 'kmvar';   %variational+kmeans initialization of the chain
Starting = 'var';       %variational initialization of the mixtures 
normalize_int = []; 	%unnormalized data
soglia_em = 1e-3;       %increase's threshold for the loglikelihood
max_loop = 300;         %max number of loops
Hidden = [];            %randomly initialized hidden parameters
info = 'yes';           %print info
reg = 1e-5;             %regularization parameter
data = [];              %paint your own data (move the mouse while you
                        %press the left button,then press the right button)

[new_trans_prob,new_start_prob,Param,gmm_obj,loglike,track,track2]=...
GmmHmmEm2(data,Clustering,Starting,normalize_int,soglia_em,max_loop,Hidden,info,reg);

