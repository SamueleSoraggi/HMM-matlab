% test for the Em algorithm with a single mixture of gaussians
close all; clear all; clc;

Starting = 'var';       %variational initialization
normalize_int = [];     %unnormalized data
soglia_em = 1e-4;       %increase's threshold for the loglikelihood
max_loop = 200;         %max number of loops
Hidden = [];            %randomly initialized hidden parameters
info = 'yes';           %print info
reg = 1e-5;            %regularization parameter
data = [];              %paint your own data (move the mouse while you
                        %press the left button,then press the right button)

[Hidden.trans_prob,Hidden.start_prob,Param,gmm_obj,loglike,track]=...
    GmmHmmEm([],Starting,normalize_int,soglia_em,max_loop,Hidden,info,reg);