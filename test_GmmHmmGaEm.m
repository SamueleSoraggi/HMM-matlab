%Hybrid Genetic-EM algorithm for HMMs with observations described by a 
%single GMM.

clear all; close all; clc;

normalize_int = [];     %unnormalized data
em_iter = 3;            %EM iteration during the genetic procedure
mt = .1;                %mutation rate
ct = .7;                %cross-over rate
pop = 10;               %population
Q_max = 15;             %max number of states
cxover_mode = 'single point'; %cross-over mode
data = [];              %paint your own data (move the mouse while you
                        %press the left button,then press the right button)


[trans,prior,gmm,LLKgen,MDL_min,track]=...
GmmHmmGaEm(data,normalize_int,em_iter,mt,ct,pop,Q_max,cxover_mode);