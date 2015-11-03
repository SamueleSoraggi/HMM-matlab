%EM algorithm for HMMs with observation described by multiple GMMs and
%discrete hidden states
%    [new_trans_prob,new_start_prob,Param,gmm_obj,loglike,track,track2]=...
%    GmmHmmEm2(data,Clustering,Starting,normalize_int,soglia_em,...
%    max_loop,Hidden,info,reg,printOpt,varargin)
%
% Input:
%   data : NxD vector, where N is the number of observations and D their
%          dimension. If you set data=[], a simulator window will be opened
%          and you can paint your own trajectory with the mouse.
%   clustering : clustering method for the hidden states
%          write a number Q to make the kmeans on that number of hidden
%          states
%          kmvar = kmeans with the number of states set by the variational
%                  EM
%          var = initialization with the variational EM
%   starting : a string that set the initialization of the gaussian mixtures 
%          kmvar = kmeans with the number of states set by the variational
%                  EM
%          var = initialization with the variational EM
%          write a vector of length Q with the number of gaussians on which
%          you want to do the kmeans for each hidden state
%          write a number to have the same number of gaussians in each
%          state
%   normalize_int : normalization interval of the dataset, of the type
%          [X1_min X2_min ... XD_min ;X1_max X2_max ... XD_max]. 
%          If you set [], the data will remain unnormalized.
%   soglia_em : threshold of the increasing of the likelihood. If you are
%          below the threshold, the algorithm will stop.
%   max_loop  : max number of iterations
%   Hidden : matlab structure which contains Hidden.trans_prob,Hidden.start_prob 
%          to initialize the markov chain. If you set [], the parameters
%          will be randomly initialized.  
%   info : if you set 'yes', you will see the information of the model for
%          each iteration. If you write any other string, it means that the
%          information will be hidden.
%   reg  : parameter of regularization for the covariances.
%   printOpt: 1 to plot the evolution of the EM.
%       OPZIONAL INPUTS
%   idx : indexes of the clustered observations.
%         It can be inserted only if the input of "Clustering" is a struct
%         which contains the dataset subdivision. For example, idx can
%         comes from the dataset as it follows (kmeans)
%
%                     Clustering = {};
%                     [idx,ctrs] = kmeans(track,4);
%                     for j = 1:4
%                         Clustering{j} = track(idx==j,:);
%                     end
%
% Output:
%   new_trans_prob : Transition matrix for the hidden chain
%   new_start_prob : Starting probabilities for the hidden chain
%   Param : structure with the mixture's parameters ''mu,sigma,mix''
%   gmm_obj : obj gmdistribution which represents the matlab object for a
%           mixture of gaussians
%   track : normalized/unnormalized data
%   track2 : data labeled with the different hidden states
%   loglike: loglikelihood for each iteration of the EM
 
function [new_trans_prob,new_start_prob,Param,gmm_obj,loglike,track,track2]=...
       GmmHmmEm2(data,clustering,starting,normalize_int,soglia_em,max_loop,Hidden,info,reg,varargin)
     
addpath(genpath('./My Rubbish'))  

nVarargs = length(varargin);
if nVarargs == 0, idx = []; end
if nVarargs == 1, idx = varargin{1}; end

% Print and organize the dataset
if ~isempty(data) && ~isempty(normalize_int)
    track=My_normalize_data(data,1,normalize_int);
end
if ~isempty(data) && isempty(normalize_int)
    track=data;
end
if isempty(data) && ~isempty(normalize_int)
    [full_data,~] = grabDataFromCursorDynamics;
    data = full_data(1:5,:);
    track=My_normalize_data(data',1,normalize_int);
end
if isempty(data) && isempty(normalize_int)
    [full_data,~] = grabDataFromCursorDynamics;
    track = full_data(1:5,:)';
end

%separate the dataset into clusters
track2 = {}; %new dataset

[N D] = size(track);

if isstr(clustering)
 %Kmeans + Variational   
 if strcmp(clustering , 'kmvar')  
    clear options;
    options.maxIter = 200;
    options.threshold = 1e-5;
    % 1 = shows 2D plot of each iteration, 0 = otherwise
    options.displayFig = 0; 
    % 1 = shows which iteratin is running, 0 = otherwise
    options.displayIter = 0; 

    PriorPar.alpha = 0.001; %0.001
    if ~isempty(normalize_int)
        PriorPar.mu = normalize_int(1,:)';
    else
        PriorPar.mu = min(track)';
    end
    PriorPar.beta = 1;
    PriorPar.v = 3*D;
    PriorPar.W = inv(PriorPar.v * cov(track) + reg*eye(D));

    [~, ~, out_gmm_obj, ~] =...
        fn_VBGMM_clustering(track, 12, PriorPar, options);

    Q = size(out_gmm_obj.PComponents,2);
    
    % traditional clustering with Kmeans, 
    % saving indexes and centroids
    [idx,ctrs] = kmeans(track,Q);
    for j = 1:Q
        track2{j} = track(idx==j,:);        
    end    
 else
    error('Error in the ''clustering'' option.')
 end
elseif isnumeric(clustering)
    %Kmeans   
    Q=clustering;   
    % traditional clustering with Kmeans, 
    % saving indexes and centroids
    [idx,ctrs] = kmeans(track,Q);
    for j = 1:Q
        track2{j} = track(idx==j,:);
    end
 else
     track2 = clustering;
     Q = length(track2);
end

 
%% GMMs initializations
Param = {};      % struct of the parameters
Q2 = zeros(1,Q); % Q2(j) = number of gaussians in the j-th state

for j=1:Q
    if isstr(starting)
     %Variational method
     if strcmp(starting , 'var')
        clear options;
        options.maxIter = 200;
        options.threshold = 1e-5;
        % 1 = shows 2D plot of each iteration, 0 = otherwise
        options.displayFig = 0; 
        % 1 = shows which iteratin is running, 0 = otherwise
        options.displayIter = 0; 

        PriorPar.alpha = 0.0001;
    if ~isempty(normalize_int)
        PriorPar.mu = normalize_int(1,:)';
    else
        PriorPar.mu = min(track2{j})';
    end
    PriorPar.beta = 1;
    PriorPar.v = 3*D;
    PriorPar.W = inv(PriorPar.v * cov(track2{j}) + reg*eye(D));
    
    
        [~, ~, out_gmm_obj, ~] =...
            fn_VBGMM_clustering(track2{j}, 10, PriorPar, options);

        Q2(j) = size(out_gmm_obj.PComponents,2);

        Param{j}.sigma = out_gmm_obj.Sigma;
        Param{j}.mu = out_gmm_obj.mu';
        Param{j}.mix = out_gmm_obj.PComponents;
     %Kmeans + variational
     elseif strcmp(starting , 'kmvar')  
        clear options;
        options.maxIter = 200;
        options.threshold = 1e-5;
        % 1 = shows 2D plot of each iteration, 0 = otherwise
        options.displayFig = 0; 
        % 1 = shows which iteratin is running, 0 = otherwise
        options.displayIter = 0; 

        PriorPar.alpha = 0.0001; %0.001
        if ~isempty(normalize_int)
            PriorPar.mu = normalize_int(1,:)';
        else
            PriorPar.mu = min(track2{j})';
        end
    PriorPar.beta = 1;
    PriorPar.v = 3*D;
    PriorPar.W = inv(PriorPar.v * cov(track2{j}) + reg*eye(D)); %200

        [~, ~, out_gmm_obj, ~] =...
            fn_VBGMM_clustering(track2{j}, 10, PriorPar, options);

        Q2(j) = size(out_gmm_obj.PComponents,2);
        
        %Kmeand
        [Param{j}.mix, Param{j}.mu, Param{j}.sigma] =...
            My_GmmInitKM(track2{j},Q2(j));
     else
        error('Unexisting initialization option.')
     end
    elseif length(starting)>1 && ~isstr(starting)
     Q2(j) = starting(j);
     [Param{j}.mix, Param{j}.mu, Param{j}.sigma] = My_GmmInitKM(track2{j},Q2(j));
    elseif length(starting)==1 && ~isstr(starting)
     Q2(j) = starting;
     [Param{j}.mix, Param{j}.mu, Param{j}.sigma] = My_GmmInitKM(track2{j},Q2(j));  
    else
        error('No initialization option.')   
    end
end

% creation of the structure gmm_obj, containing the matlab objects for GMMs
gmm_obj = {};
for j = 1:Q
    Param{j}.sigma = Param{j}.sigma + repmat(reg*eye(D),[1,1,Q2(j)]);
    gmm_obj{j} = gmdistribution(Param{j}.mu',Param{j}.sigma,Param{j}.mix);
end

%Painting :-)
if D==2
    colori = jet(length(Param));
    figura=figure('Position', get(0,'ScreenSize'));
    hold on
    for j = 1:length(Param)
        plot(track2{j}(:,1),track2{j}(:,2),'Color',colori(j,:),'LineStyle','*','LineWidth',2)
        for i = 1:length(Param{j}.mix)
                MyEllipse(Param{j}.sigma(:,:,i), Param{j}.mu(:,i),'style','k','intensity',Param{j}.mix(i), 'facefill',.8);
                text(Param{j}.mu(1,i), Param{j}.mu(2,i), num2str(j),'BackgroundColor', [.7 .9 .7]);
        end
    end
    hold off
    s1 = sprintf('Gmm with initialized parameters');
    s2 = sprintf('\n %d Hidden states');
    title(strcat(s1,s2))
    drawnow;
else    
    colori = jet(length(Param));
    figura=figure('Position', get(0,'ScreenSize'));
    hold on
    for j = 1:length(Param)
        plot(track2{j}(:,2),track2{j}(:,3),'Color',colori(j,:),'LineStyle','*','LineWidth',2)
        for i = 1:length(Param{j}.mix)
            S = Param{j}.sigma(:,:,i); S = S([2 3],[2 3]);
            M = Param{j}.mu(:,i); M = M([2,3]);
            MyEllipse(S, M,'style','r','intensity',Param{j}.mix(i), 'facefill',.8);
            text(M(1), M(2), num2str(j),'BackgroundColor', [.7 .9 .7]);
        end
    end
    hold off
    s1 = sprintf('Gmm with initialized parameters');
    s2 = sprintf('\n %d Hidden states');
    title(strcat(s1,s2))
    drawnow;    
end

%Hidden parameters
if isempty(Hidden)  
    new_trans_prob = My_normalizza(rand(Q,Q));
    new_start_prob = My_normalizza(rand(1,Q));
else
    new_trans_prob = Hidden.trans_prob;
    new_start_prob = Hidden.start_prob;
end

Nu = zeros(N,Q); %responsibilities
fine = 0; %end flag 
loglike = [];
cont = 0;

 while fine == 0 
    
    trans_prob=new_trans_prob;
    start_prob=new_start_prob;
    cont = cont + 1;
    
    %% E-step: forward-backward
   
    %"density" matrix of the observations
    B = zeros(N,Q);
    for j = 1:Q  
          f = @(x)pdf(gmm_obj{j},x);
          B(:,j) = f(track);
    end
    
    %rescaled forward
    [alpha,c,c_prod,P_OL,loglike(cont)] =...
        My_forward_gmm(trans_prob,start_prob,B,N,Q);
    %rescaled backward
    [beta,d_prod] =...
        My_backward_gmm(trans_prob,B,N,Q,c);
    
    %% M-step:
    %Baum-Welch iteration for the hidden states
    [new_trans_prob,new_start_prob] =... 
        My_hidden_states_gmm(trans_prob,alpha,beta,Q,B,c,N);
    %normalizing
    new_start_prob = My_normalizza(new_start_prob);
    new_trans_prob = My_normalizza(new_trans_prob);
    %print info
    if strcmp(info,'yes')
        str = sprintf('Loglike: %.5f at the step %d',loglike(cont),cont);
        disp(str);    
    end
    %calculating responsibilities and saving them in Nu
    Nu = {};
    num = alpha.*beta;
    den = sum(num,2);     
   
    for k = 1:Q
        Nu{k} = zeros(size(track2{k},1),Q2(k)); 
        f = @(x)pdf(gmm_obj{k},x);
        den2 = f(track2{k});
        for j=1:Q2(k)       
             Nu{k}(:,j)= (num(idx==k,k)./den(idx==k)) .*...
             (Param{k}.mix(j)*mvnpdf(track2{k},Param{k}.mu(:,j)',Param{k}.sigma(:,:,j))) ./ den2;
        end
    end
       
    %EM for the GMMs
    sum_Nu = {};
    
    for j = 1:Q
        sum_Nu = sum(Nu{j},1);
        for k=1:Q2(j)
        % means and weights of the mixtures
            Param{j}.mix(k) = sum_Nu(k) / sum(sum_Nu);
            Param{j}.mu(:,k) = sum(repmat(Nu{j}(:,k),1,D) .* track2{j}) / sum_Nu(k); 
            % variances
            Param{j}.sigma(:,:,k) = zeros(D);
                for n=1:size(track2{j},1)
                    Param{j}.sigma(:,:,k) = Param{j}.sigma(:,:,k)+(Nu{j}(n,k)...
                    *kron((track2{j}(n,:)-Param{j}.mu(:,k)')',(track2{j}(n,:)-Param{j}.mu(:,k)')));
                end     
            Param{j}.sigma(:,:,k) = Param{j}.sigma(:,:,k)/sum_Nu(k)+reg*eye(D);
        end
        gmm_obj{j}=gmdistribution(Param{j}.mu',Param{j}.sigma,Param{j}.mix);
    end
    
    %Painting :-)
    if D==2
        clf(figura); 
        colori = jet(Q);
        subplot(3,4,[1 2 3 5 6 7 9 10 11])
        hold on
        for j = 1:Q
            plot(track2{j}(:,1),track2{j}(:,2),'Color',colori(j,:),'LineStyle','*','LineWidth',2)
            for i = 1:length(Param{j}.mix)
                    MyEllipse(Param{j}.sigma(:,:,i), Param{j}.mu(:,i),'style','r','intensity',Param{j}.mix(i), 'facefill',.8);
                    text(Param{j}.mu(1,i), Param{j}.mu(2,i), num2str(j),'BackgroundColor', [.7 .9 .7]);
            end
        end
        hold off
        s1 = sprintf('Gmm (projected) at the step %d',cont);
        s2 = sprintf('\nLogLikelihood = %f',loglike(cont));
        s3 = sprintf('\n%d Hidden states',Q);
        title(strcat(s1,s2,s3))
        subplot(3,4,4)
        hold on
        axis([0 Q+1 0 Q+1])
        imagesc(flipud(new_trans_prob))
        hold off
        title('Transition matrix')
        subplot(3,4,8)
        hold on
        bar(new_start_prob')
        hold off
        title('Starting probabilities') 
        subplot(3,4,12)
        plot(loglike)       
        title('loglikelihood')        
        drawnow;
        pause(0.02);
    else
        clf(figura); 
        colori = jet(Q);
        subplot(2,3,[1 2 4 5])
        hold on
        for j = 1:Q
            plot(track2{j}(:,2),track2{j}(:,3),'Color',colori(j,:),'LineStyle','*','LineWidth',2)
            for i = 1:length(Param{j}.mix)
                    S = Param{j}.sigma(:,:,i); S = S([2 3],[2 3]);
                    M = Param{j}.mu(:,i); M = M([2,3]);
                    MyEllipse(S, M,'style','r','intensity',Param{j}.mix(i), 'facefill',.8);
                    text(M(1), M(2), num2str(j),'BackgroundColor', [.7 .9 .7]);
            end
        end
        hold off
        s1 = sprintf('Gmm (projected) at the step %d',cont);
        s2 = sprintf('\nLogLikelihood = %f',loglike(cont));
        s3 = sprintf('\n%d Hidden states',Q);
        title(strcat(s1,s2,s3))
        subplot(3,4,4)
        hold on
        axis([0 Q+1 0 Q+1])
        imagesc(flipud(new_trans_prob))
        hold off
        title('Transition matrix')
        subplot(3,4,8)
        hold on
        bar(new_start_prob')
        hold off
        title('Starting probabilities')
        subplot(3,4,12)
        plot(loglike)  
        title('Loglikelihood')
        drawnow;
        pause(0.02);    
    end
      
    %Convergence control
    if cont > 1
       if abs(loglike(cont)-loglike(cont-1)) < soglia_em, fine=1; end
       if cont >= max_loop, fine=1; end
    end
        
end


     
     
     
     
     
     
     