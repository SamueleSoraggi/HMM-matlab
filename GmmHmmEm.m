%EM algorithm for HMMs with observation described by a single GMM
%        [new_trans_prob,new_start_prob,Param,gmm_obj,loglike]=...
%  GmmHmmEm(data,Starting,normalize_int,soglia_em,max_loop,Hidden,info,reg)
% Input:
%   data : NxD vector, where N is the number of observations and D their
%          dimension. If you set data=[], a simulator window will be opened
%          and you can paint your own trajectory with the mouse.
%   Starting : a string that set the initialization of the model. 
%          km = kmeans
%          kmvar = kmeans with the number of states set by the variational
%                  EM
%          var = initialization with the variational EM
%          kmgap = kmeans with the gap statistic (very slow and abandoned)
%          Instead of inserting a string, you can give an initialization
%          by defining a struct called Param which contains:
%          Param.mu - DxQ matrix of the means
%          Param.sigma - DxDxQ matrix of the covariances
%          Param.mix - 1xQ vector of the mixture coefficients
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
% Output:
%   new_trans_prob : transition matrix for the hidden states
%   new_start_prob : starting probability for the hidden states
%   Param : structure with the mixture's parameters ''mu,sigma,mix''
%   gmm_obj : obj gmdistribution which represents the matlab object for a
%           mixture of gaussians
%   track : normalized/unnormalized data



function [new_trans_prob,new_start_prob,Param,gmm_obj,loglike,track]=...
       GmmHmmEm(data,Starting,normalize_int,soglia_em,max_loop,Hidden,info,reg)
     
format long;
addpath(genpath('./My Rubbish'))  


% Avvio del disegno della traiettoria e sistemazione dataset
if ~isempty(data) && ~isempty(normalize_int)
    track=My_normalize_data(data,1,normalize_int);
end
if ~isempty(data) && isempty(normalize_int)
    track=data;
end
if isempty(data) && isempty(normalize_int)
    [full_data,~] = grabDataFromCursorDynamics;
    track=full_data';  
end
if isempty(data) && ~isempty(normalize_int)
    [full_data,~] = grabDataFromCursorDynamics;
    track=My_normalize_data(full_data',1,normalize_int);
end


% altri parametri
[N D] = size(track);
%cov_type = 'full';  %covarianza piena
 
%% Inizializzazione
if isstr(Starting)
     if strcmp(Starting , 'var')
        % Variational initialization
        clear options;
        options.maxIter = 200;
        options.threshold = 1e-5;
        % 1 = shows 2D plot of each iteration, 0 = otherwise
        options.displayFig = 0; 
        % 1 = shows which iteratin is running, 0 = otherwise
        options.displayIter = 0; 

        PriorPar.alpha = 0.01; %0.001
    if ~isempty(normalize_int)
        PriorPar.mu = normalize_int(1,:)';
    else
        PriorPar.mu = min(track)';
    end
    PriorPar.beta = 1;
    PriorPar.v = 3*D;
    PriorPar.W = inv(PriorPar.v * cov(track) + reg*eye(D)); %200
    
    
        [~, ~, out_gmm_obj, ~] =...
            fn_VBGMM_clustering(track, 10, PriorPar, options);

        Q = size(out_gmm_obj.PComponents,2);

        Param.sigma = out_gmm_obj.Sigma;
        Param.mu = out_gmm_obj.mu';
        Param.mix = out_gmm_obj.PComponents;

     elseif strcmp(Starting , 'km')
        Q=input('Number of hidden states (Q >= 2): ');

        % Inizializzazione 'tradizionale' con k means
        [Param.mix, Param.mu, Param.sigma] = My_GmmInitKM(track,Q);
        Param.sigma = Param.sigma + repmat(reg*eye(D),[1,1,Q]);

     elseif strcmp(Starting , 'kmgap')% % % % % !!! ABANDONED !!!
        num_clusters = 3 : 15;  
        num_reference_bootstraps = 2000; 
        iter_test = 20;
        opt_index = zeros(iter_test,1);
        max_gap = zeros(iter_test,1);

        % 1 for using compactness as the dispersion measure, 
        % instead of the distance from the mean 
        compactness_as_dispersion = 0; 

        for ii = 1 : iter_test
             [ opt_index(ii), max_gap(ii)] = ...
             gap_statistics(track, num_clusters,...
             num_reference_bootstraps, compactness_as_dispersion);
        end
        Q = round(mean(opt_index));

        [Param.mix, Param.mu, Param.sigma] = My_GmmInitKM(track,Q);
        Param.sigma = Param.sigma + repmat(reg*eye(D),[1,1,Q]);
        
        str = sprintf('Il gap ha selezionato %d stati.',Q);
        disp(str);
     elseif strcmp(Starting , 'kmvar')  
        % Kmeans + Variational initialization
        clear options;
        options.maxIter = 200;
        options.threshold = 1e-5;
        % 1 = shows 2D plot of each iteration, 0 = otherwise
        options.displayFig = 0; 
        % 1 = shows which iteratin is running, 0 = otherwise
        options.displayIter = 0; 

        PriorPar.alpha = 0.01; %0.001
        if ~isempty(normalize_int)
            PriorPar.mu = normalize_int(1,:)';
        else
            PriorPar.mu = min(track)';
        end
    PriorPar.beta = 1;
    PriorPar.v = 3*D;
    PriorPar.W = inv(PriorPar.v * cov(track) + reg*eye(D)); %200

        [~, ~, out_gmm_obj, ~] =...
            fn_VBGMM_clustering(track, 10, PriorPar, options);

        Q = size(out_gmm_obj.PComponents,2);

        [Param.mix, Param.mu, Param.sigma] = My_GmmInitKM(track,Q);
        Param.sigma = Param.sigma + repmat(reg*eye(D),[1,1,Q]);
     else
        error('Unexisting option for the initialization.')
     end
else
 % Initialization with given parameters
 Param = Starting;
 Q = size(Param.mix,2);
end

%Inizializzazioni parametri
    
    % Riordino etichette
    [Param.mu,sortidx] = sortrows(Param.mu');
    Param.mu = Param.mu';
    Param.sigma = Param.sigma(:,:,sortidx);
    Param.mix = Param.mix(sortidx);
    
    if isempty(Hidden)  
        new_trans_prob = My_normalizza(rand(Q,Q));
        new_start_prob = My_normalizza(rand(1,Q));
    else
        new_trans_prob = Hidden.trans_prob;
        new_start_prob = Hidden.start_prob;
    end
    
    
    Nu = zeros(N,Q); %matrix containing the responsibilities
    fine = 0; %flag for the end of the algorithm 
    loglike = [];
    cont = 0;
    gmm_obj = gmdistribution(Param.mu',Param.sigma,Param.mix);
    %pause%%%%%%%%%%%
    figura=figure('Position', get(0,'ScreenSize'));
    
    if D==2
        clf(figura);        
        subplot(3,4,[1 2 3 5 6 7 9 10 11])
        hold on
        plot(track(:,1),track(:,2),'Color',[0 .1 .9],'LineStyle','*','LineWidth',2)
        for i = 1:length(Param.mix)
                MyEllipse(Param.sigma(:,:,i), Param.mu(:,i),'style','r','intensity',Param.mix(i), 'facefill',.8);
                text(Param.mu(1,i), Param.mu(2,i), num2str(i),'BackgroundColor', [.7 .9 .7],'FontSize',16);
        end
        xlabel('x position')
        ylabel('y position')
        hold off
        s1 = sprintf('Gmm (projected) at the step %d',cont);
        s2 = sprintf('\nLogLikelihood = --------');
        s3 = sprintf('\n%d Hidden States',Q);
        title(strcat(s1,s2,s3))
        subplot(3,4,4)
        hold on
        axis([0 Q+1 0 Q+1])
        %imagesc(new_trans_prob)
        colormap winter
        hold off
        title('Transition matrix')
        subplot(3,4,8)
        hold on
        %bar(new_start_prob)
        colormap winter
        hold off
        title('Starting probabilities') 
        subplot(3,4,12)
        %plot(loglike)       
        title('loglikelihood')        
        drawnow;
        pause(0.02);
    else
        clf(figura); 
        subplot(3,4,[1 2 3 5 6 7 9 10 11])
        hold on
        plot(track(:,2),track(:,3),'Color',[0 .1 .9],'LineStyle','*','LineWidth',2)
        for i = 1:length(Param.mix)
                S = Param.sigma(:,:,i); S = S([2 3],[2 3]);
                M = Param.mu(:,i); M = M([2,3]);
                MyEllipse(S, M,'style','r','intensity',Param.mix(i), 'facefill',.8);
                text(M(1), M(2), num2str(i),'BackgroundColor', [.7 .9 .7],'FontSize',16);
        end
        xlabel('x position')
        ylabel('y position')
        hold off
        s1 = sprintf('Gmm (projected) at the step %d',cont);
        s2 = sprintf('\nLogLikelihood = --------');
        s3 = sprintf('\n%d Hidden States',Q);
        title(strcat(s1,s2,s3))
        subplot(3,4,4)
        hold on
        axis([0 Q+1 0 Q+1])
        %imagesc(new_trans_prob)
        colormap winter
        hold off
        title('Transition matrix')
        subplot(3,4,8)
        hold on
        %bar(new_start_prob)
        colormap winter
        hold off
        title('Starting probabilities') 
        subplot(3,4,12)
        %plot(loglike)       
        title('loglikelihood')        
        drawnow;
        pause(0.02);    
    end
    
      
 while fine ~= 10 
    
    trans_prob=new_trans_prob;
    start_prob=new_start_prob;
    cont = cont + 1;
    
    %% E-step: forward-backward procedure
   
    %"Probability" matrix of the observations
    B = zeros(N,Q);
    for j = 1:Q  
          B(:,j) = Param.mix(j)*mvnpdf(track,Param.mu(:,j)',Param.sigma(:,:,j));
    end
    %we want to avoid the NaN values in the calculus. The Nan are given by 
    %small values of the normal densities, so we replace them with eps, the
    %smallest matlab value
    B(isnan(B))=eps;
    
    
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
    if any(isnan(new_trans_prob))
        new_trans_prob = rand(Q);
    end
    
    %normalizations
    new_start_prob = My_normalizza(new_start_prob);
    new_trans_prob = My_normalizza(new_trans_prob);
    %printing
    if strcmp(info,'yes')
        str = sprintf('Loglike: %.5f at the step %d',loglike(cont),cont);
        disp(str);    
    end
    %we calculate the responsibilities, saving them in Nu
    Nu = zeros(N,Q);
    num = alpha.*beta;
    num(isnan(num))=eps;
    den = sum(num,2); 
    den(den==0)=1e-10; 
    idx2={};
    for j=1:Q 
         idx2{j} = find(B(:,j) ~= 0);
         Nu(idx2{j},j)= (num(idx2{j},j)./den(idx2{j}));
    end
    Nu(isnan(Nu)) = 1e-10;
    
    
    %EM for the GMM
    sum_Nu = sum(Nu,1);
    sum_Nu(sum_Nu == 0) = 1e-10;
    for k=1:Q
        %means and weights of the mixtures
        Param.mix(k) = sum_Nu(k) / sum(sum_Nu);
        Param.mu(:,k) = sum(repmat(Nu(idx2{k},k),1,D) .* track(idx2{k},:)) / sum_Nu(k);            
        %variances
        Param.sigma(:,:,k) = zeros(D);
        for n = 1:N
            if any(idx2{k}(:)==n)
            Param.sigma(:,:,k) = Param.sigma(:,:,k)+(Nu(n,k)...
            *kron((track(n,:)-Param.mu(:,k)')',(track(n,:)-Param.mu(:,k)')));
            end
        end  
        Param.sigma(:,:,k) = Param.sigma(:,:,k)/sum_Nu(k);
    end 
    
    %we remove the unrelevant mixture coefficients
    mixidx = find(Param.mix < 1e-6);
    if ~isempty(mixidx)
        Param.mix(mixidx) = [];
        Param.mix = My_normalizza(Param.mix);
        Param.sigma(:,:,mixidx) = [];
        Param.mu(:,mixidx) = [];
        Q = length(Param.mix);
    end
   
    %we merge all the gaussian with same means and variance matrixes
    for j = 1:Q
        cont2=1;
        Buffer = cont2;
        for k = 1:length(Param.mix)-1
            if sum(Param.mu(:,k)==Param.mu(:,k+1))==D
               Buffer(1,k+1) = cont2;
            else
               cont2=cont2+1;
               Buffer(1,k+1) = cont2;
            end
        end
        for l = max(Buffer):-1:1
            idxbuf = find(Buffer==l);
            if length(idxbuf)>1
                Param.mu(:,idxbuf(2:end))=[];
                Param.mix(idxbuf(2:end))=[];
                Param.sigma(:,:,idxbuf(2:end))=[];
                Param.mix(idxbuf(1))=...
                Param.mix(idxbuf(1))*length(idxbuf);
                Q = Q - length(idxbuf) + 1;
            end
        end
    end
    
    
    %regularization of the variances
    Param.sigma = Param.sigma + repmat(reg*eye(D),[1,1,Q]);
    gmm_obj = gmdistribution(Param.mu',Param.sigma,Param.mix);
        
    %printing    
    if D==2
        clf(figura);        
        subplot(3,4,[1 2 3 5 6 7 9 10 11])
        hold on
        plot(track(:,1),track(:,2),'Color',[0 .1 .9],'LineStyle','*','LineWidth',2)
        for i = 1:length(Param.mix)
                MyEllipse(Param.sigma(:,:,i), Param.mu(:,i),'style','r','intensity',Param.mix(i), 'facefill',.8);
                text(Param.mu(1,i), Param.mu(2,i), num2str(i),'BackgroundColor', [.7 .9 .7],'FontSize',16);
        end
        hold off
        s1 = sprintf('Gmm (projected) at the step %d',cont);
        s2 = sprintf('\nLogLikelihood = %f',loglike(cont));
        s3 = sprintf('\n%d Hidden states',Q);
        title(strcat(s1,s2,s3))
        xlabel('x position')
        ylabel('y position')
        subplot(3,4,4)
        hold on
        axis([0 Q+1 0 Q+1])
        imagesc(flipud(new_trans_prob))
        colormap winter
        hold off
        title('Transition matrix')
        subplot(3,4,8)
        hold on
        bar(new_start_prob)
        colormap winter
        hold off
        title('Starting probabilities') 
        subplot(3,4,12)
        plot(loglike)       
        title('loglikelihood')        
        drawnow;
        pause(0.02);
    else
        clf(figura); 
        subplot(3,4,[1 2 3 5 6 7 9 10 11])
        hold on
        plot(track(:,2),track(:,3),'Color',[0 .1 .9],'LineStyle','*','LineWidth',2)
        for i = 1:length(Param.mix)
                S = Param.sigma(:,:,i); S = S([2 3],[2 3]);
                M = Param.mu(:,i); M = M([2,3]);
                MyEllipse(S, M,'style','r','intensity',Param.mix(i), 'facefill',.8);
                text(M(1), M(2), num2str(i),'BackgroundColor', [.7 .9 .7],'FontSize',16);
        end
        hold off
        s1 = sprintf('Gmm (projected) at the step %d',cont);
        s2 = sprintf('\nLogLikelihood = %f',loglike(cont));
        s3 = sprintf('\n%d Hidden states',Q);
        title(strcat(s1,s2,s3))
        xlabel('x position')
        ylabel('y position')
        subplot(3,4,4)
        hold on
        axis([0 Q+1 0 Q+1])
        imagesc(flipud(new_trans_prob))
        colormap winter
        hold off
        title('Transition matrix')
        subplot(3,4,8)
        hold on
        bar(new_start_prob)
        colormap winter
        hold off
        title('Starting probabilities') 
        subplot(3,4,12)
        plot(loglike)       
        title('loglikelihood')        
        drawnow;
        pause(0.02);    
    end

    %convergence control
    if cont > 1
       if abs(loglike(cont)-loglike(cont-1)) < 1e-5 ...
          && loglike(cont-1)<loglike(cont), fine=fine+1;
       elseif loglike(cont-1)>loglike(cont), fine = fine+1 ; 
       end
       if cont >= max_loop, fine=10; end
    end   
    

 end


     
     
     
     
     
     
     