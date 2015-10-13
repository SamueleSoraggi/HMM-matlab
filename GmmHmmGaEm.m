%Hybrid Genetic-EM algorithm for HMMs with observations described by a 
%single GMM.
% function [trans,prior,gmm,loglike,MDL_min,track]=...
%       GmmHmmGaEm(data,normalize_int,em_iter,mt,ct,pop,Q_max,cxover_mode)
% Input:
%   data : NxD vector, where N is the number of observations and D their
%          dimension. If you set data=[], a simulator window will be opened
%          and you can paint your own trajectory with the mouse.
%   normalize_int : normalization interval of the dataset, of the type
%          [X1_min X2_min ... XD_min ;X1_max X2_max ... XD_max]. 
%          If you set [], the data will remain unnormalized.
%   em_iter  : max number of iterations
%   mt : mutation rate. range: [1e-3 , 1e-2]
%   ct : crossover rate. range: [.1 , 1]
%   pop : population. suggested range: [6 , 20]
%   Q_max : max number of states. suggested range: [8 , 20]
%   cxover_mode: a string that decides the cross-over operator.
%                At the moment, only 'single point' is implemented.
% Output:
%   trans : transition matrix for the hidden states
%   prior : starting probability for the hidden states
%   gmm : gmm matlab object
%   loglike : loglikelihood vector
%   MDL_min : lowest value of the objective function
%   track : normalized/unnormalized dataset

function [trans,prior,gmm,loglike,MDL_min,track]=...
         GmmHmmGaEm(data,normalize_int,em_iter,mt,ct,pop,Q_max,cxover_mode)
     
addpath(genpath('./My Rubbish'))  

% Print and organize the dataset
if ~isempty(data) && ~isempty(normalize_int)
    track=My_normalize_data(data,1,normalize_int);
end
if ~isempty(data) && isempty(normalize_int)
    track=data;
end
if isempty(data) && ~isempty(normalize_int)
        [full_data,~] = grabDataFromCursorDynamics;
    data = full_data;
    track=My_normalize_data(data',1,normalize_int); 
end
if isempty(data) && isempty(normalize_int)
    [full_data,~] = grabDataFromCursorDynamics;
    data = full_data;
    normalize_int = [ min(data);max(data)];
    track=My_normalize_data(data',1,normalize_int);
end

    [N D] = size(track);  
%% Genetic Parameters
    t = 0;          % counter of the iterations
    MDL_Size = 0;   % minimum MDL
    c_end = 0;      % end flag
    alpha = .5;     % convex factor for the single-point crossover
    L = D+D*(D+1)/2;% useful factor for the MDL formulae 
    reg = 1e-5;     %regularization factor
    
%% population 1

for j = 1:10*pop
    % randomly selected elements of the mixture
    select0{j} = randi(2,1,Q_max) - 1;   
        
    % Kmeans initialization
    [~, Buffer.mu, Buffer.sigma] = My_GmmInitKM(track,Q_max);
    Buffer.mix = My_normalizza(ones(1,sum(select0{j})));
    Buffer.sigma = Buffer.sigma + repmat(reg*eye(D),[1,1,Q_max]);

    
    Param0{j}.mu = Buffer.mu;
    Param0{j}.sigma = Buffer.sigma;
    
    Param0{j}.mix = zeros(1,Q_max); 
    Param0{j}.mix(1,select0{j}==1) = Buffer.mix;
    %Hidden chain
    Hidden0{j}.trans_prob = My_normalizza(rand(sum(select0{j})));
    Hidden0{j}.start_prob = My_normalizza(rand(1,sum(select0{j})));
    
    B = zeros(N,sum(select0{j}));
    colonna = 1;
    for jj = 1:Q_max
        if select0{j}(jj)==1
            B(:,colonna) = Param0{j}.mix(jj)*...
            mvnpdf(track,Param0{j}.mu(:,jj)',Param0{j}.sigma(:,:,jj));
            colonna = colonna+1;
        end
    end
    
    [~,~,~,~,ll] =...
    My_forward_gmm(Hidden0{j}.trans_prob,Hidden0{j}.start_prob,B,N,sum(select0{j}));    
    MDL0(j) = - ll + log(N)*sum(select0{j})*(L+1)/2;
end

MDL0(isnan(MDL0)) = +inf;
[MDL0,perm]  = sort(MDL0,'ascend');
MDL         = MDL0(1:pop);

for j = 1:pop 
        Param{j}  = Param0{perm(j)}; 
        select{j} = select0{perm(j)};
        Hidden{j} = Hidden0{perm(j)};
end

clear Param0 select0 Hidden0;


%% Regularization

for k = 1:pop
    Param{k}.sigma = Param{k}.sigma + repmat(reg*eye(D),[1,1,Q_max]);
end

figura=figure('Position', get(0,'ScreenSize'));
drawnow;

%% Main Program
while c_end ~= 5
    
    MDL_1 = zeros(1,pop);
    %Em on the POPULATION 1
    for j = 1:pop
        %buffering
        Buffer1 = {};
        Buffer1.mu = Param{j}.mu(:,select{j}==1);
        Buffer1.sigma = Param{j}.sigma(:,:,select{j}==1);
        Buffer1.mix = My_normalizza(rand(1,sum(select{j})));
        %EM
        [Hidden{j}.trans_prob,Hidden{j}.start_prob,Buffer1,gmm_obj,loglike,select{j}]=...
        My_Em4Genetic(track,Hidden{j},Buffer1,reg,'noinfo',em_iter,select{j},[]);
        %MDL criteria
        MDL_1(j) = - loglike(end) + log(N)*sum(select{j})*(L+1)/2; 
        %debuffering
        Param{j}.mu(:,select{j}==1) = Buffer1.mu;
        Param{j}.sigma(:,:,select{j}==1) = Buffer1.sigma;
        Param{j}.mix(1,select{j}==1) = Buffer1.mix;
    end
        
    % single point cross-over 
    cross = ceil(pop*ct/2); %number of crossovers

    if strcmp(cxover_mode,'single point')
        scelta = randi(pop,1,2*cross);
        point = randi(Q_max-1,1,cross);
        Param2 = {}; % population 2
        select2 = {};% selected elements from the second population
        for j = 1:cross
            %cxover of the element selectors
            select2{2*j-1} = zeros(1,Q_max);
            select2{2*j} = zeros(1,Q_max);
            select2{2*j-1}(1:point(j)) = select{2*j-1}(1:point(j));
            select2{2*j-1}(point(j)+1:end) = select{2*j}(point(j)+1:end);
            select2{2*j}(1:point(j)) = select{2*j}(1:point(j));
            select2{2*j}(point(j)+1:end) = select{2*j-1}(point(j)+1:end); 
            %cxover of the means
            Param2{2*j-1}.mu = zeros(D,Q_max);
            Param2{2*j}.mu = zeros(D,Q_max);
            Param2{2*j-1}.mu(:,1:point(j)) = Param{2*j-1}.mu(:,1:point(j));
            Param2{2*j-1}.mu(:,point(j)+1:end) =...
                                           Param{2*j}.mu(:,point(j)+1:end);
            Param2{2*j}.mu(:,1:point(j)) = Param{2*j}.mu(:,1:point(j));
            Param2{2*j}.mu(:,point(j)+1:end) =...
                                         Param{2*j-1}.mu(:,point(j)+1:end);
            %cxover of the variances
            Param2{2*j-1}.sigma = zeros(D,D,Q_max);
            Param2{2*j}.sigma = zeros(D,D,Q_max);               
            Param2{2*j-1}.sigma(:,:,1:point(j)) =...
                                        Param{2*j-1}.sigma(:,:,1:point(j));
            Param2{2*j-1}.sigma(:,:,point(j)+1:end) =...
                                      Param{2*j}.sigma(:,:,point(j)+1:end);                                   
            Param2{2*j}.sigma(:,:,1:point(j)) =...
                                         Param{2*j}.sigma(:,:,1:point(j));
            Param2{2*j}.sigma(:,:,point(j)+1:end) =...
                                   Param{2*j-1}.sigma(:,:,point(j)+1:end);                                   
            %cxover of the mixture weights
            Param2{2*j-1}.mix = Param{2*j-1}.mix;
            Param2{2*j}.mix = Param{2*j}.mix;
            %randomly initialized hidden parameters for the new elements
            Hidden2{2*j-1}.start_prob = My_normalizza(rand(1,sum(select2{2*j-1})));
            Hidden2{2*j-1}.trans_prob = My_normalizza(rand(sum(select2{2*j-1})));
            Hidden2{2*j}.start_prob = My_normalizza(rand(1,sum(select2{2*j})));
            Hidden2{2*j}.trans_prob = My_normalizza(rand(sum(select2{2*j})));
        end
    end    
        
    %EM on the population 2
    MDL_2 = zeros(1,2*cross); %MDL population 2
    for j = 1:2*cross
        %buffering
        Buffer2 = {};
        Buffer2.mu = Param{j}.mu(:,select2{j}==1);
        Buffer2.sigma = Param{j}.sigma(:,:,select2{j}==1);
        Buffer2.mix = My_normalizza(rand(1,sum(select2{j})));
        %EM
        [Hidden2{j}.trans_prob,Hidden2{j}.start_prob,Buffer2,gmm_obj,loglike,select2{j}]=...
        My_Em4Genetic(track,Hidden2{j},Buffer2,reg,'noinfo',em_iter,select2{j},[]);
        %MDL criteria
        MDL_2(j) = - loglike(end) + log(N)*sum(select2{j})*(L+1)/2; 
        %debuffering
        Param2{j}.mu(:,select2{j}==1) = Buffer2.mu;
        Param2{j}.sigma(:,:,select2{j}==1) = Buffer2.sigma;
        Param2{j}.mix(1,select2{j}==1) = Buffer2.mix;
    end
    
    % selecting the best individuals (those with the lowest MDL),
    %creating population 3
    MDL_3 = [MDL_1,MDL_2];
    MDL_3(isnan(MDL_3)) = +inf;
    [MDL_3,perm] = sort(MDL_3,'ascend');
    Param3 = {};
    select3 = {};
    Hidden3 = {};
    for j = 1:pop
        if perm(j)<=pop 
            Param3{j} = Param{perm(j)}; 
            select3{j} = select{perm(j)};
            Hidden3{j} = Hidden{perm(j)};
        else
            indice = perm(j)-pop;
            Param3{j} = Param2{indice}; 
            select3{j} = select2{indice};
            Hidden3{j} = Hidden2{indice};
        end
    end
    MDL_3 = MDL_3(1:pop);
    
    %substituting the worst elements of population 3 with new random
    %elements, with a subsequent application of the EM
    for j = 4:pop
        select3{j} = randi(2,1,Q_max) - 1;   

        % Kmeans
        [~, Buffer.mu, Buffer.sigma] =...
            My_GmmInitKM(track, Q_max);
        Buffer.mix = My_normalizza(ones(1,sum(select3{j})));
        Buffer.sigma = Buffer.sigma + repmat(reg*eye(D),[1,1,Q_max]);


        Param3{j}.mu = Buffer.mu;
        Param3{j}.sigma = Buffer.sigma;
        Param3{j}.mix = zeros(1,Q_max);
        Param3{j}.mix(1,select3{j}==1) = Buffer.mix;

        Hidden3{j}.trans_prob = My_normalizza(rand(sum(select3{j})));
        Hidden3{j}.start_prob = My_normalizza(rand(1,sum(select3{j}))); 

        Buffer.mu = Param3{j}.mu(:,select3{j}==1);
        Buffer.sigma = Param3{j}.sigma(:,:,select3{j}==1);
        Buffer.mix = My_normalizza(ones(1,sum(select3{j})));
        
        [Hidden3{j}.trans_prob,Hidden3{j}.start_prob,Buffer,gmm_obj,loglike,select3{j}]=...
        My_Em4Genetic(track,Hidden3{j},Buffer,reg,'noinfo',em_iter*t,select3{j},[]);
        %MDL criteria
        MDL_3(j) = - loglike(end) + log(N)*sum(select3{j})*(L+1)/2; 
        %writing parameters
        Param3{j}.mu(:,select3{j}==1) = Buffer.mu;
        Param3{j}.sigma(:,:,select3{j}==1) = Buffer.sigma;
        Param3{j}.mix(1,select3{j}==1) = Buffer.mix;
        
    end
    
   % Mutation operator 
   Param = {}; 
   select = {};
        
   Param = Param3;  % New generation (population 4) parameters
   select = select3;% selection for population 4
   Hidden = Hidden3;% struct for hidden chain
   MDL = zeros(1,pop);
   MDL(1) = MDL_3(1);
   
   for i = 2:pop
      random = rand(1,Q_max); %randomly extracted mutation probability
      for j = 1:Q_max
          if random(j) <= mt  % mutation
             if select{i}(1,j) == 0 && j<Q_max && j>1
                 new_start = [Hidden{i}.start_prob(1:sum(select{i}(1:j))),...
                     rand(1),Hidden{i}.start_prob(sum(select{i}(1:j))+1:end)];
                 new_start = My_normalizza(new_start);
                 new_trans = [Hidden{i}.trans_prob(1:sum(select{i}(1:j)),:);...
                              rand(1,sum(select{i}));...
                              Hidden{i}.trans_prob(sum(select{i}(1:j))+1:end,:)];
                 new_trans = [new_trans(:,1:sum(select{i}(1:j))),...
                              rand(sum(select{i})+1,1),...
                              new_trans(:,sum(select{i}(1:j))+1:end)];
                 new_trans = My_normalizza(new_trans);
                 
                 Hidden{i}.start_prob = new_start;
                 Hidden{i}.trans_prob = new_trans;
                 select{i}(j) = 1;
             elseif select{i}(1,j) == 0 && j==Q_max
                 new_start = [Hidden{i}.start_prob, rand(1)];
                 new_start = My_normalizza(new_start);
                 new_trans = [Hidden{i}.trans_prob;...
                              rand(1,sum(select{i}))];
                 new_trans = [new_trans,...
                              rand(sum(select{i})+1,1)];         
                 new_trans = My_normalizza(new_trans);

                 Hidden{i}.start_prob = new_start;
                 Hidden{i}.trans_prob = new_trans;
                 select{i}(j) = 1;
             elseif select{i}(1,j) == 0 && j==1
                 new_start = [rand(1), Hidden{i}.start_prob];
                 new_start = My_normalizza(new_start);
                 new_trans = [rand(1,sum(select{i}));
                              Hidden{i}.trans_prob];
                 new_trans = [rand(sum(select{i})+1,1),...
                              new_trans];
                 new_trans = My_normalizza(new_trans);

                 Hidden{i}.start_prob = new_start;
                 Hidden{i}.trans_prob = new_trans;
                 select{i}(j) = 1;
             elseif select{i}(1,j) == 1 && sum(select{i})>1
                 Hidden{i}.start_prob(sum(select{i}(1:j))) = [];
                 Hidden{i}.trans_prob(sum(select{i}(1:j)),:) = [];
                 Hidden{i}.trans_prob(:,sum(select{i}(1:j))) = [];
                 Hidden{i}.start_prob =...
                     My_normalizza(Hidden{i}.start_prob);
                 Hidden{i}.trans_prob =...
                     My_normalizza(Hidden{i}.trans_prob);
                 select{i}(1,j) = 0;
             end   
          end        
      end
   end
    
    % finding the minimum MDL and its index
    [MDL_min(t+1),idx(t+1)] = min(MDL)
    components(t+1) = sum(select{idx(t+1)}==1);
    
    %painting :-)
    if D==2
        clf(figura); 
        subplot(3,4,[1 2 3 5 6 7 9 10 11]);
        hold on
        plot(track(:,1),track(:,2),'Color',[0 .1 .9],'LineStyle','*','LineWidth',2)
        for i = find(select{idx(t+1)}==1)
                MyEllipse(Param{idx(t+1)}.sigma(:,:,i), Param{idx(t+1)}.mu(:,i),'style','r','intensity',Param{idx(t+1)}.mix(i), 'facefill',.8);
                text(Param{idx(t+1)}.mu(1,i), Param{idx(t+1)}.mu(2,i), num2str(i),'BackgroundColor', [.7 .9 .7],'FontSize',16);
        end
        hold off
        s1 = sprintf('Gmm with minimum MDL - step %d',t+1);
        s2 = sprintf('\n MDL = %f',MDL_min(t+1));
        s3 = sprintf('\n%d Hidden states',components(t+1));
        title(strcat(s1,s2,s3))
        xlabel('x position')
        ylabel('y position')        
        subplot(3,4,4)
        hold on
        axis([0 components(t+1)+1 0 components(t+1)+1])
        imagesc(flipud(Hidden{idx(t+1)}.trans_prob))
        colormap winter
        hold off
        title('Transition matrix')
        subplot(3,4,8)
        hold on
        bar(Hidden{idx(t+1)}.start_prob)
        hold off
        title('Starting probabilities') 
        subplot(3,4,12)
        plot(MDL_min)       
        title('Minimum MDL')        
        drawnow;
        pause(0.02);
    else
        clf(figura); 
        subplot(3,4,[1 2 3 5 6 7 9 10 11]);
        hold on
        plot(track(:,2),track(:,3),'Color',[0 .1 .9],'LineStyle','*','LineWidth',2)
        for i = find(select{idx(t+1)}==1)
            S = Param{idx(t+1)}.sigma(:,:,i); S = S([2 3],[2 3]);
            M = Param{idx(t+1)}.mu(:,i); M = M([2,3]);
            MyEllipse(S, M,'style','r','intensity',Param{idx(t+1)}.mix(i), 'facefill',.8);
            text(M(1), M(2), num2str(i),'BackgroundColor', [.7 .9 .7],'FontSize',16);
        end
        hold off
        s1 = sprintf('Gmm (projected) with lowest MDL - step %d',t+1);
        s2 = sprintf('\n MDL = %f',MDL_min(t+1));
        s3 = sprintf('\n%d Hidden states',components(t+1));
        title(strcat(s1,s2,s3))
        xlabel('x position')
        ylabel('y position')        
        subplot(3,4,4)
        hold on
        axis([0 components(t+1)+1 0 components(t+1)+1])
        imagesc(flipud(Hidden{idx(t+1)}.trans_prob))
        colormap winter
        hold off
        title('Transition matrix')
        subplot(3,4,8)
        hold on
        bar(Hidden{idx(t+1)}.start_prob)
        hold off
        title('Starting probabilities') 
        subplot(3,4,12)
        plot(MDL_min)       
        title('Minimum MDL')        
        drawnow;
        pause(0.02);
    end
        
    % end flag
    if components(t+1) ~= MDL_Size
        c_end = 0;
        MDL_Size = components(t+1);
    else
        c_end = c_end+1;
    end
   format short
   disp(sprintf('Genetic Step %d; MDL min = %f; c_end = %d.',t+1,MDL_min(t+1),c_end));    
   t=t+1;
   
end

disp(sprintf('Applying the EM to the best individual !!! '))

   Parameters.mix = Param{idx(t)}.mix(select{idx(t)}==1);
   Parameters.mu = Param{idx(t)}.mu(:,(select{idx(t)}==1));
   Parameters.sigma = Param{idx(t)}.sigma(:,:,(select{idx(t)}==1));
    
   [trans,prior,~,gmm,loglike,~]=...
   My_Em4Genetic(track,Hidden{idx(t)},Parameters,reg,'yes',1000,ones(1,sum(select{idx(t)}==1)),figura);
     
     
     
     
     
     
     