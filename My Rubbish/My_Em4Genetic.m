% Algoritmo EM per HMM con osservazioni modellate da densità miste
% gaussiane. Ad ogni stato nascosto corrisponde una mistura di gaussiane,
% con un numero di esse che non è uguale per ogni stato, ma varia a seconda
% dei criteri di inizializzazione.
%
%    [new_trans_prob,new_start_prob,Param,gmm_obj,loglike,track,track2]=...
%    GmmHmmEm2(data,Clustering,Starting,normalize_int,max_loop,max_loop,Hidden,info,reg,varargin)
% Input:
%   data : vettore NxD, dove N è il numero di osservazioni e D la
%          dimensione. Se si pone data=[], viene avviato l'input di una
%          traiettoria.
%   Clustering : modalità di clustering per gli stati nascosti e la
%          suddivisione del dataset.
%          km = kmeans con scelta del numero di stati
%          kmvar = kmeans con numero di stati variazionale
%          var = inizializzazione con metodo variazionale
%          kmgap = kmeans con statistica gap che indica il num di stati
%   Starting : stringa che indica l'inizializzazione dei dati. 
%          km = kmeans con scelta del numero di stati
%          kmvar = kmeans con numero di stati variazionale
%          var = inizializzazione con metodo variazionale
%          kmgap = kmeans con statistica gap che indica il num di stati
%          Al posto di una stringa si può dare un numero che indica quante
%          gaussiane assegnare a ogni stato nascosto, che il programma
%          suddividerà con il kmeans.
%   normalize_int : intervallo di normalizzazione dei dati, che deve essere
%          del tipo [X1_min X2_min ... XD_min ;X1_max X2_max ... XD_max]. 
%          Se si pone [], allora i dati non vengono normalizzati.
%   max_loop : soglia di incremento della likelihood che ferma l'algoritmo
%   max_loop  : max numero di iterazioni dell'algoritmo
%   Hidden : matlab struct che contiene Hidden.trans_prob,Hidden.start_prob 
%            per l'inizializzazione. Inserendo [], si ha inizializzazione
%            casuale.
%   info : stringa per visualizzare le informazioni sulla likelihood,
%          scrivendo 'yes'. Scrivere qualsiasi altra stringa per non
%          visualizzare le informazioni.
%   reg : parametro di regolarizzazione per la diagonale delle covarianze.
%       OPZIONALI
%   idx : indici delle osservazioni clusterizzate, da dare solamente se 
%         l'input clustering è una struttura che contiene già la suddi-
%         visione del dataset. Ad esempio, idx può provenire dal dataset
%         nel seguente modo, con il kmeans
%
%                     Clustering = {};
%                     [idx,ctrs] = kmeans(track,4);
%                     for j = 1:4
%                         Clustering{j} = track(idx==j,:);
%                     end
%
% Output:
%   new_trans_prob : matrice di transizione tra gli stati nascosti
%   new_start_prob : probabilità iniziali degli stati nascosti
%   Param : struct con i parametri ''mu,sigma,mix'' della mistura
%   gmm_obj : obj gmdistribution che rappresenta l'oggetto mistura
%             gaussiana
%   track : dati normalizzati
%   track2 : dati suddivisi per i vari stati nascosti

 
function [new_trans_prob,new_start_prob,Param,gmm_obj,loglike,select]=...
       My_Em4Genetic(track,Hidden,Param,reg,info,max_loop,select,figura)
     
addpath(genpath('./My Rubbish'))  


[N,D] = size(track);
Q = sum(select);

%Inizializzazioni parametri
if isempty(Param)
        [Param.mix, Param.mu, Param.sigma] = My_GmmInitKM(track,Q);
        Param.sigma = Param.sigma + repmat(reg*eye(D),[1,1,Q]);
end

if isempty(Hidden)  
    new_trans_prob = My_normalizza(rand(Q,Q));
    new_start_prob = My_normalizza(rand(1,Q));
else
    new_trans_prob = Hidden.trans_prob;
    new_start_prob = Hidden.start_prob;
end

Nu = zeros(N,Q); %matrice delle responsibilities
fine = 0; %flag fine algoritmo 
loglike = [];
cont = 0;

%disp(sprintf('Sono stati creati %d stati',Q))

gmm_obj = gmdistribution(Param.mu',Param.sigma,Param.mix);

if ~isempty(figura)
    close(figura);
    scrsz = get(0,'ScreenSize');
    figura = figure('Position',[1 scrsz(4) scrsz(3) scrsz(4)]);
end


 while fine ~= 10 
    
    trans_prob=new_trans_prob;
    start_prob=new_start_prob;
    cont = cont + 1;
    
    %% E-step: calcolo di alpha e beta mediante B, che è determinata 
    %  a partire dal Gmm
   
    %Matrice probabilità di estrazione dei dati dalle misture
    B = zeros(N,Q);
    for j = 1:Q  
          B(:,j) = Param.mix(j)*mvnpdf(track,Param.mu(:,j)',Param.sigma(:,:,j));
    end
    %Per fare si' che la procedura di forward backward non abbia dei NaN
    %nelle costanti di normalizzazione e quindi anche negli alpha e beta,
    %pongo i NaN della matrice B uguali al valore più vicino a zero, cioè
    %eps. Se mettessi zero, potrei trovare delle righe che, una volta
    %sommate, mi restituiscono zero e creano dei NaN come costanti di
    %normalizzazione.
    B(isnan(B))=eps;
    
    %B(B==0)=eps;
    %procedura di forward riscalata
    [alpha,c,c_prod,P_OL,loglike(cont)] =...
        My_forward_gmm(trans_prob,start_prob,B,N,Q);
    %procedura di backward riscalata
    [beta,d_prod] =...
        My_backward_gmm(trans_prob,B,N,Q,c);
    
    %% M-step:
    %iterazione tipo Baum Welch per stimare transizioni tra stati nascosti
    %e probabilità iniziali
    [new_trans_prob,new_start_prob] =... 
        My_hidden_states_gmm(trans_prob,alpha,beta,Q,B,c,N);
    if any(isnan(new_trans_prob))
        new_trans_prob = rand(Q);
    end
    
    %normalizzazioni
    new_start_prob = My_normalizza(new_start_prob);
    new_trans_prob = My_normalizza(new_trans_prob);
    %stampa
    if strcmp(info,'yes')
        str = sprintf('Loglike: %.5f all''iterata %d',loglike(cont),cont);
        disp(str);    
    end
    %calcolo delle responsibilities nella struttura Nu
    Nu = zeros(N,Q);
    num = alpha.*beta;
    num(isnan(num))=eps;
    den = sum(num,2); 
    den(den==0)=1e-10;
    idx2 = [];
    %for k = 1:Q
        %f = @(x)pdf(gmm_obj,x);
        %den2 = f(track);
        %cerco quelle posizioni in cui den2 vale zero e non le considero in
        %quanto portano contributo bassissimo nel ristimare il modello di HMM,
        %ponendo in loro luogo uno zero come responsibility. 
        idx2={};
        for j=1:Q 
             idx2{j} = find(B(:,j) ~= 0);
             Nu(idx2{j},j)= (num(idx2{j},j)./den(idx2{j})); %.*...
             %(Param.mix(j)*mvnpdf(track,Param.mu(:,j)',Param.sigma(:,:,j))) ./ den2;
        end
        %Se eventualmente Nu ha dei valori NaN, li pongo a zero in quanto
        %tali responsibilities erano sostanzialmente irrilevanti.
        Nu(isnan(Nu)) = 1e-10;
    %end
    
    %ristima dei parametri del Gmm
%    for j = 1:Q
        sum_Nu = sum(Nu,1);
        sum_Nu(sum_Nu == 0) = 1e-10;
        for k=1:Q
            %coefficienti della mistura e
            %medie della mistura
            Param.mix(k) = sum_Nu(k) / sum(sum_Nu);
            Param.mu(:,k) = sum(repmat(Nu(idx2{k},k),1,D) .* track(idx2{k},:)) / sum_Nu(k);            
            %varianze della mistura
            Param.sigma(:,:,k) = zeros(D);
            for n = 1:N
                if any(idx2{k}(:)==n)
                Param.sigma(:,:,k) = Param.sigma(:,:,k)+(Nu(n,k)...
                *kron((track(n,:)-Param.mu(:,k)')',(track(n,:)-Param.mu(:,k)')));
                end
            end  
            Param.sigma(:,:,k) = Param.sigma(:,:,k)/sum_Nu(k);
        end 
    %end
    
    %rimozione di eventuali gaussiane con coefficiente di mistura
    %irrilevante
        mixidx = find(Param.mix < 1e-6);
        if ~isempty(mixidx)
            Param.mix(mixidx) = 1e-6;
            Param.mix = My_normalizza(Param.mix);          
        end
    
    
        %regolarizzazione covarianze con reg aggiunto sulla diagonale
        Param.sigma = Param.sigma + repmat(reg*eye(D),[1,1,Q]);
        gmm_obj = gmdistribution(Param.mu',Param.sigma,Param.mix);
    
if ~isempty(figura)    
    if D==2
        clf(figura);        
        subplot(3,4,[1 2 3 5 6 7 9 10 11])
        hold on
        plot(track(:,1),track(:,2),'Color',[0 .1 .9],'LineStyle','*','LineWidth',2)
        for i = 1:length(Param.mix)
                MyEllipse(Param.sigma(:,:,i), Param.mu(:,i),'style','r','intensity',Param.mix(i), 'facefill',.8);
                text(Param.mu(1,i), Param.mu(2,i), num2str(i),'BackgroundColor', [.7 .9 .7]);
        end
        hold off
        s1 = sprintf('Gmm (proiettato) all''iterata %d',cont);
        s2 = sprintf('\nLogLikelihood = %f',loglike(cont));
        s3 = sprintf('\n%d stati nascosti',Q);
        title(strcat(s1,s2,s3))
        subplot(3,4,4)
        hold on
        axis([0 Q+1 0 Q+1])
        imagesc(flipud(new_trans_prob))
        colormap winter
        hold off
        title('Matrice transizione')
        subplot(3,4,8)
        hold on
        bar(new_start_prob)
        colormap winter
        hold off
        title('Probabilita'' iniziali') 
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
                text(M(1), M(2), num2str(i),'BackgroundColor', [.7 .9 .7]);
        end
        hold off
        s1 = sprintf('Gmm (proiettato) all''iterata %d',cont);
        s2 = sprintf('\nLogLikelihood = %f',loglike(cont));
        s3 = sprintf('\n%d stati nascosti',Q);
        title(strcat(s1,s2,s3))
        subplot(3,4,4)
        hold on
        axis([0 Q+1 0 Q+1])
        imagesc(flipud(new_trans_prob))
        colormap winter
        hold off
        title('Matrice transizione')
        subplot(3,4,8)
        hold on
        bar(new_start_prob)
        colormap winter
        hold off
        title('Probabilita'' iniziali') 
        subplot(3,4,12)
        plot(loglike)       
        title('loglikelihood')        
        drawnow;
        pause(0.02);    
    end
end

    %controllo della convergenza o del max numero di cicli
    if cont > 1
       if abs(loglike(cont)-loglike(cont-1)) < 1e-5 ...
          && loglike(cont-1)<loglike(cont), fine=fine+1;
       elseif loglike(cont-1)>loglike(cont), fine = fine+1 ; 
       end
       if cont >= max_loop, fine=10; end
    end   
    

 end
 

     
     
     
     
     
     
     