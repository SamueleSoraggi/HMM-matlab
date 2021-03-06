function [obs,states] = My_gmm_hmm_sim2(trans_prob,start_prob,gmm_obj,N)
    % errori di input
    %if nargin ~= 4, error('myApp:argChk', 'Vanno messi 4 input');end
    % inizializzazione
    obs = zeros(N,size(gmm_obj{1}.mu,2));  %output1
    states = zeros(1,N);        %output2    
    Q = size(start_prob,1);             %numero di stati
    salti = rand(1,N);                  %salti tra gli hidden states
%scelta stato iniziale
    somma = cumsum(start_prob);
    indici = find(somma >= salti(1));
    stato = indici(1);
    states(1) = stato;
%prima osservazione    
    obs(1,:) = random(gmm_obj{stato},1);
%generazione delle osservazioni
    for i=2:N
        %salto
        somma  = zeros(1,Q);
        somma  = cumsum(trans_prob(stato,:));
        indici = find(somma >= salti(i));
        stato  = indici(1);
        states(i) = stato;
        %i-esima osservazione.      
        obs(i,:) = random(gmm_obj{stato},1);
    end
end