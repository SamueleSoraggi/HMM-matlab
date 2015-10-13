%Procedura di forward riscalata, con matrice B adattata agli Hmm con Gmm.
function [alpha,c,c_prod,P_OL,loglike] =...
         My_forward_gmm2(trans_prob,start_prob,B,Q)
         
    dimB    = length(B);
    alpha   = {};
    c       = {}; 
    c_prod  = {};
    % inizializzazione
    
    for j = 1:dimB
    
    alpha{j}(1,:)  = start_prob .* B{j}(1,:); 
    c{j}(1)        = 1/sum(alpha{j}(1,:));
    alpha{j}(1,:)  = c{j}(1)*alpha{j}(1,:);
    
        % induzione
        for t=2:size(B{j},1)         
           alpha{j}(t,:) = sum(repmat(alpha{j}(t-1,:)',1,Q).*trans_prob) .* B{j}(t,:);
           c{j}(t) = 1/sum(alpha{j}(t,:));
           alpha{j}(t,:) = c{j}(t)*alpha{j}(t,:);
        end
    
    alpha{j}(isnan(alpha{j}))=eps;
    c_prod{j} = cumprod(c{j});
    %Likelihood senza applicazione del logaritmo
    P_OL(j)   = 1/c_prod{j}(N); 
    %Log-likelihood
    logllk(j) = -sum(log(c));
    
    end
    loglike = sum(logllk);
    
end