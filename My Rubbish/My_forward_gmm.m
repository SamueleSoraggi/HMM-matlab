%Procedura di forward riscalata, con matrice B adattata agli Hmm con Gmm.
function [alpha,c,c_prod,P_OL,loglike] =...
         My_forward_gmm(trans_prob,start_prob,B,N,Q)
    
    alpha   = zeros(N,Q);
    c       = zeros(1,N);    
    % inizializzazione
    alpha(1,:)  = start_prob .* B(1,:); 
    c(1)        = 1/sum(alpha(1,:));
    alpha(1,:)  = c(1)*alpha(1,:);
    
    % induzione
    for t=2:N         
       alpha(t,:) = sum(repmat(alpha(t-1,:)',1,Q).*trans_prob) .* B(t,:);
       c(t) = 1/sum(alpha(t,:));
       alpha(t,:) = c(t)*alpha(t,:);
    end
    alpha(isnan(alpha))=eps;
    c_prod = cumprod(c);
    %Likelihood senza applicazione del logaritmo
    P_OL   = 1/c_prod(N); 
    %Log-likelihood
    loglike = -sum(log(c));
end