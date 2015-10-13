%Procedura di backward riscalata, con matrice B adattata agli Hmm con Gmm.
function [beta,d_prod] = My_backward_gmm(trans_prob,B,N,Q,c)
    beta    = zeros(N,Q);
    d_prod = cumprod(rot90(c,2));
    d_prod = rot90(d_prod,2);    
    % inizializzazione
    beta(N,:) = c(N) * ones(1,Q);
    
    % induzione
    for t=N-1 : -1 :1
        beta(t,:) = sum((trans_prob.*repmat(B(t+1,:),Q,1)) .*...
                    repmat(beta(t+1,:),Q,1),2)';
        beta(t,:) = c(t) * beta(t,:);
    end    
    beta(isnan(beta))=eps;
end