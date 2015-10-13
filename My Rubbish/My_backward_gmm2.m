%Procedura di backward riscalata, con matrice B adattata agli Hmm con Gmm.
function [beta,d_prod] = My_backward_gmm2(trans_prob,B,Q,c)
    
    dimB = length(B);
    beta    = {};
    
    for j = 1:dimB
    
    d_prod{j} = cumprod(rot90(c,2));
    d_prod{j} = rot90(d_prod,2);    
    % inizializzazione
    N=size(B{j},1);
    beta{j}(N,:) = c{j}(N) * ones(1,Q);
    
       % induzione
       for t=N-1 : -1 :1
         beta{j}(t,:) = sum((trans_prob.*repmat(B{j}(t+1,:),Q,1)) .*...
                    repmat(beta{j}(t+1,:),Q,1),2)';
         beta{j}(t,:) = c{j}(t) * beta{j}(t,:);
       end  
       
    beta{j}(isnan(beta{j}))=eps;
    
    end
    
end