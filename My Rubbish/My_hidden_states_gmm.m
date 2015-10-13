%funzione di stima dei parametri degli stati nascosti (discreti) per un
%Hmm con Gmm
function [new_trans_prob,new_start_prob] =...
         My_hidden_states_gmm(trans_prob,alpha,beta,Q,B,c,N)
    
    new_trans_prob = zeros(Q);
    %stima prob iniziali
    new_start_prob = alpha(1,:) .* beta(1,:) / c(1); 
    %stima prob transizione tra gli stati
    den = sum(alpha(1:N-1,:) .* (beta(1:N-1,:) ./ repmat(c(1:N-1)',1,Q)));
    den(den==0) = eps;
    den(isnan(den)) = eps;
    for i = 1:Q
        for j = 1:Q
            new_trans_prob(i,j) =...
            ((trans_prob(i,j)*B(2:N,j))'*(alpha(1:N-1,i).*beta(2:N,j)))/...
            den(j);    
        end
    end
   
end