%funzione di stima dei parametri degli stati nascosti (discreti) per un
%Hmm con Gmm
function [new_trans_prob,new_start_prob] =...
         My_hidden_states_gmm(trans_prob,alpha,beta,Q,B,c,P_OL)
    
    dimB = length(B);
    new_trans_prob = zeros(Q);
    %stima prob iniziali
    new_start_prob = zeros(1,Q);
    
    for j = 1:dimB
    new_start_prob = new_start_prob + alpha{j}(1,:) .* beta{j}(1,:) / ( c{j}(1)*P_OL(j) ); 
    end
    
    %stima prob transizione tra gli stati
    for j = 1:dimB
        
        N = size(B{j},1);
        den = 0;
        
        for j = 1:dimB
            den = den + sum(alpha{j}(1:N-1,:) .* (beta{j}(1:N-1,:) ./ repmat(c{j}(1:N-1)',1,Q))) /P_OL(j);
            den(den==0) = eps;
            den(isnan(den)) = eps;
        end
                
        for i = 1:Q
            for n = 1:Q
                new_trans_prob(i,n) = new_trans_prob(i,n) + ...
                ((trans_prob(i,n)*B{j}(2:N,n))'*(alpha{j}(1:N-1,i).*beta{j}(2:N,n)) / P_OL(j) ) / den(n);    
            end
        end
    
    end
   
end