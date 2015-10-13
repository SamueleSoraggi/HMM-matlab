%% normalizzazione matrice quadrata o rettangolare
function An = My_normalizza(A)
[r,c] = size(A);
An = zeros(r,c);
for i = 1 : r
    An(i,:) = A(i,:)/norm(A(i,:),1);    
end
end