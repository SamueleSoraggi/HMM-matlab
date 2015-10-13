function [mix,mu,sigma] = My_GmmInitKM (data,k)

[N D] = size(data);
mix = zeros(1,k);
sigma = zeros(D,D,k);
mu = zeros(D,k);

[idx,ctr] = kmeans(data,k);

for j=1:k
  mix(j) = sum(idx==j)/N;
  mu(:,j) = mean(data(idx==j,:));
  sigma(:,:,j) = cov(data(idx==j,:));    
end








