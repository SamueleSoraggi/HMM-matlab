function [sX, gmm_adjusted_obj, out_gmm_obj, vbgmm_obj] = fn_VBGMM_clustering(X, k, PriorPar, options)


% X: N x D
% k: initial cluster for GMM

%standardize the data

% % approach I -- standardize the data around mean with var = 1
% mu_X = mean(X,1);
% X_trans = X - repmat(mu_X,size(X,1),1);
% % sX = X_trans./repmat(var(X_trans,1),size(X,1),1); % normalized by the variance
% sX = X_trans./repmat(sqrt(var(X_trans,1)),size(X,1),1); % normalized by std

% approach II --standardlize the data into 0 - 1
% min_X = min(X,[],1);
% X_trans = X - repmat(min_X,size(X,1),1);
% max_X_trans = max(X_trans,[],1);
sX = X;
%X_trans./repmat(max_X_trans,size(X,1),1); % normalized by max

% figure(1); plot(sX(:,1),sX(:,2),'b*'); title('after standardized');
[N D] = size(X);


if nargin < 4
    % set the options for VBEM
    clear options;
    options.maxIter = 200;
    options.threshold = 1e-5;
    options.displayFig = 0;
    options.displayIter = 0;
end

if nargin < 3
    % intialize the priors %====== Play with these parameters!!!
    % =========================================================================
    PriorPar.alpha = 0.01; %0.001
    PriorPar.mu = zeros(D,1);
    PriorPar.beta = 1;
    PriorPar.W = 2*eye(D); %200
    PriorPar.v = 20;
    % =========================================================================
end

if nargin < 2
    k = 20;
end

%%%%%%%%%%%%%%%%%%%%
% GMM VBEM clustering
%%%%%%%%%%%%%%%%%%%%

% ===================
% initial VBGMM with GMM (in MATLAB)
% ===================
%gmm_obj = gmdistribution.fit(sX,k,'Regularize',1e-4);
[pw, mu, Sigma] = My_GmmInitKM(sX,k);
Sigma = Sigma + repmat(1e-5*eye(D),[1,1,k]);
gmm_obj = gmdistribution(mu',Sigma,pw);

% Call the function gmmVBEM
[vbgmm_obj] = gmmVBEM4(sX, gmm_obj, PriorPar, options);

% ===================================================
% convert the VBGMM object into MATLAB's GMM object
% ===================================================
vbgmm_mu = vbgmm_obj.m';
vbgmm_Sigma = vbgmm_obj.W;
vbgmm_pw = vbgmm_obj.alpha;
for j = 1:k
    vbgmm_Sigma(:,:,j) = inv(vbgmm_obj.W(:,:,j))/(vbgmm_obj.v(j)-D-1);
    vbgmm_pw(j) = (vbgmm_obj.alpha(j) + vbgmm_obj.Nk(j))/(k*vbgmm_obj.alpha0 + N);
end

% ===================================================
% --- remove the trivial component which degrades to 0 ----
% ===================================================
epsilon = 1e-7;
valid_cluster_index = sum(vbgmm_mu.^2-repmat(PriorPar.mu',k,1).^2,2) > epsilon;

vbgmm_mu = vbgmm_mu(valid_cluster_index,:);
vbgmm_Sigma = vbgmm_Sigma(:,:,valid_cluster_index);
vbgmm_pw = vbgmm_pw(valid_cluster_index,:);

% check for positive definite of the Sigma before made to gmm object
for j = 1:size(vbgmm_Sigma,3)
    [R,err] = cholcov(vbgmm_Sigma(:,:,j));
    if isempty(R)
        [U L] = eig(vbgmm_Sigma(:,:,j));
        vbgmm_Sigma(:,:,j) = U*(L+1e-5)*U';
    end
end

out_gmm_obj = gmdistribution(vbgmm_mu,vbgmm_Sigma,vbgmm_pw');

% =======================================
% Since the covariance obtained from VBGMM is too large, hence gives
% incorrect segmentation using GMM, so we might want to use the mean
% obtained from the VBGMM as the initial guess in the GMM again to get a
% good covariance matrix.
% =======================================
K = out_gmm_obj.NComponents;
initial = {};
initial.mu = out_gmm_obj.mu;
initial.Sigma = out_gmm_obj.Sigma; 
initial.PComponents = out_gmm_obj.PComponents;
gmm_adjusted_obj = [];%gmdistribution.fit(sX,K,'Regularize',1e-4,'Start',initial);
