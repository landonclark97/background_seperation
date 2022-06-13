function [Ls, Lt] = compute_laplacian(X, pp, pK, pht, phs)
%% create a pair of graph Laplacians in the spatiotemporal domain
% Inputs:
% X: video of size n1 x n2 x m with n1xn2 pixels and m frames
% pars.hs: spatial filtering parameter
%     .ht: temporal filtering parameter
%     .K:  number of nearest neighbors in the spatial domain

% Outputs:
% Ls: graph Laplacian in the spatial domain
% Lt: graph Laplacian in the temporal domain

% Jing Qin, 04-10-2022

%% Read parameters
p = pp; % patch radius in the spatial domain
K = pK; % k-nearest neighbor search
ht= pht; % filtering parameter in the temporal domain
hs= phs; % filtering parameter in the spatial domain

ht = ht^2;
hs = hs^2;

[n1,n2,t] = size(X);
n = n1*n2;

D1 = reshape(X,n,t);

%% Create Lt
[dt,ind] = pdist2(D1',D1','squaredeuclidean','Smallest',K+1);
tmp = exp(-dt(2:end,:)./ht);
idx = reshape(repmat((1:t),K,1),t*K,1); %row index
idy = reshape(ind(2:end,:),t*K,1); %column index
A = sparse(idx,idy,tmp(:),t,t);
A = A+A';
d = 1./sqrt(sum(A,2));
W = spdiags(d,0,t,t);
Lt = speye(t)-W*A*W;

%% Create Ls based on patches
% pad each frame 
D2 = padarray(X, [p p 0], 'symmetric');
ps = 2*p+1; % patch length

% create the matrix with patches as rows for each frame
patches = zeros(n, ps^2, t);
for i = 1:t
    patches(:,:,i) = im2col(D2(:,:,i), [ps, ps], 'sliding')';
end
patches = reshape(patches, n, ps^2*t);

%% Method 1: global search, slow! 
% it takes ~10 minutes for K=4, p=2, 150x200x65 video
% [dt,ind] = pdist2(patches,patches,'squaredeuclidean','Smallest',K+1); 
% tmp = exp(-dt(2:end,:)./hs);

%% Method 2: use all pixels in a centered patch without search (fast)
% compute all ps^2 neighbors in the patch level
% (2p+1)^2-1 %number of pixels in the patch except the center
index = padarray(reshape(1:n,n1,n2), [p p], 'symmetric');
index = im2col(index, [ps, ps], 'sliding');
K = 4*p^2+4*p;
ind = [1:n; index([1:K/2,K/2+2:end],:)];

ds = zeros(K*n,1);
id = 0;
for i = 1:n
    for j = 2:K+1
        id = id + 1;
        di = patches(i,:);
        dj = patches(ind(j,i),:);
        ds(id) = norm(di-dj,2);
    end
end
tmp = exp(-ds.^2/hs);


idx = reshape(repmat((1:n),K,1),n*K,1); %row index
idy = reshape(ind(2:end,:),n*K,1); % column index
A = sparse(idx,idy,tmp,n,n);  % adjacency matrix
A = A+A';
d = 1./sqrt(sum(A,2));
W = spdiags(d,0,n,n);
Ls = speye(n)-W*A*W;

