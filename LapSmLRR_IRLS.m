function  [Z, value, Lr] = LapSmLRR_IRLS(X,A, p,lambda,lambda0,rc,rho,para)

% Solve the smoothed LRR problem (6) by IRLS shown in the following paper:
% Canyi Lu, Zhouchen Lin, Shuicheng Yan, Smoothed Low Rank and Sparse
% Matrix Recovery by Iteratively Reweighted Least Squares Minimization,
% IEEE Transactions on Image Processing (TIP), 2014
% 
% Written by Canyi Lu (canyilu@gmail.com), December 2014.
%

if nargin < 10
    display = true;
end
[~, n] = size(X);
XtX = p/2*(A'*A);
XtD = p/2*(A'*X);
maxiter = 200;
mu = rc*norm(X,2);
tol2 = 1e-4;
I = eye(n);
Z = ones(size(XtD));
Z_old = Z;
W1 = eye(n,n);
W2 = ones(n,1); 
%W = W1*diag(W2); 
% if display
%    obj = zeros(maxiter,1);
% end

[pairs,wcost,numpairs]=get_nn_graph(X,para.knn);
R = zeros(size(X,2),numpairs);
for i=1:numpairs
    R(pairs(1,i)+1,i) =  wcost(i);
    R(pairs(2,i)+1,i) = -wcost(i);
end
R = R/(para.knn-1);
L = 0.5*R*R';

Lr = L+para.elpson*eye(size(W1));


%
W = (lambda*Lr+p/2*lambda0*W1)*diag(W2);
for t = 1 : maxiter
   % calculate Z: XtX*Z + Z*W - XtX = 0
   % X = lyap(A,B,C) solves AX+XB+C=0.

   Z = lyap(XtX,W,-XtD);
   %Z = max(Z, 0);
   
%  calculate W1 = (Z^T*Z+mu*I)^{-0.5} with SVD
%    [~,S,V]=svd(Z,'econ');  
%    s = diag(S);   
%    s = 1./sqrt( s.*s + mu^2 );
%    W1 = V*diag(s)*V';
   
   % or calculate W1 = (Z^T*Z+mu*I)^{-0.5} without SVD
   W1 = (Z'*Z+mu^2*I)^(p/2-1);

   % calculate W2 which is a diagonal matrix
   E = X-A*Z;   
   E = dot(E,E);
   W2 =  (E+mu^2).^(1-p/2);   
   W = (lambda*Lr+p/2*lambda0*W1)*diag(W2);  
   
   % update mu
   mu = mu/rho; 
   
   % compute the objective function value
%    if display
%        %EE = X-A*Z; trace(L'*L+mu^2)^(q1/2)+lambda*sum((sum(S.*S)+mu^2).^(q2/2))
%        obj(t) = lambda/2*trace(Z*Lr*Z')+lambda0/2*trace(Z'*Z+mu^2)^(p/2)+1/2*sum((sum(E)+mu^2).^(p/2));
%    end  
   value(t) = norm(Z_old-Z,'fro')/norm(Z,'fro');
   if value(t) <tol2       
       break;
   end 
   Z_old = Z;
end
% if display
%     if t<maxiter
%        obj(t+1:end) = []; 
%     end   
% else    
%     obj = [];
% end

