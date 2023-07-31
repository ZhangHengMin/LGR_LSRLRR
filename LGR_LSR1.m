function [L] = LGR_LSR1(X,para)
%This routine solves the smooth representation problem with fro-norm data term
%
alpha = para.alpha;
gamma = para.gamma;
beta = para.beta;

[d,n] =size(X);
nX = sqrt(sum(X.^2));
% step 1 laplacian MATRIX
[pairs,wcost,numpairs]=get_nn_graph(X,para.knn);
R = zeros(n,numpairs);
for i=1:numpairs
    R(pairs(1,i)+1,i) =  wcost(i);
    R(pairs(2,i)+1,i) = -wcost(i);
end
R = R/(para.knn-1);
xtx = X'*X;
rtr = 0.5*R*R';


% step 2 Coefficient MATRIX
A = xtx + beta*eye(n);
B = alpha*(rtr+para.elpson*eye(size(xtx)));
C = - xtx;
J = lyap(A,B,C); %AX + XB + C = 0
J = max(J, 0);
%L = J;

if strcmp(para.aff_type,'J1')
    L =(abs(J)+abs(J'))/2;
elseif strcmp(para.aff_type,'J2')
    L=abs(J'*J./(nX'*nX)).^gamma;
elseif strcmp(para.aff_type,'J2_nonorm')
    L=abs(J'*J).^gamma;
end