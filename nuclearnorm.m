function nnorm = nuclearnorm( X )

% �������X��nuclear norm, |X|_*

s = svd(X) ;
nnorm = sum( s ) ;

% index = 1 : 7 ;%length(s) ;
% nnorm = sum( s(index) ) ;