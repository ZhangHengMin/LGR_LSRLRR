clear all;
close all;
addpath('dataset'); 
addpath('Commoncodes');
 

CluAcc = zeros(6, 1);
accClu = zeros(6, 6);


allc = [5 8 10];
for i = 1 : length(allc)
    nCluster = allc(i);


    lambda1all =  [1e-6 0.00001 0.0001 0.001 0.01 0.1];
    for iparas1 = 1  : length(lambda1all)
        lambda1 = lambda1all(iparas1);
        disp([' lambda1 = ' num2str(lambda1)]);


        lambda2all =  [1e-6  0.00001 0.0001 0.001 0.01 0.1];
        for iparas2 =  1    : length(lambda2all)
            lambda2 = lambda2all(iparas2);


            load YaleB  % YaleB dataset
            num = nCluster * 64 ;   % number of data used for subspace segmentation
            start = 0 ;
            fea = fea(:,start+1:start+num) ;
            gnd = gnd(:,start+1:start+num) ;

            %% Projection
            % PCA
            [ eigvector , eigvalue ] = PCA( fea ) ;
            maxDim = length(eigvalue);
            fea = eigvector' * fea ;
            % fea = [fea ; ones(1,size(fea,2)) ] ;
            for i = 1 : num
                fea(:,i) = fea(:,i) / norm(fea(:,i)) ;
            end
            d = nCluster*6 ;
            data = fea(1:d,:) ;


            para.knn = 4;
            para.gamma = 6;
            para.elpson = 0.001;
            para.aff_type = 'J2'; 
            p = 0.5;
            rc = 0.01;
            rho = 1.1;
            Q = orth(data');
            A = data*Q;  nX = sqrt(sum(data.^2));
            tic;
            [W, value] = LapSmLRR_IRLS(data, A, p,lambda1,lambda2,rc,rho,para);
            time_cost = toc;
            W = Q*W; J = W;
            if strcmp(para.aff_type,'J1')
                L =(abs(J)+abs(J'))/2;
            elseif strcmp(para.aff_type,'J2')
                L=abs(J'*J./(nX'*nX)).^para.gamma;
            elseif strcmp(para.aff_type,'J2_nonorm')
                L=abs(J'*J).^para.gamma;
            end
            W = L;

            W2 = W;

            %%
            for ic = 1 : size(W,2)
                W2(:,ic) = W(:,ic)/(max(abs(W(:,ic)))+eps) ;
            end

            groups = clu_ncut(W2,max(gnd));
            [ACC, NMI, PUR] = ClusteringMeasure(gnd,groups);

            disp([' lambda2 = ' num2str(lambda2), ' choosecluster = ' num2str(nCluster), ' acc = ' num2str(ACC*100), ' nmi = ' num2str(NMI*100), ' PUR= ' num2str(PUR*100), ' time= ' num2str(time_cost)]);


            CluAcc(iparas2) = ACC*100;

        end

        accClu(:,iparas1)  =  CluAcc;
    end


    eval(['accLGRSmLRRYaleB_' num2str(nCluster) '= accClu']);
    eval(['save accLGRSmLRRYaleB_' num2str(nCluster) ' accLGRSmLRRYaleB_' num2str(nCluster)]);

end


% figure;
% R = W2(1,:);
% stem(R,'LineWidth',1)
% 
% hold on
% 
% correct_class = 1;
% num_train = 64;
% correct_samples = (correct_class-1)*num_train+1:correct_class*num_train;
% % R_Prime = [zeros(1,(correct_class-1)*num_train) R(correct_samples)];
% R_Prime = R(correct_samples);
% stem(correct_samples,R_Prime,'r','LineWidth',1)
% 
% ax1 = gca;
% set(ax1,'FontSize',12)
% legend('Samples with Incorrect Subject','Samples with Correct Subject','Location','northeast','FontSize',15);
% ylabel('Values of Coefficients','interpreter','latex', 'FontSize',15);
% xlabel('Index of Samples','interpreter','latex', 'FontSize',15); 
% 
% set(gcf,'color','w');
% export_fig(gcf, '-pdf', '-r300', '-painters', 'LGRSmLRR_YaleB5.pdf'); 



% figure; 
% hist(W2);  
% ax1 = gca;
% set(ax1,'FontSize',12);
% xlabel(' Values of Coefficients','interpreter','latex', 'FontSize',15);
% ylabel(' Number of Each Column Coefficients','interpreter','latex', 'FontSize',15);
% set(gcf,'color','w');
% export_fig(gcf, '-pdf', '-r300', '-painters', 'LGRSmLRR_YaleB8.pdf');

% figure;
% imagesc(W2); colorbar; 
% ax1 = gca;
% set(ax1,'FontSize',12);
% xlabel(' Index of Samples','interpreter','latex', 'FontSize',15);
% ylabel(' Index of Samples','interpreter','latex', 'FontSize',15);
% set(gcf,'color','w');
% export_fig(gcf, '-pdf', '-r300', '-painters', 'LGRSmLRR_YaleB10.pdf');