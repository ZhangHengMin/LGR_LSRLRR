clear all;
close all;
addpath('dataset'); 


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
        for iparas2 =  1  : length(lambda2all)
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
            para.alpha =  lambda1;
            para.beta = lambda2;
            tic;
            W = LGR_LSR1(data,para);
            time_cost = toc;


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

    eval(['accLGRLSRYaleB_' num2str(nCluster) '= accClu']);
    eval(['save accLGRLSRYaleB_' num2str(nCluster) ' accLGRLSRYaleB_' num2str(nCluster)]);

end
