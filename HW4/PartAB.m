%% load data
clc;close all;clear;
% [xtrain,ltrain] = readMNIST('train-images.idx3-ubyte','train-labels.idx1-ubyte',20000,0);
% [xtest,ltest] = readMNIST('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte',10000,0);
% % load('data.mat');
% [ntr,D] = size(xtrain);
% [nte,~] = size(xtest);
load('data.mat');
options = '-s 0 -t 0 -c 2';
nc = 10;
poe = zeros(nc,1);
total = zeros(nc,1);
posSV = zeros(nc,1);
negSV = zeros(nc,1);
prob = zeros(nte,nc);
for i = 0:9
    fprintf('Learning classifier for digit %d ...\n',i)
    binary_labels = -ones(ntr,1);
    binary_labels(ltrain == i) = 1;
    test_labels = -ones(nte,1);
    test_labels(ltest == i) = 1;
    model = svmtrain(binary_labels,xtrain,options);
    [y_predict,accuracy,prob_estimates] = svmpredict(test_labels, xtest, model);
    prob(:,i+1) = prob_estimates; 
    w = model.SVs' * model.sv_coef;
    b = -model.rho;
    margin_test = test_labels.*(xtest*w+b);
    figure
    cdfplot(margin_test);
    xlim([-20 60])
    title(strcat('cdf plot for digit ',int2str(i)))
    saveas(gcf,strcat('c=2/test_cdf_digit',int2str(i),'.png'))
    
    margin_train = binary_labels.*(xtrain*w+b);
    figure
    cdfplot(margin_train);
    xlim([-20 60])
    title(strcat('cdf plot for digit ',int2str(i)))
    saveas(gcf,strcat('c=2/train_cdf_digit',int2str(i),'.png'))
    
    poe(i+1) = accuracy(1);
    total(i+1) = model.totalSV;
    posSV(i+1) = model.nSV(1);
    negSV(i+1) = model.nSV(2);
    
    pos = model.sv_coef(1:model.nSV(1));
    pos_side = find(model.sv_coef==max(pos));
    pos_index = zeros(3,1);
    p = zeros(3,D);
    for j = 1:3
        pos_index(j) = model.sv_indices(pos_side(j));
        p(j,:) = xtrain(pos_index(j),:);
    end
    plot_3(p(1,:),p(2,:),p(3,:));
    saveas(gcf,strcat('c=2/y=1_digit',int2str(i),'.png'))
    
    neg_index = zeros(3,1);
    n = zeros(3,D);
    neg = model.sv_coef(model.nSV(1)+1:model.nSV(2));
    neg_side = find(model.sv_coef==min(neg));
    for k = 1:3
        neg_index(k) = model.sv_indices(neg_side(k));
        n(k,:) = xtrain(neg_index(k),:);
    end
    plot_3(n(1,:),n(2,:),n(3,:));
    saveas(gcf,strcat('c=2/y=-1_digit',int2str(i),'.png'))
    fprintf('End learning for digit %d ...\n',i)
    fprintf('#######################################################\n')
end
[~,overall] = max(prob,[],2);
all = sum((overall-1)==ltest)/nte;
