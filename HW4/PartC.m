%% load data
clc;close all;clear;
% [xtrain,ltrain] = readMNIST('train-images.idx3-ubyte','train-labels.idx1-ubyte',20000,0);
% [xtest,ltest] = readMNIST('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte',10000,0);
% % load('data.mat');
% [ntr,D] = size(xtrain);
% [nte,~] = size(xtest);
load('data.mat');
options = '-s 0 -t 2 -c 32 -g 0.015625';
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
%     xlim([-20 60])
    title(strcat('cdf plot for digit ',int2str(i)))
    saveas(gcf,strcat('c/test_cdf_digit',int2str(i),'.png'))
    
    margin_train = binary_labels.*(xtrain*w+b);
    figure
    cdfplot(margin_train);
%     xlim([-20 60])
    title(strcat('cdf plot for digit ',int2str(i)))
    saveas(gcf,strcat('c/train_cdf_digit',int2str(i),'.png'))
    
    poe(i+1) = accuracy(1);
    total(i+1) = model.totalSV;
    posSV(i+1) = model.nSV(1);
    negSV(i+1) = model.nSV(2);
    
    pos_side = zeros(3,1);
    pos = model.sv_coef(1:model.nSV(1));
    pos_sort = sort(pos,'descend');
    pos_side(1) = find(model.sv_coef==pos_sort(1));
    pos_side(2) = find(model.sv_coef==pos_sort(2));
    pos_side(3) = find(model.sv_coef==pos_sort(3));
    pos_index = zeros(3,1);
    p = zeros(3,D);
    for j = 1:3
        pos_index(j) = model.sv_indices(pos_side(j));
        p(j,:) = xtrain(pos_index(j),:);
    end
    plot_3(p(1,:),p(2,:),p(3,:));
    saveas(gcf,strcat('c/y=1_digit',int2str(i),'.png'))
    
    neg_index = zeros(3,1);
    n = zeros(3,D);
    neg = model.sv_coef(model.nSV(1)+1:model.nSV(2));
    neg_side = zeros(3,1);
    neg_sort = sort(neg,'descend');
    neg_side(1) = find(model.sv_coef==neg_sort(1));
    neg_side(2) = find(model.sv_coef==neg_sort(2));
    neg_side(3) = find(model.sv_coef==neg_sort(3));    
    for k = 1:3
        neg_index(k) = model.sv_indices(neg_side(k));
        n(k,:) = xtrain(neg_index(k),:);
    end
    plot_3(n(1,:),n(2,:),n(3,:));
    saveas(gcf,strcat('c/y=-1_digit',int2str(i),'.png'))
    fprintf('End learning for digit %d ...\n',i)
    fprintf('#######################################################\n')
end
[~,overall] = max(prob,[],2);
all = sum((overall-1)==ltest)/nte;
