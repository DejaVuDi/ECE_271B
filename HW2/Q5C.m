%% load data
clc;close all;clear;
[xtrain,ltrain] = readMNIST('train-images.idx3-ubyte','train-labels.idx1-ubyte',60000,0);
[xtest,ltest] = readMNIST('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte',10000,0);
xtrain = xtrain';
ltrain = ltrain';
dtrain = label2onehot(ltrain')';
xtest = xtest';
ltest = ltest';
dtest = label2onehot(ltest')';
%% main
layer = 1;
lr = 1e-2;
maxEpoch = 20;
iter = 200;
ni = size(xtrain,1);
no = size(dtrain,1);
netinit = init_single(ni,no);
yinit = forward(xtrain,netinit);
% poe = eval_err(yinit,dtrain);
[Ptr,Pte,nettrain] = back(xtrain,dtrain,netinit,maxEpoch,lr,xtest,dtest);
% ytest = forward(xtest,nettrain);
%% functions
function param = init_single(ni,no)
    % ni = input number, no = output number
    param{1} = randn(no,ni); % w1
    param{2} = randn(no,1); % b1
end
 
function y = forward(x,net)
    w1 = net{1};
    b1 = net{2};
    a1 = w1*x + b1;
    y = active(a1,'softmax');
%      y = softmax(a1);
end
function poe = eval_err(y,d)
    maxidx = onehot2label(y);
    lbl = onehot2label(d);
    poe = nnz(maxidx-lbl)/size(y,2);
end
function param = update(x,d,net,lr)
    w1 = net{1};
    b1 = net{2};
    ni = size(w1,2);
    no = size(b1,1);
    a1 = w1*x + b1;
%     y = softmax(a1);
    y = active(a1,'softmax');
%     e = -d./y;
%     delta = activep(a1,e,'softmaxp');
    delta = -(d-y);
    param{1} = w1-lr*delta*x';
    param{2} = b1-lr*reshape(sum(delta,2),[no,1]);
end
 
function [ptr,pte,trained] = back(x,d,net,T,lr,te,td)
    ptr = zeros(1,T);
    pte = zeros(1,T);
    for t = 1:T
        for i = 1:size(x,2)
            net = update(x(:,i),d(:,i),net,lr);
        end
        test = forward(te,net);
        y = forward(x,net);
        ptr(t) = eval_err(y,d);
        pte(t) = eval_err(test,td);
    end
    trained = net;
    figure
    plot(1:T,ptr);
    hold on
    plot(1:T,pte);
    title('Single layer')
    xlabel('number of epoch')
    ylabel('poe')
    legend('train','test')
end
