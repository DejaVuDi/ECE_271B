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
lr = 2e-6;
% maxEpoch = 10;
iter = 6000;
lambda = 0.0001;
decay = lambda*2/lr;
ni = size(xtrain,1);
no = size(dtrain,1);
nh = 20;
netinit = init_two(ni,no,nh);
yinit = forward(xtrain,netinit);
% poe = eval_err(yinit,dtrain);
[Ptr,Pte,nettrain] = back(xtrain,dtrain,netinit,iter,lr,xtest,dtest,decay);
% ytest = forward(xtest,nettrain);
%% functions
function param = init_two(ni,no,nh)
    % ni = input number, no = output number, nh = hidden unit number
    param{1} = randn(nh,ni); % w1
    param{2} = randn(nh,1); % b1
    param{3} = randn(no,nh); % w2
    param{4} = randn(no,1); % b2
end
 
function y = forward(x,net)
    w1 = net{1};
    b1 = net{2};
    w2 = net{3};
    b2 = net{4};
    
    a1 = w1*x + b1;
    h1 = active(a1,'reLU');
    a2 = w2*h1 + b2;
    y = active(a2,'softmax');
%      y = softmax(a1);
end
function poe = eval_err(y,d)
    maxidx = onehot2label(y);
    lbl = onehot2label(d);
    poe = nnz(maxidx-lbl)/size(y,2);
end
function param = update(x,d,net,lr,decay)
    w1 = net{1};
    b1 = net{2};
    w2 = net{3};
    b2 = net{4};
    ni = size(w1,2);
    nh = size(w1,1);
    no = size(w2,1);
    
    a1 = w1*x + b1;
    h1 = active(a1,'reLU');
    a2 = w2*h1 + b2;
    y = active(a2,'softmax');
%     a1 = w1*x + b1;
%     y = softmax(a1);
%     y = active(a1,'softmax');
%     e = -d./y;
%     e = y-d;
%     delta2 = activep(a1,e,'softmaxp');
    delta2 = y-d;
    delta1 = activep(a1, w2'*delta2+b2'*delta2,'reLUp');
%     dg = a1.*(1-a1);
%     dg = matlabFunction(diff(active(a1,'sigmoid')));
%     delta1 = dg.*(w2'*delta2+b2'*delta2);
    
    param{3} = w2-lr*(delta2*h1'+decay.*w2);
    param{1} = w1-lr*(delta1*x'+decay.*w1);
    param{4} = b2-lr*reshape(sum(delta2,2),[no,1]);
    param{2} = b1-lr*reshape(sum(delta1,2),[nh,1]);
end
 
function [ptr,pte,trained] = back(x,d,net,T,lr,te,td,decay)
    ptr = zeros(1,T);
    pte = zeros(1,T);
    for t = 1:T
        net = update(x,d,net,lr,decay);
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
    title('reLU H = 20, weight decay')
    xlabel('number of iteration')
    ylabel('poe')
    legend('train','test')
end
