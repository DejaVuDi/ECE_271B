%% load data
clc;close all;clear;
[xtrain,ltrain] = readMNIST('train-images.idx3-ubyte','train-labels.idx1-ubyte',20000,0);
[xtest,ltest] = readMNIST('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte',10000,0);
% load('data.mat');
[ntr,D] = size(xtrain);
[nte,~] = size(xtest);
class = 10; % class
threshold = (0:50)'/50; % threshold
T = 250; % 250 weak classifier
it = [5, 10, 50, 100, 250];
 
u_ = @(x,t)(sign(x-t));
u_oppo = @(x,t)(-sign(x-t));
u = @(x,t)[u_(x,t); u_oppo(x,t)];
ui = @(i,x)(sign(x-i/50));
uit = @(i,x)(-sign(x-i/50));
 
Wsave = zeros(T,3,class);
% wsave = zeros(T,1,class);
% wsave = zeros(ntr,T,class)
%%
for c = 1:class
    fprintf('digit %d \n',c-1)
    % label Y
    xtr = xtrain;
    xte = xtest;
    ytr = -ones(ntr,1);
    ytr(ltrain == c-1) = 1;
    yte = -ones(nte,1);
    yte(ltest == c-1) = 1;
    errtr = zeros(T,1);
    errte = zeros(T,1);
    gtr = 0;
    gte = 0;
    w = [];
    weight_log = zeros(T,1); % store weight
    sumi = zeros(2*51,D);
    gammatr = zeros(ntr,5);
    gammate = zeros(nte,5);
    a = ones(1,784)*128;
    for iter = 1:T
        w_bar = exp(-ytr.*gtr);
        [~, weight_log(iter)] = max(w_bar);

        sumi = zeros(2*51,D);
        for i = 1:ntr
            sumi  = sumi+ytr(i)*u(xtr(i,:),threshold)*w_bar(i);
        end
        
        [t, j] = find(sumi==max(sumi(:)));
        t = t(1);
        j = j(1);
        
        if t<=51
            alpha1 = ui(t-1,xtr(:,j));
            alpha2 = ui(t-1,xte(:,j));
        else
            tw = t-51;
            alpha1 = uit(tw-1,xtr(:,j));
            alpha2 = uit(tw-1,xte(:,j));
        end
        
        ind = find(ytr-alpha1);
        eps = sum(w_bar(ind))/sum(w_bar);
        wt = 0.5*log((1-eps)/eps);
        
        gtr = gtr+wt*alpha1;
        gte = gte+wt*alpha2;
 
        example = find(iter == it);
        if ~isempty(example)
            gammatr(:,example) = ytr.*gtr;
        end
            
        w = [w; wt, (t-1)/50, j];
        
        errtr(iter) = 1-sum((sign(gtr).*ytr)>0)/ntr;
        errte(iter) = 1-sum((sign(gte).*yte)>0)/nte;
 
        if rem(iter,50) == 0
           fprintf('t: %d/%d  e: %f\n',iter,T,errte(iter))
        end
    end
    Wsave(:,:,c) = w;
    % wsave(:,:,c) = weight_log;
    
    figure
    plot(1:T,errtr,1:T,errte);
    legend('train','test')
    title(strcat('digit ’,int2str(c-1)))
    
    figure
    for ex = 1:5
        cdfplot(gammatr(:,ex))
        hold on 
    end
    legend('5','10','50','100','250');
    title(strcat(''Margin CDF for digit ',int2str(c-1)))
    
    figure
    plot(1:iter,weight_log)
    title(strcat('Index of largest weight for digit ',int2str(c-1)))
end

%% load data
clc;close all;clear;
load('data.mat')
load('di_try/W.mat');
%%
for c = 1:10
    a = ones(1,784)*128;
    w = Wsave(:,:,c);
    index = w(:,3);
    u = (w(:,2)<=1); 
    for i = 1:250
        if u(i) == 1
            a(index(i)) = 0;
        elseif u(i) == 0
            a(index(i)) = 255;
        end
    end
    figure
    imshow(reshape(a,28,28)'/255);
    title(strcat('digit ',int2str(c-1)))
end
%% three heaviest
for c = 1:10
    w = wsave(:,:,c);
    [~,index] = max(w);
    [n,bin] = hist(index, unique(index));
    [~,idx] = sort(-n);
    heavy = bin(idx(1:3));
    
    for i = 1:3   
        images = reshape(xtrain(heavy(i),:),28,28)';
        figure
        imshow(images);
    end
end

function plot_3(a,b,c)
    figure
    subplot(1,3,1)
    imshow(a);
    subplot(1,3,2)
    imshow(b);
    subplot(1,3,3)
    imshow(c);
end
