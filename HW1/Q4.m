%% initializaiton
clc;
close all;
clear;
alpha = 2;
s = 10;
mu1 = [alpha 0];
mu2 = -mu1;
sigma = [1 0;0 s];
A = mvnrnd(mu1, sigma, 500);
B = mvnrnd(mu2, sigma, 500);
figure
plot(A(:,1),A(:,2),'.', B(:,1),B(:,2),'.')
axis equal
%% PCA
[U,S,V] = svd([A;B]);
hold on
quiver(0,0,V(1,1)*5,V(2,1)*5,0,'linewidth',2,'MaxHeadSize',0.8,'color','k')
%% LDA
w = (cov(A)+cov(B))\(mean(A)'-mean(B)');
hold on
quiver(0,0,w(1)*1.4,w(2)*1.4,0,'linewidth',2,'MaxHeadSize',0.8,'color','k')
