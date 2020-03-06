function s = active(a,activation)
    switch activation
        case 'softmax'
            [m,~] = size(a);
            dz = a-max(a);
            s = exp(dz)./(ones(m,1)'*exp(dz));
        case 'reLU'
            s = a.*(a>0);
        case 'sigmoid'
            s = 1./(1+exp(-a));
    end
end
