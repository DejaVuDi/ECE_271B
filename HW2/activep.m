function s = activep (a,e,activation)
    switch activation
        case 'softmaxp'
            x = active(a,'softmax');  
            p = x.*e;    
            ip = sum(p);    
            s = p-ip.*x;
        case 'reLUp'
            s = e .* (a > 0);
        case 'sigmoidp'
            x = active(a,'sigmoid');
            s = e.*(x.*(1-x));
    end
end
   
