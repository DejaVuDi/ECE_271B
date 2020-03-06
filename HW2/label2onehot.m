function d = label2onehot(y)
    [~, loc] = ismember(y, unique(y));
    y_one_hot = ind2vec(loc')';
    d = full(y_one_hot);
end
    
