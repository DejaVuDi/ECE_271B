function lbl = onehot2label(d)
%     lbb = sum(bsxfun(@times, d, cast(0:9,class(d))), 2);
    [~,lbl] = max(d);
    lbl = lbl-1;
end

