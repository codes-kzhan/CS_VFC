function [ predict_label_final ] = norm_predict_label( predict_label,num_each_track )
%NORM_PREDICT_LABEL Summary of this function goes here
%   Detailed explanation goes here

predict_label_final = [];

for i = 1:1:size(num_each_track,1)
    added = sum(num_each_track(1:i));
    num = num_each_track(i);
    stage = predict_label(added-num+1:added,:);
    table = tabulate(stage);
    [count,idx] = max(table(:,2));
    predict_label_final = [predict_label_final ; idx];
end
end

