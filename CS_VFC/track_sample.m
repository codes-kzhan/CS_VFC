function [ sample_index, num_each_track,sample_label,gt_sample_label]=track_sample(labels,sample_num)
label = labels(:,2);
classes = unique(label);
sample_index = [];
num_each_track = [];
for i = 1:1:length(classes)
    track_index = find(label== classes(i));
    if length(track_index) <= sample_num
        sample_index = [sample_index; track_index];
        num_each_track = [num_each_track; length(track_index)];
        continue;
    else
        sample_step = floor(length(track_index) ./ sample_num);
        
        for j = 1:1:sample_num
            sample_index = [sample_index;track_index(sample_step * j)];
        end
        num_each_track = [num_each_track; sample_num];
    end
end
sample_label = label(sample_index,:);
gt_sample_label = labels(sample_index,1);

end
