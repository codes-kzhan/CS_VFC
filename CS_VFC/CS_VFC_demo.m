%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                        %          
%  This is a demo for Video Face Clustering via Constrainted Sparse Subspace Clustering  %
%  You can email chengjuzhou@outlook.com if you have any question. Thanks!               %
%                                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('scsr/');

% load the data
load ('./data/whole_NH_downsample.mat');

sample_num = 3; % can tune
num_clusters = 5;
labels = dataset(:,1:2);
classes = unique(labels);

% sampling from face track
[sample_index,num_each_track,sample_label,gt_sample_label] = track_sample(labels,sample_num);


data = dataset(sample_index,:);


% re_compute the labels and classes after sampling
labels = data(:,2);
classes = unique(labels);

data = data(:,3:size(data,2));
data = data';
[num_dimension num_sample] = size(data);


% the constraints
mlink_pair = [];
clink_pair = [];

for i = 1:1:length(classes)
    track_index = find(labels == classes(i));
    
    for j = 1:1:length(track_index)
        for k = j+1:1:length(track_index)
            mlink_pair = [mlink_pair; track_index(j) track_index(k)];
        end 
    end 
end 

cannot_tracks = no_link_pair;

for i = 1:1:size(cannot_tracks,1)
    track1_index = find(labels == cannot_tracks(i,1));
    track2_index = find(labels == cannot_tracks(i,2));
    
    for j = 1:1:length(track1_index)
        for k = 1:1:length(track2_index)
            clink_pair = [clink_pair;track1_index(j) track2_index(k)];
        end
    end
end

% construct constraint using in sparse representation
constraint = zeros(num_sample,num_sample);

for i = 1:1:size(mlink_pair,1)
    left = mlink_pair(i,1);
    right = mlink_pair(i,2);
    constraint(left,right) = 1;
    constraint(right,left) = 1;
end

for i = 1:1:size(clink_pair,1)
    left = clink_pair(i,1);
    right = clink_pair(i,2);
    constraint(left,right) = 1;
    constraint(right,left) = 1;
end

% use constraints
 constraint = constraint + eye(num_sample);


alpha = 20;
r = 0; outlier = true; rho = 1;
clear dataset;

% the constraints sparse representation
% data is DxN, D is the dimention, N is the sample number 
[W] = SSC(data,r,constraint,alpha,outlier,rho,num_clusters);
 
            
% reuse the constraints
W_ml = zeros(num_sample,num_sample);
W_cl = zeros(num_sample,num_sample);

for i = 1:1:size(mlink_pair,1)
    left = mlink_pair(i,1);
    right = mlink_pair(i,2);
    W_ml(left,right) = 1;
    W_ml(right,left) = 1;
end
W_ml = W_ml + eye(num_sample);

for i = 1:1:size(clink_pair,1)
    left = clink_pair(i,1);
    right = clink_pair(i,2);
    W_cl(left,right) = -1;
    W_cl(right,left) = -1;
    
end

ratio_ml =20 ; % can tune
ratio_cl = 0;
W_final = W + ratio_ml*W_ml + ratio_cl*W_cl;
fprintf('the ratio_ml is : %f \n',ratio_ml);
fprintf('the ratio_cl is : %f \n',ratio_cl);

predict_label = SpectralClustering(W_final,num_clusters);

predict_label_final = norm_predict_label(predict_label,num_each_track);

% Notting-Hill ground truth
num_each_class=[18 11 20 13 14];% the number of tracks that an individual have

[confusion_matrix,trace_max]=confusion_compute(predict_label_final,num_each_class);

fprintf('trace_max is : %d \n',trace_max);
fprintf('Accuracy is: %f \n', trace_max/sum(num_each_class));




