function [TP, TN, FP, FN] = logical_relations(observed_image,target_image)
%LOGICAL_RELATIONS Summary of this function goes here
%   Detailed explanation goes here
TP = observed_image & target_image;
TN = ~observed_image & ~target_image;
FP = observed_image & ~target_image;
FN = ~observed_image & target_image;
end

