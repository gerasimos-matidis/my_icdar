close all; clear; clc;
addpath('D:\Gerasimos\matlab_utility_functions\npy-matlab\npy-matlab');

default_path = 'D:\Gerasimos\my_icdar\models';

% Load the files with the predictions
[files, path] = uigetfile('*.npy', 'Select one or more NPY files', ...
    default_path,'MultiSelect','on');

% If only one file is selected, turn the variable "files" to a Cell array 
% to avoid getting an error on the following lines of the code
if ischar(files)
    files = {files};
end

for i = 1:length(files)

    % Load the array with the input images, the ground truth and the
    % predictions
    images = readNPY(fullfile(path, files{i}));
    
    % Check if the selected files refer to predictions on the same input
    if i ~= 1
        if isequal(images(:, :, 1:3), input_image) == false
            error(['The Selected files must refer to predictions calculated' ...
                'on the same input image!'])
        end
    end

    input_image = images(:, :, 1:3);
    ground_truth = images(:, :, 4);
    

    
end


