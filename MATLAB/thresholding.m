close all; clear; clc;
addpath('D:\Gerasimos\matlab_utility_functions\npy-matlab\npy-matlab');

default_path = 'D:\Gerasimos\my_icdar\models';

% Select the files with the predictions
[files, path] = uigetfile('*.npy', 'Select one or more NPY files', ...
    default_path,'MultiSelect','on');

evm_flag = questdlg('Apply the Extreme Values method on the images?', ...
    '', 'Yes', 'No', 'Yes');
switch evm_flag
    case 'Yes'
        evm_flag = true;
    case 'No'
        evm_flag = false;
end

if evm_flag == true
    subplot_rows_num = 3;

    prompt = {'Kernel size (scalar)', 'Threshold (\in [0, 1])'};
    dlgtitle = 'Define the parameters for the Extreme Values method';
    dims = [1, 30];
    defaults = {'7', '0.2'};
    opts.Interpreter = 'tex';
    evm_params = inputdlg(prompt, dlgtitle, dims, defaults, opts);
    kernel_size = str2double(evm_params(1));
    threshold = str2double(evm_params(2));

else
    subplot_rows_num = 2;

end

% If only one file is selected, convert the variable "files" into a Cell 
% array to avoid getting an error on the following lines of the code
if ischar(files)
    files = {files};
end

files_number = length(files);

for i = 1:files_number

    % Load the array with the input images, the ground truth and the
    % predictions
    images = readNPY(fullfile(path, files{i}));

    if i == 1
        input_image = images(:, :, 1:3);
        ground_truth = logical(images(:, :, 4));
    end
    
    % Check if the selected files refer to predictions on the same input
    if i ~= 1
        if isequal(images(:, :, 1:3), input_image) == false
            error(['The Selected files must refer to predictions calculated' ...
                'on the same input image!'])
        end
    end    

    average_predictions = mean(images(:, :, 5:end), 3);
    binary_predictions_equal = imbinarize(average_predictions, 0.5);
    
    level = graythresh(average_predictions);
    binary_predictions_otsu = imbinarize(average_predictions, level);

    figure('Name','Post-processing the predictions','NumberTitle','off');
    if evm_flag == true
        updated_average_predictions = extreme_values_method( ...
            average_predictions, kernel_size, threshold);

        binary_predictions_equal_evm = imbinarize( ...
            updated_average_predictions, 0.5);

        level_evm = graythresh(updated_average_predictions);
        binary_predictions_otsu_evm = imbinarize( ...
            updated_average_predictions, level_evm);
    end

        subplot(subplot_rows_num, 2, 5)
        imshow(binary_predictions_equal_evm);
        title(["Binarized Image with equal class weights (threshold = 0.5)", ...
            "Extreme Values method was applied (kernel size = " ...
            + kernel_size + ", Threshold = " + threshold + ")"])
        xlabel("Mean IoU = " + mean_iou_4bins(binary_predictions_equal_evm, ...
        ground_truth))
        subplot(subplot_rows_num, 2, 6)
        imshow(binary_predictions_otsu_evm);
        title(["Binarized image using Otsu's method", ...
            "Extreme Values method was applied (kernel size = " ...
            + kernel_size + ", Threshold = " + threshold + ")"])
        xlabel("Mean IoU = " + mean_iou_4bins(binary_predictions_otsu_evm, ...
        ground_truth))
    
    subplot(subplot_rows_num, 2, 1)
    imshow(input_image);
    title('Input Image')
    subplot(subplot_rows_num, 2, 2)
    imshow(ground_truth);
    title('Ground Truth')
    subplot(subplot_rows_num, 2, 3)
    imshow(binary_predictions_equal);
    title('Binarized Image with equal class weights (threshold = 0.5)')
    xlabel("Mean IoU = " + mean_iou_4bins(binary_predictions_equal, ...
        ground_truth))
    subplot(subplot_rows_num, 2, 4)
    imshow(binary_predictions_otsu);
    title("Binarized image using Otsu's method")
    xlabel("Mean IoU = " + mean_iou_4bins(binary_predictions_otsu, ...
        ground_truth))


    %NOOOOTTEEEEE: Make the suptitle!!!
    % - Build title axes and title.
    axes( 'Position', [0, 0.95, 1, 0.05] ) ;
    set( gca, 'Color', 'None', 'XColor', 'None', 'YColor', 'None' ) ;
    text( 0.5, 0, 'My Nice Title', 'FontSize', 14', 'FontWeight', 'Bold', ...
        'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
end


