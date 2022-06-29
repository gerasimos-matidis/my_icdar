close all; clear; clc;
addpath('D:\Gerasimos\matlab_utility_functions\npy-matlab\npy-matlab');

% TODO: Add a general description

% Select the files with the predictions
default_path = 'D:\Gerasimos\my_icdar\models\';
[files, path] = uigetfile('*.npy', 'Select one or more NPY files', ...
    default_path,'MultiSelect','on');

% Raise a question dialog box to choose whether the Extreme Values method 
% is applied to the images
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

    % subplot_rows_num = 2; % IGNORE THIS LINE, WILL BE DELETED
    
    % Raise a input dialog box to define the parameters for the Extreme
    % Values method
    % TODO: To change all these dialog boxes with a modern GUI designed 
    % with App Designer 
    prompt = {'Kernel size (scalar)', 'Threshold (\in [0, 1])'};
    dlgtitle = 'Define the parameters for the Extreme Values method';
    dims = [1, 30];
    defaults = {'11', '0.2'};
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

files_num = length(files);
mean_ious_overall = cell(files_num, 1);
for i = 1:files_num
    % Extract information about the network, with which the predictions
    % were made
    file_info = string(split(replace(files{i}, '.npy', ''), '_'));
    net = upper(file_info(2));
    input_size = replace(file_info(3), 'input', '');
    input_shape = sprintf('(%s, %s, 3)', input_size, input_size);
    initial_images = replace(file_info(5), 'images', '');
    method = join([file_info(6), file_info(7)]); 
    epochs = replace(file_info(10), 'eps', '');
    
    % Load the array with the input images, the ground truth and the
    % predictions
    images = readNPY(fullfile(path, files{i}));

    if i == 1
        input_image = images(:, :, 1:3);
        ground_truth = logical(images(:, :, 4));
    end
    
    % Check if all the selected files refer to predictions on the same 
    % input
    if i ~= 1
        if isequal(images(:, :, 1:3), input_image) == false
            error(['The Selected files must refer to predictions calculated' ...
                'on the same input image!'])
        end
    end    
    
    % Calculate the image with the average predictions
    average_predictions = mean(images(:, :, 5:end), 3);

    % Binarize the image by using equal weights for both classes 
    % (threshold = 0.5). Then calculate the Mean IoU between it and the
    % ground truth
    binary_predictions_equal = imbinarize(average_predictions, 0.5);
    mean_iou_equal = mean_iou_4bins(binary_predictions_equal, ...
        ground_truth);
    
    % Binarize the image with Otsu's method. Then calculate the Mean IoU 
    % between it and the ground truth
    level = graythresh(average_predictions);
    binary_predictions_otsu = imbinarize(average_predictions, level);
    mean_iou_otsu = mean_iou_4bins(binary_predictions_otsu, ...
        ground_truth);
    
    % Create a figure to plot the images with the post-processing results
    figure('Name','Post-processing on the predictions','NumberTitle','off');

    subplot(subplot_rows_num, 2, 1)
    imshow(input_image);
    title('Input Image')
    subplot(subplot_rows_num, 2, 2)
    imshow(ground_truth);
    title('Ground Truth')
    subplot(subplot_rows_num, 2, 3)
    imshow(binary_predictions_equal);
    title('Binarized Image with equal class weights (threshold = 0.5)')
    xlabel("Mean IoU = " + mean_iou_equal)
    subplot(subplot_rows_num, 2, 4)
    imshow(binary_predictions_otsu);
    title("Binarized image using Otsu's method")
    xlabel("Mean IoU = " + mean_iou_otsu)

    suptitle = sprintf(['%s trained on %s initial images for %s epochs' ...
        '\nMethod: %s, input shape = %s'], net, initial_images, epochs, ...
        method, input_shape);
    axes( 'Position', [0, 0.95, 1, 0.05] ) ;
    set( gca, 'Color', 'None', 'XColor', 'None', 'YColor', 'None' ) ;
    text( 0.5, 0, suptitle, 'FontSize', 14', 'FontWeight', 'Bold', ...
        'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
    
    % If requested, apply the Extreme Values method to the images and
    % incorporate the new images into the plot
    if evm_flag == true
        % Update the image with the average predictions by applying the
        % Extreme Values method.
        updated_average_predictions = extreme_values_method( ...
            average_predictions, kernel_size, threshold);

        % Binarize the updated image by using equal weights for both 
        % classes (threshold = 0.5). Then calculate the Mean IoU between it
        % and the ground truth
        binary_predictions_equal_evm = imbinarize( ...
            updated_average_predictions, 0.5);
        mean_iou_equal_evm = mean_iou_4bins(binary_predictions_equal_evm, ...
            ground_truth);
    
        % Binarize the updated image with Otsu's method. Then calculate the
        % Mean IoU between it and the ground truth
        level_evm = graythresh(updated_average_predictions);
        binary_predictions_otsu_evm = imbinarize( ...
            updated_average_predictions, level_evm);
        mean_iou_otsu_evm = mean_iou_4bins(binary_predictions_otsu_evm, ...
        ground_truth);

        subplot(subplot_rows_num, 2, 5)
        % subplot(subplot_rows_num, 2, 3) % IGNORE THIS LINE, WILL BE DELETED
        imshow(binary_predictions_equal_evm);
        title(["Binarized Image with equal class weights (threshold = 0.5)" ...
            , "Extreme Values method was applied (kernel size = " ...
            + kernel_size + ", Threshold = " + threshold + ")"])
        xlabel("Mean IoU = " + mean_iou_equal_evm)
        subplot(subplot_rows_num, 2, 6)
        % subplot(subplot_rows_num, 2, 4) % IGNORE THIS LINE, WILL BE DELETED
        imshow(binary_predictions_otsu_evm);
        title(["Binarized image using Otsu's method", ...
            "Extreme Values method was applied (kernel size = " ...
            + kernel_size + ", Threshold = " + threshold + ")"])
        xlabel("Mean IoU = " + mean_iou_otsu_evm)
        
        % Store the Mean IoU values of each file in a cell
        mean_ious_overall{i} = {method, initial_images, [mean_iou_equal, ...
            mean_iou_otsu, mean_iou_equal_evm, mean_iou_otsu_evm]};

    else
        mean_ious_overall{i} = {method, initial_images, [mean_iou_equal, ...
            mean_iou_otsu]};
    end     
    
end

% Create a bar to compare the Mean IoU scores between the different
% predictions
Y = zeros(length(mean_ious_overall{1}{3}), files_num);
legend_args = cell(files_num, 1);
for i = 1:files_num

    if evm_flag == true
        X = categorical(["Equal", "Otsu", ...
            "Equal + EVM", "Otsu + EVM"]);
    else
        X = categorical(["Equal", "Otsu"]);
    end

    Y(:, i) = mean_ious_overall{i}{3};
    legend_args{i} = sprintf('%s, %s initial images', ...
        mean_ious_overall{i}{1}, mean_ious_overall{i}{2});

end

figure;
h = bar(X, Y);
ylim([0 1])
ylabel('Mean IoU')
set(h, {'DisplayName'}, legend_args)
legend()
