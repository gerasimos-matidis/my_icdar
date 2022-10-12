close all; clear; clc;
addpath('D:\Gerasimos\matlab_utility_functions\npy-matlab\npy-matlab');

% TODO: Add a general description

% Select the files with the predictions
default_path = 'D:\Gerasimos\my_icdar\models\';
[files, path] = uigetfile('*.npy', 'Select one or more NPY files', ...
    default_path,'MultiSelect','on');

% If only one file is selected, convert the variable "files" into a Cell 
% array to avoid getting an error on the following lines of the code
if ischar(files)
    files = {files};
end

files_num = length(files);
mean_ious_overall = cell(files_num, 1); % TODO: check if this variable is used in the code!!!
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
        input_image = uint8(images(:, :, 1:3));
        ground_truth = logical(images(:, :, 4));
        figure;
        imshow(input_image);
        title('Input image')
    else
        % Check if all the selected files refer to predictions on the same 
        % input
        if isequal(uint8(images(:, :, 1:3)), input_image) == false
            error(['The Selected files must refer to predictions calculated ' ...
                'on the same input image!'])
        end
    end    
    
    % Calculate the image with the average predictions
    average_predictions = mean(images(:, :, 5:end), 3);
    
    % Binarize the image with Otsu's method. Then calculate the Mean IoU 
    % between it and the ground truth
    level = graythresh(average_predictions);
    binary_predictions = imbinarize(average_predictions, level);
    mean_iou = mean_iou_4bins(binary_predictions, ground_truth);
    
    % Calculate the logical relations between the binarized and the ground
    % truth image, as long as, the percentage of each case (True
    % Positives/Negatives and False Positives/Negatives) in the image. Then
    % create an image to show them all together
    [TP, TN, FP, FN] = logical_relations(binary_predictions, ground_truth);
    synthetic_image = uint8(TP + FP * 2 + FN * 3);
    n = numel(binary_predictions);
    tp_percent = sum(TP, 'all') / n *100;
    tn_percent = sum(TN, 'all') / n *100;
    fp_percent = sum(FP, 'all') / n *100;
    fn_percent = sum(FN, 'all') / n *100;
    
    % Create a figure to plot the images with the post-processing results
    figure('Name', 'Post-processing on the predictions', 'NumberTitle', ...
        'off');
    
    tiledlayout(1,3, "TileSpacing","tight")
    nexttile
    image(ground_truth);
    title('Ground Truth')
    axis image off

    nexttile
    image(binary_predictions);
    title("Binarized image using Otsu's method")
    xlabel(sprintf("Mean IoU = %.4f", mean_iou))
    axis image off
    ax = gca;
    ax.XAxis.Label.Color = [0 0 0];
    ax.XAxis.Label.Visible = 'on';

    nexttile
    image(synthetic_image);
    title('Logical relations between the images')
    xlabel(sprintf('TP: %.1f%%, TN: %.1f%%, FP: %.1f%%, FN: %.1f%%', ...
        tp_percent, tn_percent, fp_percent, fn_percent))
    cmap = colormap([0 0 0;1 1 1;1 0 0;0 0 1]);
    axis image off
    ax = gca;
    ax.XAxis.Label.Color = [0 0 0];
    ax.XAxis.Label.Visible = 'on';
    
    hold on
    for K = 1 : 4 
        s(K) = surf(uint8(K-[1 1;1 1])); 
    end
    hold off

    legend(s, {'True Negative', 'True Positive', 'False Positive', ...
        'False Negative'})

    suptitle = sprintf(['%s trained on %s initial images for %s epochs' ...
        '\nMethod: %s, input shape = %s'], net, initial_images, epochs, ...
        method, input_shape);
    axes( 'Position', [0, 0.95, 1, 0.05] ) ;
    set( gca, 'Color', 'None', 'XColor', 'None', 'YColor', 'None' ) ;
    text( 0.5, 0, suptitle, 'FontSize', 14', 'FontWeight', 'Bold', ...
        'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
end