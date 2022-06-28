function [image, outliers_id] = extreme_values_method(image, kernel_size, threshold)
% EXTREME_VALUES_METHOD applies the method of extreme values on a 
% single-channel image.
% It takes as arguments the image, a kernel size, which is a positive, odd
% number that defines the side of a square kernel, and a threshold. The
% value of each pixel is compared to the average value of its neighbours, 
% according to the kernel. If their difference is greater than the 
% threshold, the pixel value is replaced by the average value of its
% neighbours, otherwise it remains the same.
% It returns the updated image and a binary image that indicates the 
% outliers

% Create the kernel
assert(kernel_size > 0 & mod(kernel_size, 2) ~= 0, ...
    'The kernel size must be defined by a positive, odd number!')
kernel = ones(kernel_size);
central_pixel_id = ceil(kernel_size / 2);
kernel(central_pixel_id, central_pixel_id) = 0;

% Normalization factor
s = 1 / (kernel_size^2 - 1); 

% Calculate the average values of the neighbours and subtract them from the
% image
neighbors_avg = s * conv2(image, kernel, 'same');
differences = abs(image - neighbors_avg);

% Replace the outliers with the average value of the neighbours
outliers_id = differences > threshold;
image(outliers_id) = neighbors_avg(outliers_id);

end

