function mean_iou = mean_iou_4bins(bin_im1,bin_im2)
%MEAN_IOU_4BINS calculates the Mean IoU of two binary images

iou1 = jaccard(bin_im1, bin_im2);
iou2 = jaccard(~bin_im1, ~bin_im2);
mean_iou = (iou1 + iou2) / 2;

end

