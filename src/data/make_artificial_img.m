clear;
close all;


% Make HR sin(x^2/1024)sin(y^2/1024) image
x = 1:0.01:512;
y = 1*sin(1/1024*x.*x);
y = y(1:100:end);
y = y'*y;
M = max(y(:));
m = min(y(:));
Y = (y-m)./(M-m);
imwrite(cat(3, Y, Y, Y),'artifical_img.png');


% Make LR sin(x^2/1024)sin(y^2/1024) image
scale = [2, 3, 4];
im_gt = imread('artifical_img.png');
for s = 1 : length(scale) 
    im_gt = modcrop(im_gt, scale(s));
    im_gt = double(im_gt);
    im_gt_ycbcr = rgb2ycbcr(im_gt / 255.0);
    im_gt_y = im_gt_ycbcr(:,:,1) * 255.0;
    im_l_ycbcr = imresize(im_gt_ycbcr, 1/scale(s), 'bicubic');
    im_b_ycbcr = imresize(im_l_ycbcr, scale(s), 'bicubic');
    im_l_y = im_l_ycbcr(:, :, 1) * 255.0;
    im_l = ycbcr2rgb(im_l_ycbcr) * 255.0;
    im_b_y = im_b_ycbcr(:, :, 1) * 255.0;
    im_b = ycbcr2rgb(im_b_ycbcr) * 255.0;
    filename = sprintf('%s_x%s.mat','artificial_img', num2str(scale(s)));
    save(filename, 'im_gt_y', 'im_b_y', 'im_b_ycbcr', ...
        'im_gt', 'im_b', 'im_l_ycbcr', 'im_l_y', 'im_l');
end


% Define utility function
function imgs = modcrop(imgs, modulo)
    if size(imgs,3)==1
        sz = size(imgs);
        sz = sz - mod(sz, modulo);
        imgs = imgs(1:sz(1), 1:sz(2));
    else
        tmpsz = size(imgs);
        sz = tmpsz(1:2);
        sz = sz - mod(sz, modulo);
        imgs = imgs(1:sz(1), 1:sz(2),:);
    end
end