%% Question 13: FT spectrum of geometric shapes and patterns
clc, clear, close all

images = {'../Q3_rect.png', '../Q3_circ.png', '../Q3_vstripes.png', '../Q3_hstripes.png'};
titles = {'Rectangle', 'Circle', 'Vertical stripes', 'Horizontal stripes'};

for i=1:length(images)
    im = imread(images{i});
    im = rgb2gray(im);
    [h, w] = size(im);
    ft = fft2(im);
    ft = fftshift(ft);
    spectrum = abs(ft);
    figure, imagesc(spectrum), axis equal off, ylim([0 h])
    exportgraphics(gcf, strcat('../Q13_', titles{i}, '.png'))
    figure, imagesc(log(spectrum)), axis equal off, ylim([0 h])
    exportgraphics(gcf, strcat('../Q13_log_', titles{i}, '.png'))
end

%% Question 14: FT spectrum of blurred image
clc, clear, close all

images = {'../Q3_rect.png', '../Q3_circ.png', '../Q3_vstripes.png', '../Q3_hstripes.png'};
titles = {'Rectangle', 'Circle', 'Vertical stripes', 'Horizontal stripes'};

for i=1:length(images)
    im = imread(images{i});
    im = rgb2gray(im);
    im = {imfilter(im, fspecial('disk', 10)), imfilter(im, fspecial('disk', 50))};
    ft = {fft2(im{1}), fft2(im{2})};
    ft = {fftshift(ft{1}), fftshift(ft{2})};
    spectrum = {abs(ft{1}), abs(ft{2})};
    figure, imagesc(spectrum{1}), axis equal off
    exportgraphics(gcf, strcat('../Q14_Disk10_', titles{i}, '.png'))
    figure, imagesc(log(spectrum{1})), axis equal off
    exportgraphics(gcf, strcat('../Q14_Disk10_log_', titles{i}, '.png'))
    figure, imagesc(spectrum{2}), axis equal off
    exportgraphics(gcf, strcat('../Q14_Disk50_', titles{i}, '.png'))
    figure, imagesc(log(spectrum{2})), axis equal off
    exportgraphics(gcf, strcat('../Q14_Disk50_log_', titles{i}, '.png'))
end

%% Question 15: Extraction of field
clc, clear, close all

im = imread('champs.bmp');
[width, height, channels] = size(im);
% figure, imagesc(im)

function [ft, spectrum] = fft_spectrum(image)
    image = rgb2gray(image); % convert to grayscale as color is not important
    ft = fft2(image);
    ft = fftshift(ft);
    spectrum = abs(ft);
end

[ft, spectrum] = fft_spectrum(im);

target_angle = deg2rad(52);
tolerance    = deg2rad(5);

ft_mask = zeros(width, height);
center_cutoff = 32;
for w=1:width
    for h=1:height
        x = width/2 - w;
        y = height/2 - h;
        if abs(x) < center_cutoff && abs(y) < center_cutoff
            continue
        end
        angle = atan2(y, x);
        if (target_angle - tolerance) < angle && angle < (target_angle + tolerance)
            ft_mask(w, h) = 1;
        end
    end
end
ft_mask = ft_mask + rot90(ft_mask, 2);
ft_filtered = ft.*ft_mask;
spectrum_filtered = spectrum.*ft_mask;

figure, axis('equal'), imagesc(spectrum_filtered), exportgraphics(gca, '../Q15_FilteredSpectrum.png')

ft_filtered = ifftshift(ft_filtered);
im_mask= ifft2(ft_filtered);
im_mask = abs(im_mask);

m = max(im_mask(:));
im_mask(im_mask < m*0.4) = 0;
im_mask(im_mask > m*0.4) = 1;

im_filtered = zeros(width, height, channels);
for i=1:channels
    im_filtered(:,:,i) = im(:,:,i) .* uint8(im_mask);
end

figure, imshow(uint8(im_filtered)), exportgraphics(gca, '../Q15_ExtractedField.png')