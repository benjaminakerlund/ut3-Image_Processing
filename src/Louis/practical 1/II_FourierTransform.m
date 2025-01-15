%% Question 13: FT spectrum of geometric shapes and patterns
width = 512;
height = 512;
stripe_width = 32;
radius = 32;
rect_w = 64;
rect_h = 32;
figure;

image = rectangle(width, height, rect_w, rect_h);
ft_horizontal_stripes = fft2(image, width, height);
ft_horizontal_stripes = fftshift(ft_horizontal_stripes);
ft_spectrum = abs(ft_horizontal_stripes);

subplot(2,4,1)
imshow(image);
subplot(2,4,2)
imagesc(ft_spectrum);

image = circle(width, height, radius);
ft_horizontal_stripes = fft2(image, width, height);
ft_horizontal_stripes = fftshift(ft_horizontal_stripes);
ft_spectrum = abs(ft_horizontal_stripes);

subplot(2,4,3)
imshow(image);
subplot(2,4,4)
imagesc(ft_spectrum);

image = horizontal_stripes(width, height, stripe_width);
ft_horizontal_stripes = fft2(image, width, height);
ft_horizontal_stripes = fftshift(ft_horizontal_stripes);
ft_spectrum = abs(ft_horizontal_stripes);

subplot(2,4,5)
imshow(image);
subplot(2,4,6)
imagesc(ft_spectrum);

image = vertical_stripes(width, height, stripe_width);
ft_horizontal_stripes = fft2(image, width, height);
ft_horizontal_stripes = fftshift(ft_horizontal_stripes);
ft_spectrum = abs(ft_horizontal_stripes);

subplot(2,4,7)
imshow(image);
subplot(2,4,8)
imagesc(ft_spectrum);

%% Question 14: FT spectrum of blurred image

%% Question 15: Extraction of field
clc
clear all
close all

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
%figure, imagesc(log(ft)) % take the log here since the center is magnitude is too strong
%figure, mesh(log(ft))

target_angle = deg2rad(52);
tolerance    = deg2rad(5);

ft_mask = zeros(width, height);
%figure, subplot(1,2,1), imagesc(log(ft)), subplot(1,2,2), imshow(mask)

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

figure, imagesc(spectrum_filtered)

ft_filtered = ifftshift(ft_filtered);
im_mask= ifft2(ft_filtered);
im_mask = abs(im_mask);
%figure, imagesc(im_filtered)

m = max(im_mask(:));
size(im_mask)

for w=1:width
    for h=1:height
        if im_mask(w, h) < m*0.4
            im_mask(w, h) = 0;
        else
            im_mask(w, h) = 1;
        end
    end
end

im_filtered = zeros(width, height, channels);
for i=1:channels
    im_filtered(:,:,i) = im(:,:,i) .* uint8(im_mask);
end

figure, imagesc(uint8(im_filtered))

% apply some kind of blur, actually dilation and then erode