%% Question 2
clc; clear; close all; 

im_1 = imread('../BE2_IntroMorphoMath/bloodBW.png');
%im_1 = imcomplement(im_1); % invert the colors, else erosion becomes dilation
im_2 = imread('../BE2_IntroMorphoMath/cameraman.tif');

% Answer to question combined
figure; 
sgtitle({'Operating bin image with imerode and imdilate', 'with different structuring elements'});

% rectangle
se_1 = strel('rectangle', [10 20]);
im_eroded = imerode(im_1, se_1);
im_dilated = imdilate(im_1, se_1);
subplot(4, 3, 1), imshow(im_1), title('Original');
subplot(4, 3, 2), imshow(im_eroded), title('Eroded');
subplot(4, 3, 3), imshow(im_dilated), title('Dilated');

% disk
se_2 = strel('disk', 10);
im_eroded = imerode(im_1, se_2);
im_dilated = imdilate(im_1, se_2);
subplot(4, 3, 4), imshow(im_1), title('Original');
subplot(4, 3, 5), imshow(im_eroded), title('Eroded');
subplot(4, 3, 6), imshow(im_dilated), title('Dilated');

% diamond
se_3 = strel('diamond', 10);
im_eroded = imerode(im_1, se_3);
im_dilated = imdilate(im_1, se_3);
subplot(4, 3, 7), imshow(im_1), title('Original');
subplot(4, 3, 8), imshow(im_eroded), title('Eroded');
subplot(4, 3, 9), imshow(im_dilated), title('Dilated');

% line
se_4 = strel('line', 10, 0);
im_eroded = imerode(im_1, se_4);
im_dilated = imdilate(im_1, se_4);
subplot(4, 3, 10), imshow(im_1), title('Original');
subplot(4, 3, 11), imshow(im_eroded), title('Eroded');
subplot(4, 3, 12), imshow(im_dilated), title('Dilated');


%% Question 3
clc; clear; close all;

im = imread('../BE2_IntroMorphoMath/EuropeBW.bmp');
im = imcomplement(im);

se = strel('disk', 1);
edges_internal = imcomplement(im - imerode(im, se));
edges_external = imcomplement(imdilate(im, se) - im);

gradient = imdilate(im, se) - imerode(im, se);

figure
sgtitle('Internal and External edges of bin image')
subplot(1, 4, 1), imshow(im), title('Original')
subplot(1, 4, 2), imshow(edges_internal), title('Internal edges')
subplot(1, 4, 3), imshow(edges_external), title('External edges')
subplot(1, 4, 4), imshow(gradient), title('Morphological gradient')


%% Question 4
clear; clc; close all;

im = imread('../BE2_IntroMorphoMath/EuropeBW.bmp');
im = imcomplement(im);

n_erosions = 25;
[width, height] = size(im);
stack = zeros(width, height, n_erosions+1, 'logical');

stack(:, :, 1) = im;
structural_element = strel('disk', 7);
for i=1:n_erosions
    stack(:, :, i+1) = imerode(stack(:, :, i), structural_element);
end

subplot(1, 2, 1), imshow(im), title('Original')

subplot(1,2,2)
collapsed = sum(stack, 3);
imagesc(collapsed)
c = colorbar;
c.Label.String = 'Distance from the sea in number of erosions';
title('Pixel distances')
%% Question 5
clear, clc;

im = imread('../BE2_IntroMorphoMath/Images2.jpg');
im = imcomplement(im);

figure
subplot(1, 4, 1), imshow(im), title('Original')

se = strel('rectangle', [20 56]);
ime = imerode(im, se);
ime = imdilate(ime, se);
subplot(1, 4, 2), imshow(ime), title('Direct filter of large object')

se = strel('disk', 15);
ime = imerode(im, se);
subplot(1, 4, 3), imshow(ime), title('Filtered large objects')

ime = im - ime;

se = strel('rectangle', [11 35]);
ime = imerode(ime, se);
ime = imdilate(ime, se);
subplot(1, 4, 4), imshow(ime), title('Filtered smaller object')

%% Question 6

clear
clc
