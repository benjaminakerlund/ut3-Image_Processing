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
clear; clc; close all;

im = imread('../BE2_IntroMorphoMath/Images2.jpg');
im = imcomplement(im);

figure, sgtitle('Algorithm response for detecting rectangular objects')
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
clear; clc; close all;

function ime = opening(im, strel)
    ime = imerode(im, strel);
    ime = imdilate(ime, strel);
end

function ime = closing(im, strel)
    ime = imdilate(im, strel);
    ime = imerode(ime, strel);
end

im = imread('../BE2_IntroMorphoMath/Images1.jpg');
im = imcomplement(im);

se = strel('disk', 15);
closed = imclose(im, se);
opened = imopen(im, se);

figure, sgtitle('Opening and Closing Morphological filters')
subplot(1, 3, 1), imshow(im), title('Original')
subplot(1, 3, 2), imshow(closed), title('Closed')
subplot(1, 3, 3), imshow(opened), title('Opened')

%% Question 7
% See question 5

%% Question 8
clc; clear; close all;

% Salt and pepper noise
im = imread('../BE2_IntroMorphoMath/Images2.jpg');
im = imnoise(im, 'salt & pepper');
im = imcomplement(im);

% Question 5 algorithm:
figure, sgtitle('Shape detection algorithm with noise')
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


%% Question 9
clear; clc; close all;

im = imread('../BE2_IntroMorphoMath/Nebuleuse.jpg');

% add salt and pepper noise
im_noisy = imnoise(im, 'salt & pepper');

% use opening to filter out noise
se = strel('disk', 1);
im_filtered = imopen(im, se);

figure
subplot(1,3,1), imshow(im), title('original')
subplot(1,3,2), imshow(im_noisy), title('noisy')
subplot(1,3,3), imshow(im_filtered), title('filtered')

%% Question 10
clear; clc; close all;
RGB = imread("../BE1_IntroComputerVision/SpainBeach.jpg");
HSV = rgb2hsv(RGB);
HSV_filtered = HSV;
mask = HSV_filtered(:,:,1) > 0.05 & HSV_filtered(:,:,1) < 0.06;

figure, sgtitle('Filtering beaches')
subplot(3,2,1), imshow(RGB), title('Original')
subplot(3,2,2), imshow(HSV_filtered), title('HSV')
subplot(3,2,3), imshow(mask), title('Mask1: bright pixels')

se = strel('disk', 2);
mask = imopen(mask, se);
subplot(3,2,4), imshow(mask), title('Mask2: noise filtering')

se = strel('disk', 25);
mask = imdilate(mask, se);
subplot(3,2,5), imshow(mask), title('Combined Mask')
im_filtered = RGB .* uint8(mask);
subplot(3,2,6), imshow(im_filtered), title('Filtered')

%% Question 13
clear, clc, close all

% Load image
im = imread("../BE2_IntroMorphoMath/Diplo1.gif") * 255;

[w, h] = size(im);
skel = uint8(zeros(w, h));

figure, sgtitle('Skeletonisation with two different approaches')
subplot(1,3,1), imshow(im), title('Original')


% First erosion + opening
se = strel('disk', 1);
iterations = 25;
ime = im;
for i=0:iterations-1
    ime = imerode(ime, se);
    imo = imopen(ime, se);
    ims = ime - imo;
    skel = skel + ims;
    %subplot(iterations,4,i*4+1), imshow(ime);
    %subplot(iterations,4,i*4+2), imshow(imo);
    %subplot(iterations,4,i*4+3), imshow(ims);
    %subplot(iterations,4,i*4+4), imshow(skel);
end

subplot(1,3,2), imshow(skel), title('First erosion then opening')

% First opening then erosion
function skel = imskel(im, se, iterations)
    [w, h] = size(im);
    skel = uint8(zeros(w, h));
    ime = im;
    for i=0:iterations-1
        imo = imopen(ime, se);
        ims = ime-imo;
        skel = skel+ims;
        ime = imerode(ime, se);
    end
end

se = strel('disk', 1);
skel = imskel(im, se, 25);
subplot(1,3,3), imshow(skel), title('Firsts opening then erosion')

%% Question 14
clc; clear; close all;
im1 = imread('../BE2_IntroMorphoMath/Images1.jpg');
im = imcomplement(im1);
se = strel('disk', 3);
skel = imskel(im, se, 5);

figure, sgtitle({'Binary Skeletonization by first inverting image'})
subplot(1,3,1), imshow(im1), title('Original')
subplot(1,3,2), imshow(im), title('Complemented')
subplot(1,3,3), imshow(skel), title('Skeletonized')

%% Question 15
clc; clear; close all;

im1 = imread('../BE2_IntroMorphoMath/bloodBW.png');
im = im1(:,:,1);
se = strel('disk', 3);
skel = imskel(im, se, 50);


figure
subplot(2,3,1), imshow(im1)
subplot(2,3,2), imshow(im)
subplot(2,3,3), imshow(im), hold on;
spy(skel, 'r'), hold off;

imreal = imread('../BE2_IntroMorphoMath/blood1.png');
im = imreal(:,:,1);
im = im2uint8(imbinarize(im));
se = strel('disk', 3);
skel = imskel(im, se, 50);

subplot(2,3,4), imshow(imreal)
subplot(2,3,5), imshow(im)
subplot(2,3,6), imshow(im), hold on;
spy(skel, 'r'), hold off;

% Still need to do circle erosion to match the skeletonisation 
% to the actual blood cells... see above for examples?