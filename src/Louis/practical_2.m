%% Question 2

im_1 = imread('bloodBW.png');
%im_1 = imcomplement(im_1); % invert the colors, else erosion becomes dilation
im_2 = imread('cameraman.tif');

function question_2(im, structural_element)
    im_eroded = imerode(im, structural_element);
    im_dilated = imdilate(im, structural_element);
    
    % plot images
    figure
    subplot(1, 3, 1), imshow(im), title('Original');
    subplot(1, 3, 2), imshow(im_eroded), title('Eroded');
    subplot(1, 3, 3), imshow(im_dilated), title('Dilated');
end

se_1 = strel('rectangle', [10 20]);
se_2 = strel('disk', 10);
se_3 = strel('diamond', 10);
se_4 = strel('line', 10, 0);
question_2(im_1, se_1)
question_2(im_1, se_2)
question_2(im_1, se_3)
question_2(im_1, se_4)
% question_2(im_2, se_1)
% question_2(im_2, se_2)
% question_2(im_2, se_3)
% question_2(im_2, se_4)

%% Question 3

clear
clc

im = imread('EuropeBW.bmp');
im = imcomplement(im);

se = strel('disk', 1);
edges_internal = imcomplement(im - imerode(im, se));
edges_external = imcomplement(imdilate(im, se) - im);

figure
subplot(1, 3, 1), imshow(im), title('Original')
subplot(1, 3, 2), imshow(edges_internal), title('Internal edges')
subplot(1, 3, 3), imshow(edges_external), title('External edges')


%% Question 4
clear;
clc

im = imread('EuropeBW.bmp');
im = imcomplement(im);

n_erosions = 25;
[width, height] = size(im);
stack = zeros(width, height, n_erosions+1, 'logical');

stack(:, :, 1) = im;
structural_element = strel('disk', 7);
for i=1:n_erosions
    stack(:, :, i+1) = imerode(stack(:, :, i), structural_element);
end

figure
collapsed = sum(stack, 3);
imagesc(collapsed)
c = colorbar;
c.Label.String = 'Distance from the sea in number of erosions';

%% Question 5
clear, clc;

im = imread('Images2.jpg');
im = imcomplement(im);

figure
subplot(1, 4, 1)
imshow(im)

se = strel('rectangle', [20 56]);
ime = imerode(im, se);
ime = imdilate(ime, se);
subplot(1, 4, 2), imshow(ime), title('Direct filter of large object')

se = strel('disk', 13);
ime = imerode(im, se);
subplot(1, 4, 3), imshow(ime), title('Filtered large objects')

ime = im - ime;
subplot(1, 4, 3), imshow(ime), title('Filtered large objects')


se = strel('rectangle', [11 35]);
ime = imerode(ime, se);
ime = imdilate(ime, se);
subplot(1, 4, 4), imshow(ime), title('Filtered smaller object')

%% Question 6

clear
clc

function ime = opening(im, strel)
    ime = imerode(im, strel);
    ime = imdilate(ime, strel);
end

function ime = closing(im, strel)
    ime = imdilate(im, strel);
    ime = imerode(ime, strel);
end

im = imread('Images1.jpg');
im = imcomplement(im);

se = strel('disk', 15);
closed = imclose(im, se);
opened = imopen(im, se);

figure
subplot(1, 3, 1), imshow(im), title('Original')
subplot(1, 3, 2), imshow(closed), title('Closed')
subplot(1, 3, 3), imshow(opened), title('Opened')

%% Question 7

clear, clc

im = imread('Images2.jpg');
im = imcomplement(im);

iterations = 5;

figure
subplot(1, iterations, 1), imshow(im), title('original')

se = strel('rectangle', size(im));
ime = imerode(im, se);
for i=2:iterations
    se = strel('rectangle', size(im) / 10);
    ime = imerode(im, se);
    subplot(1, iterations, i), imshow(ime)
end

%% Question 9

clear, clc

im = imread('Nebuleuse.jpg');

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
clear, clc
RGB = imread("SpainBeach.jpg");
HSV = rgb2hsv(RGB);
HSV_filtered = HSV;
mask = HSV_filtered(:,:,1) > 0.05 & HSV_filtered(:,:,1) < 0.06;

se = strel('disk', 2);
mask = imopen(mask, se);
se = strel('disk', 25);
mask = imdilate(mask, se);
imshow(mask)

im_filtered = RGB .* uint8(mask);
imshow(im_filtered)

%% Question 13

clear, clc, close all

% Load image
im = imread("Diplo1.gif") * 255;
%figure
%imshow(im)

[w, h] = size(im);
skel = uint8(zeros(w, h));

%% First erosion + opening
se = strel('disk', 1);
iterations = 25;
ime = im;
figure
for i=0:iterations-1
    ime = imerode(ime, se);
    imo = imopen(ime, se);
    ims = ime - imo;
    skel = skel + ims;
    subplot(iterations,4,i*4+1), imshow(ime);
    subplot(iterations,4,i*4+2), imshow(imo);
    subplot(iterations,4,i*4+3), imshow(ims);
    subplot(iterations,4,i*4+4), imshow(skel);
end

figure, imshow(skel)
figure, imshow(im)

%% First opening then erosion

function skel = imskel(im, se, iterations)
    %iterations = 25;
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
figure, imshow(skel);

%% Question 14
im = imread('Images1.jpg');
im = imcomplement(im);
figure, imshow(im);
se = strel('disk', 3);
skel = imskel(im, se, 5);
figure, imshow(skel)

%% Question 15
im = imread('bloodBW.png');
im = im(:,:,1);
figure, imshow(im);
se = strel('disk', 3);
skel = imskel(im, se, 50);
figure, imshow(im), hold on;
spy(skel, 'r');