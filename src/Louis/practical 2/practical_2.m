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
