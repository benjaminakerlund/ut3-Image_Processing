%% Question 1: Image I/O 
clc, clear, close all

function show_images(extension) 
    file_list = dir(extension);
    [file_number, tmp] = size(file_list);

    for i = 1:file_number 
        im = imread(file_list(i).name);
        figure;
        imshow(im);
    end
end

show_images('*.png')
show_images('*.jpg')
show_images('*.bmp')

%% Question 2: Grayscale
clc, clear, close all

im = ones(50, 100);
for i = 1:50
    for j = 1:100
        im(i,j) = im(i,j) * j/100;
    end
end

im(22:28,:) = ones(7, 100) * 0.5;
figure, imshow(im), exportgraphics(gcf, '../Q2.png')

%% Question 3: Geometric shapes and patterns
clc, clear, close all

% vertical stripes
function im = im_vstripes(width, height, stripe_width)
    white = ones(height, stripe_width);
    black = zeros(height, stripe_width);
    stripe = [black white];
    n = ceil(width / (2*stripe_width));
    im = repmat(stripe, 1, n);
    im = im(:, 1:width); % crop the image to the right width
end

% horizontal stripes
function im = im_hstripes(width, height, stripe_height)
    white = ones(stripe_height, width);
    black = zeros(stripe_height, width);
    stripe = [black; white];
    n = ceil(height / (2*stripe_height));
    im = repmat(stripe, n, 1);
    im = im(1:height,:);
end

% rectangle
function im = im_rect(width, height, rect_width, rect_height)
    im = zeros(height, width);
    rect = ones(rect_height, rect_width);

    start_height = floor((height - rect_height) / 2) + 1;
    start_width = floor((width - rect_width) /  2) + 1;
    im(start_height:start_height+rect_height-1, start_width:start_width+rect_width-1) = rect;
end

% circle
function im = im_circ(width, height, radius)
    im = zeros(height, width);
    circ = zeros(2*radius);
    for i = 1:2*radius
        for j = 1:2*radius
            if (i - radius).^2 + (j - radius).^2 <= radius.^2
                circ(i, j) = 1;
            end
        end
    end
    start_height = floor((height - 2 * radius) / 2) + 1;
    start_width = floor((width - 2 * radius) /  2) + 1;
    im(start_height:start_height+2*radius-1, start_width:start_width+2*radius-1) = circ;
end

imv = im_vstripes(512, 256, 31);
imh = im_hstripes(512, 256, 31);
imr = im_rect(384, 256, 64, 32);
imc = im_circ(384, 256, 32);

figure
subplot(2,2,1), imshow(imv), exportgraphics(gca, '../Q3_vstripes.png')
subplot(2,2,2), imshow(imh), exportgraphics(gca, '../Q3_hstripes.png')
subplot(2,2,3), imshow(imr), exportgraphics(gca, '../Q3_rect.png')
subplot(2,2,4), imshow(imc), exportgraphics(gca, '../Q3_circ.png')

%% Question 4: RGB colour coding
clc, clear, close all

function [r, g, b] = split_colors(im)
    r = zeros(size(im), 'double');
    g = zeros(size(im), 'double');
    b = zeros(size(im), 'double');

    r(:, :, 1) = im(:, :, 1);
    g(:, :, 2) = im(:, :, 2);
    b(:, :, 3) = im(:, :, 3);
end


im_names = {"Teinte.jpg", "oeil.jpg", "cargo.jpg", "CoulAdd.jpg"};
figure
for j =1:length(im_names)
    im = imread(im_names{j});
    [imr, img, imb] = split_colors(im);
    ims = {im, imr, img, imb};
    for i = 1:4
        subplot(4,4,i+4*(j-1))
        imshow(uint8(ims{i}))
    end
end

%% Question 5: French flag
clc, clear, close all

flag = zeros(200, 300, 3);
flag(:, 1:100, 3) = 1;
flag(:, 101:200, :) = 1;
flag(:, 201:301, 1) = 1;

figure, imshow(flag), exportgraphics(gcf, '../Q5_Flag.png')

%% Question 6: HSV colour coding
clc, clear, close all

function im = hsv_spectrum(height, width)
    im = zeros(height, width, 3);
    
    H = linspace(0, 1, width);
    S = linspace(0, 1, height);
    
    im(:, :, 1) = repmat(H, height, 1);
    im(:, :, 2) = repmat(S', 1, width);
    im(:,:,3) = 1;
    im = hsv2rgb(im);
end

im = hsv_spectrum(256, 512);
figure, imshow(im), exportgraphics(gcf, '../Q6_HSV_spectrum.png')

%% Question 7: RGB to grayscale conversion
clc, clear, close all

a = 0.299;
b = 0.587;
c = 0.114;

im = hsv_spectrum(256, 512);
[imr, img, imb] = split_colors(im);
img = sum(a.*imr + b.*img + c.*imb, 3);

figure
subplot(1,3,1), imshow(im), title('Original')
subplot(1,3,2), imshow(img), title('Converted'), exportgraphics(gca, '../Q7_grayscale.png')
subplot(1,3,3), imshow(rgb2gray(im)), title('Built-in conversion')

%% Question 8: Isolation of beach
clc, clear, close all

% filter the image in the HSV space based on a range of hues
RGB = imread("SpainBeach.jpg");
HSV = rgb2hsv(RGB);
HSV_filtered = HSV;
mask = HSV_filtered(:,:,1) > 0.05 & HSV_filtered(:,:,1) < 0.06;
RGB_filtered = RGB .* uint8(mask);

figure
subplot(1,2,1), imshow(RGB)
subplot(1,2,2), imshow(RGB_filtered), exportgraphics(gca, '../Q8_FilteredBeach.png')

%% Question 9: Histograms
clc, clear, close all

im = imread('SpainBeach.jpg');
figure, axis('equal')
subplot(1,2,1), imhist(im), title('Default'), exportgraphics(gca, '../Q9_Hist.png')
subplot(1,2,2), imhist(histeq(im)), title('Equalized'), exportgraphics(gca, '../Q9_HistEq.png')


%% Question 10: Mysterious .bmp files
clc, clear, close all

[imx, mapx]  = imread('imagex.bmp');
[imxx, mapxx] = imread('imagexx.bmp');

figure
subplot(1,3,1), imshow(imx), title('Imagex.bmp')
subplot(1,3,2), imshow(imxx), title('Imagexx.bmp as uint8')
subplot(1,3,3), imshow(double(imxx)), title('Imagexx.bmp as double'), exportgraphics(gca, '../Q10_Mystery.png')

%% Question 11: Blur and Edge filtering
clc, clear, close all 

im = im_vstripes(512, 512, 64);
blur_filter = fspecial('gaussian', [10 10], 10);
edge_filter = fspecial('laplacian');
imb = imfilter(im, blur_filter, 'replicate');
ime = imfilter(uint8(im), edge_filter);

figure, imshow(im), title('Original'), saveas(gcf, '../Q11_Stripes_Original.png')
figure, imshow(imb), title('Blurred'), saveas(gcf, '../Q11_Stripes_Blurred.png')
figure, imshow(imadjust(ime)), title('Edges'), saveas(gcf, '../Q11_Stripes_Edges.png')

im = imread('SpainBeach.jpg');
blur_filter = fspecial('gaussian', [10 10], 10);
edge_filter = fspecial('sobel');
imb = imfilter(im, blur_filter, 'replicate');
ime = imfilter(uint8(im), edge_filter);

figure, imshow(im), title('Original'), saveas(gcf, '../Q11_Beach_Original.png')
figure, imshow(imb), title('Blurred'), saveas(gcf, '../Q11_Beach_Blurred.png')
figure, imshow(ime), title('Edges'), saveas(gcf, '../Q11_Beach_Edges.png')

%% Question 12: Isolation of stars
clc, clear, close all

imo = imread('Etoiles.png');
im = rgb2gray(imo);

% blur the image
imb = imfilter(im, fspecial('gaussian', [50 50], 10));
imb = imadjust(imb);

% find edges and blur the result
ime = imfilter(imb, fspecial('laplacian', 0.1));
imeb = imfilter(ime, fspecial('disk', 15));

% laplacian edges are the complement to what we want
mask = uint8(~logical(imeb));
imf = imo .*  mask; 

figure, imshow(imo), title('Original'), saveas(gcf, '../Q12_Original.png')
figure, imshow(imb), title('Blurred'), saveas(gcf, '../Q12_Blurred.png')
figure, imshow(imadjust(ime)), title('Edges'), saveas(gcf, '../Q12_Edges.png')
figure, imshow(imf), title('Masked'), saveas(gcf, '../Q12_Masked.png')

