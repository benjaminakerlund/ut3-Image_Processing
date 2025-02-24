clear all;

%% Question 1: Image I/O 

function show_images(extension) 
    file_list = dir(extension);
    [file_number, tmp] = size(file_list);

    for i = 1:file_number 
        im = imread(file_list(i).name);
        figure;
        imshow(im);
    end
end

% im_1 = imread('cargo.jpg');
% im_2 = imread('Circle.png');
% im_3 = imread('imagex.bmp');

%show_images('*.png')
%show_images('*.jpg')
%show_images('*.bmp')

%% Question 2: Greyscale

im = ones(50, 100);
for i = 1:50
    for j = 1:100
        im(i,j) = im(i,j) * j/100;
    end
end

im(22:28,:) = ones(7, 100) * 0.5;
imshow(im)

%% Question 3: Geometric shapes and patterns

height = 512;
width = 1024;
stripe_width = 160;

function im_striped = vertical_stripes(width, height, stripe_width)
    im_striped = ones(height, width);
    for i = 1:2*stripe_width:width
        % avoid the image from expanding
        end_index = i+stripe_width-1;
        if end_index > width
            stripe_width = stripe_width - (end_index - width);
            end_index = width;
        end
        % draw the stripes
        im_striped(:,i:end_index) = zeros(height, stripe_width);
    end
end

imshow(vertical_stripes(width, height, stripe_width));

function im_striped = horizontal_stripes(width, height, stripe_width)
    im_striped = ones(height, width);
    for i = 1:2*stripe_width:height
        % avoid the image from expanding
        end_index = i+stripe_width-1;
        if end_index > height
            stripe_width = stripe_width - (end_index - height);
            end_index = height;
        end
        % draw the stripes
        im_striped(i:end_index,:) = zeros(stripe_width, width);
    end
end

figure
imshow(horizontal_stripes(width, height, stripe_width));

%% rectangle
function image = rectangle(width, height, rect_width, rect_height)
    image = zeros(height, width);
    rect = ones(rect_height, rect_width);
    start_height = floor((height - rect_height) / 2) + 1;
    start_width = floor((width - rect_width) /  2) + 1;

    image(start_height:start_height+rect_height-1, start_width:start_width+rect_width-1) = rect;
end

imshow(rectangle(1024, 512, 128, 64))

%% Circle
function image = circle(width, height, radius)
    image = zeros(height, width);

    circle = zeros(2*radius, 2*radius);
    for i = 1:2*radius
        for j = 1:2*radius
            if (i - radius).^2 + (j - radius).^2 <= radius.^2
                circle(i, j) = 1;
            end
        end
    end
    
    start_height = floor((height - 2 * radius) / 2) + 1;
    start_width = floor((width - 2 * radius) /  2) + 1;
    image(start_height:start_height+2*radius-1, start_width:start_width+2*radius-1) = circle;
end

imshow(circle(512, 512, 64))

%% Question 4: RGB colour coding
function r_image = filter_r(image)
    r_image = image;
    r_image(:, :, 2:3) = 0; 
end
function g_image = filter_g(image)
    g_image = image;
    g_image(:, :, [1, 3]) = 0; 
end
function b_image = filter_b(image)
    b_image = image;
    b_image(:, :, 1:2) = 0; 
end

image_names = {"Teinte.jpg", "oeil.jpg", "cargo.jpg", "CoulAdd.jpg"};
figure
for j =1:length(image_names)
    image = imread(image_names{j});
    images = {image, filter_r(image), filter_g(image), filter_b(image)};
    for i = 1:4
        subplot(4,4,i+4*(j-1))
        imshow(images{i})
    end
end

%% Question 5: French flag
flag = zeros(200, 300, 3);
flag(:, 1:100, 3) = 1;
flag(:, 101:200, :) = 1;
flag(:, 201:301, 1) = 1;

imshow(flag)

%% Question 6: HSV colour coding
clc, clear all
width = 256; 
height = 256;
image = zeros(height, width, 3);
H = linspace(0, 1, width);
S = linspace(0, 1, height);
for i = 1:height
    image(i, :, 1) = H;
end
for i = 1:width
    image(:, i, 2) = S;
end
image(:,:,3) = 1;

imshow(hsv2rgb(image))

%% Question 7: RGB to grayscale conversion


%% Question 8: Isolation of beach
clc
RGB = imread("SpainBeach.jpg");
HSV = rgb2hsv(RGB);
HSV_filtered = HSV;
mask = HSV_filtered(:,:,1) > 0.05 & HSV_filtered(:,:,1) < 0.06;
%HSV_filtered(mask)
subplot(2,1,1)
imshow(mask)
subplot(2,1,2)
imshow(RGB)

RGB_filtered = hsv2rgb(HSV_filtered);
% imshow(RGB_filtered);

%% Question 9: Histograms

%% Question 10: Mysterious .bmp files

%% Question 11: Blur and Edge filtering

%% Question 12: Isolation of stars