%% Question 1
clc; clear; close all;

% Load colour image and convert to B&W 
cargo = imread('../BE1_IntroComputerVision/cargo.jpg');
cargo_gray = im2gray(cargo);

% Display images
figure Name 'Part1_Question1' FileName 'Part1_Question1'
sgtitle('Loading and generating a RGB and B&W image')
subplot(1,2,1)
imshow(cargo)
title('RGB image')

subplot(1,2,2)
imshow(cargo_gray)
title('B&W image')

% Data types of read images = 'uint8'
class(cargo);
class(cargo_gray);


%% Question 2: Greyscale Illusion
clc; clear; close all;

A = repmat(linspace(0,1,1000),500,1);
A(220:280,:) = 0.5;
figure Name 'Part1_Question2' FileName 'Part1_Question2'
imshow(A)
title('Grayscale Illusion')
imsave(Question2)


%% Question 3: custom images
clc; clear; close all;

% Stripes
B = repmat(0,500,500);
for i = 1:10
    numb = i*50-25;
    numb2 = numb+25;
    B(:, numb:numb2) = 1;
end

C = repmat(0,500,500);
for i = 1:5
    numb = i*100-50;
    numb2 = numb+50;
    C(:, numb:numb2) = 1;
end

% Color every 2nd row black
D = repmat(1,500,500);
for i = 1:size(D,1)
    %for j = 1:size(C,2)
        %disp(C(i,j))
        %if mod(C(i,j))
    %end
    if mod(i, 2) == 0
        D(:, i) = 0;
    end
end

% Plot Stripes
figure Name 'Part1_Question3a' FileName 'Part1_Question3a'
sgtitle('Stripes')
subplot(2,2,1)
imshow(B)

subplot(2,2,2)
imshow(C)

subplot(2,2,3)
imshow(D)

subplot(2,2,4)
imshow(B')


% Rectangles & Disks
rect_gen = repmat(0,500,500);
%Rect_data(225:275, 200:300) = 1;
rect = insertShape(rect_gen,'filled-rectangle',[200 225 100 50], ...
    ShapeColor = 'white', Opacity = 1);

disk_gen = repmat(0,500,500);
disk = insertShape(disk_gen,'filled-circle',[250 250 50], ...
    ShapeColor = 'white', Opacity = 1);

% Plot Rectangles & Disks
figure Name 'Part1_Question3b' FileName 'Part1_Question3b'
sgtitle('Rectangles & Disks')
subplot(1,2,1)
imshow(rect)
title('Rectangle generated with insertShape')

subplot(1,2,2)
imshow(disk)
title('Circle generated with insertShape')


%% Question 4
clc; clear; close all;

% Load images
teinte = imread('../BE1_IntroComputerVision/Teinte.jpg');
oeil = imread('../BE1_IntroComputerVision/oeil.jpg');
cargo = imread('../BE1_IntroComputerVision/cargo.jpg');
CoulAdd = imread('../BE1_IntroComputerVision/CoulAdd.png');

% Conversions to R, G and B components for teinte
[teinte_R, teinte_G, teinte_B] = deal(teinte);
teinte_R(:, :, 2:3) = 0;
teinte_G(:, :, [1 3]) = 0;
teinte_B(:, :, 1:2) = 0;

% Conversions to R, G and B components for oeil
[oeil_R, oeil_G, oeil_B] = deal(oeil);
oeil_R(:, :, 2:3) = 0;
oeil_G(:, :, [1 3]) = 0;
oeil_B(:, :, 1:2) = 0;

% Conversions to R, G and B components for cargo
[cargo_R, cargo_G, cargo_B] = deal(cargo);
cargo_R(:, :, 2:3) = 0;
cargo_G(:, :, [1 3]) = 0;
cargo_B(:, :, 1:2) = 0;

% Conversions to R, G and B components for CoulAdd
[CoulAdd_R, CoulAdd_G, CoulAdd_B] = deal(CoulAdd);
CoulAdd_R(:, :, 2:3) = 0;       % RGB vector 1 value is RED, put others to 0
CoulAdd_G(:, :, [1 3]) = 0;     % RGB vector 2 value is GREEN, put others to 0
CoulAdd_B(:, :, 1:2) = 0;       % RGB vector 3 value is BLUE, put others to 0

% Plot original images
figure Name 'Part1_Question4a' FileName 'Part1_Question4a'
sgtitle('Comparison of original images and red, green and blue components')

% Plots for teinte
subplot(4,4,1), imshow(teinte), title('teinte')
subplot(4,4,2), imshow(teinte_R)
subplot(4,4,3), imshow(teinte_G) 
subplot(4,4,4), imshow(teinte_B)

% Plots for oeil
subplot(4,4,5), imshow(oeil), title('oeil')
subplot(4,4,6), imshow(oeil_R), 
subplot(4,4,7), imshow(oeil_G)
subplot(4,4,8), imshow(oeil_B)

% Plots for cargo
subplot(4,4,9), imshow(cargo), title('cargo')
subplot(4,4,10), imshow(cargo_R)
subplot(4,4,11), imshow(cargo_G)
subplot(4,4,12), imshow(cargo_B)

% Plots for CoulAdd
subplot(4,4,13), imshow(CoulAdd), title('CoulAdd')
subplot(4,4,14), imshow(CoulAdd_R)
subplot(4,4,15), imshow(CoulAdd_G)
subplot(4,4,16), imshow(CoulAdd_B)


%% Questinon 5
% a) Build and display the french flag. 
% b) Build and display your flag.
clc; clear; close all;

% RGB vector 1 value is RED, put others to 0
% RGB vector 2 value is GREEN, put others to 0
% RGB vector 3 value is BLUE, put others to 0

% Parameters
H = 360;
W = 640;

% Build French flag
FR = zeros(H,W,3);
FR(:, 1:W/3, 3) = 255;          % Blue
FR(:, W/3:2*W/3, 1:3) = 255;    % White
FR(:, 2*W/3:W, 1) = 255;        % Red

% Build Finnish flag
%teinte = imread('../Teinte.jpg')
col = floor(H/5);
row = floor(W/7);
FI = zeros(H,W,3);
FI(:, :, 1:3) = 255;            % White background
FI(2*col:3*col, :, 1:2) = 0;    % Vertical stripe, make all else than blue 0
FI(:, 2*row:3*row, 1:2) = 0;    % Horisontal stripe, make all else than blue 0
% DO STUFF


% A = [1,2;3,4];
% B = [5,6;7,8];
% C = [9,10;11,12];
% Z = cat(3,A,B,C)

% Display flags
figure Name 'Part1_Question5' FileName 'Part1_Question5'
sgtitle('French and Finnish flags')
subplot(1,2,1), imshow(FR), title("French flag")
subplot(1,2,2), imshow(FI), title("Finnish flag")


%% Question 6
% a) Use the HSV code
% b) How is the type of the matrix?
% c) Build and display image in Fig. 4, 

clc; clear; close all;

H = 360;
W = 640;

I_hsv = zeros(H, W, 3);
I_hsv(:,:,3) = 1;

figure Name 'Part1_Question6a' FileName 'Part1_Question6a'
sgtitle('Generate HSV Color Space')
subplot(2,2,1), imshow(I_hsv), title('Blue image')

% Hue
for v = 1 : W
    I_hsv(:, v, 1) = v / W;
end
subplot(2,2,2), imshow(I_hsv), title('Adding horisontal hue')

% Saturation
for h = 1 : H
    I_hsv(h, :, 2) = h / W;
end
subplot(2,2,3), imshow(I_hsv), title('Adding vertical saturation')
subplot(2,2, 4), imshow(hsv2rgb(I_hsv)), title('HSV Color Space')

figure Name 'Part1_Question6b' FileName 'Part1_Question6b'
sgtitle('Grayscale of the HSV colorspace image')
imshow(rgb2gray(hsv2rgb(I_hsv)))

%% Question 7
% What are the values of \alpha, \beta and \gamma?
clc; clear; close all;
im = imread('../BE1_IntroComputerVision/cargo.jpg');
im_double = im2double(im);

alpha = 1.9;
beta = 0.3;
gamma = 2.2;

im_a_adjusted = alpha*im_double;
im_b_adjusted = im_double + beta;
im_ab_adjusted = alpha * im_double + beta;
im_ab_adjusted = min(max(im_ab_adjusted, 0), 1);
im_g_adjusted = im_double .^ gamma;
im_abg_adjusted = im_ab_adjusted .^ gamma;

figure, sgtitle('\alpha, \beta and \gamma correction')
subplot(2,3,1), imshow(im_double), title('original')
subplot(2,3,2), imshow(im_a_adjusted), title(['\alpha increased by \alpha='  num2str(alpha)]);
subplot(2,3,3), imshow(im_b_adjusted), title(['\beta increased by \beta=' num2str(beta)]);
subplot(2,3,4), imshow(im_ab_adjusted), title('\alpha and \beta adjusted')
subplot(2,3,5), imshow(im_g_adjusted), title(['\gamma increased by \gamma =' num2str(gamma)])
subplot(2,3,6), imshow(im_abg_adjusted), title('\alpha, \beta and \gamma adjusted')


%% Question 8
% Load and display SpainBeach.jpg and isolate the beach.
clc; clear; close all;

RGB = imread('../BE1_IntroComputerVision/SpainBeach.jpg');

HSV = rgb2hsv(RGB);
HSV_filtered = HSV;
mask = HSV_filtered(:,:,1) > 0.05 & HSV_filtered(:,:,1) < 0.06;

% Plot the results
figure Name 'Part1_Question8' FileName 'Part1_Question8'
sgtitle('Isolating the beach')
subplot(1,3,1), imshow(RGB), title('Original Image')
subplot(1,3,2), imshow(rgb2hsv(RGB)), title('Filtered Image')
subplot(1,3,3), imshow(mask), title('Isolated Beaches')

%% Question 9
clc; clear; close all;

% Create grayscale image and plot histogram
A = repmat(linspace(0,1,1000),500,1);
A(220:280,:) = 0.5;
figure, sgtitle('Grayscale illusion and histogram')
subplot(1,2,1), imshow(A), title('Grayscale Illusion')
subplot(1,2,2), histogram(A)

%% Question 10: Mysterious .bmp images
clc; clear; close all;
imagex = imread('../BE1_IntroComputerVision/imagex.bmp');
imagexx = imread('../BE1_IntroComputerVision/imagexx.bmp');

% convert to hsv
% imagex_hsv = rgb2hsv(imagex);
% imagexx_hsv = rgb2hsv(imagexx);

% Show R, G and B components?
[imagex_R, imagex_G, imagex_B] = deal(imagex);
[imagexx_R, imagexx_G, imagexx_B] = deal(imagexx);

imagex_R(:, :, 2:3) = 0; 
imagex_G(:, :, [1 3]) = 0; 
imagex_B(:, :, 1:2) = 0;
imagexx_R(:, :, 2:3) = 0; 
imagexx_G(:, :, [1 3]) = 0;
imagexx_B(:, :, 1:2) = 0;

% Plots
figure, sgtitle('Imagex and Imagexx being worked?')
subplot(2,5,1), imshow(imagex), title('Original')
subplot(2,5,2), histogram(imagex), title('Histogram')
subplot(2,5,3), imshow(imagex_R), title('R')
subplot(2,5,4), imshow(imagex_G), title('G')
subplot(2,5,5), imshow(imagex_B), title('B')

subplot(2,5,6), imshow(imagexx), title('Original')
subplot(2,5,7), histogram(imagexx), title('Histogram')
subplot(2,5,8), imshow(imagexx_R), title('R')
subplot(2,5,9), imshow(imagexx_G), title('G')
subplot(2,5,10), imshow(imagexx_B), title('B')

%% Question 11: Filtering and Edge filtering on stripes and real image
clc; clear; close all;


% Parameters
width = 512;
height = 512;
stripe_width = 32;
radius = 32;
rect_w = 64;
rect_h = 32;


% STRIPES
image = horizontal_stripes(width, height, stripe_width);
image_T = image';

h = fspecial("motion", 50, 45);
image_filtered = imfilter(image, h);
image_edge_C = edge(image, "Canny");
image_edge_P = edge(image, "Prewitt");

imT_filtered = imfilter(image_T, h);
imT_edge_C = edge(image_T, "Canny");
imT_edge_P = edge(image_T, "Prewitt");

figure, sgtitle('Blur- and Edge Filtering on stripes')
subplot(2,5,1), imshow(image), title('Original')
subplot(2,5,2), imshow(image_filtered), title('Blur filtered')
subplot(2,5,3), imshow(image_edge_C), title('Canny')
subplot(2,5,4), imshow(image_edge_P), title('Prewitt')
subplot(2,5,5), imshowpair(image_edge_C, image_edge_P), title('Combined techniques')

subplot(2,5,6), imshow(image_T), title('Original')
subplot(2,5,7), imshow(imT_filtered'), title('Blur filtered')
subplot(2,5,8), imshow(imT_edge_C), title('Canny')
subplot(2,5,9), imshow(imT_edge_P), title('Prewitt')
subplot(2,5,10), imshowpair(imT_edge_C, imT_edge_P), title('combined techniques')

% REAL IMAGE
% image =
%figure, sgtitle('Blur- and Edge Filtering on real image')



%% Question 12: Isolation of stars
clc; clear; close all;
imo = imread('../BE1_IntroComputerVision/Etoiles.png');
im = rgb2gray(imo);

% inspect a patch of the image containing noise
patch = imcrop(im); close;

% Compute variance of patch, which approximates variance of the noise
patchvar = std2(patch)^2;

% Define Degree of smoothing
DoS = 4*patchvar;
J = imbilatfilt(im, DoS,25);

figure, sgtitle('Isolating 5 biggest stars with multiple filters')
subplot(1,4,1), imshow(im), title('Original in BW')
subplot(1,4,2), imshow(J), title('Gaussian filter to smooth image')

level = graythresh(J);
J_BW = imbinarize(J, level);
subplot(1,4,3), imshow(J_BW), title("Binarising with adaptive thresholding")

% Median filter to remove salt-and-pepper noise
im_median = medfilt2(J_BW, [25 25]);
subplot(1,4,4), imshow(im_median), title('Median filter to remove salt-and-pepper noise')


figure, sgtitle({'Comparison plot with only', 'binarising and median filtering'})
subplot(1,2,1), imshow(imo), title('Original in RGB')
im = imbinarize(im);
im = medfilt2(im, [30 30]);
subplot(1,2,2), imshow(im), title('Filtered image')


%% Question 13: FT spectrum of geometric shapes and patterns
clc; clear; close all;
% Rectangle function
function image = rectangle(width, height, rect_width, rect_height)
    image = zeros(height, width);
    rect = ones(rect_height, rect_width);
    start_height = floor((height - rect_height) / 2) + 1;
    start_width = floor((width - rect_width) /  2) + 1;

    image(start_height:start_height+rect_height-1, start_width:start_width+rect_width-1) = rect;
end

% Circle function
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

% vertical stripes function
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

% horizontal stripes function
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

% Parameters
width = 512;
height = 512;
stripe_width = 32;
radius = 32;
rect_w = 64;
rect_h = 32;

% Rectangle and Circle
image = rectangle(width, height, rect_w, rect_h);
ft_horizontal_stripes = fft2(image, width, height);
ft_horizontal_stripes = fftshift(ft_horizontal_stripes);
ft_spectrum = abs(ft_horizontal_stripes);

figure Name 'Part1_Question13' FileName 'Part1_Question13'
sgtitle('FT of rectangles and disks')
subplot(2,3,1), imshow(image), title('Original');
subplot(2,3,2), imagesc(ft_spectrum), colorbar, title('FT spectrum');
subplot(2,3,3), imagesc(log(abs(ft_spectrum))), colorbar, title('FT log spectrum');

image = circle(width, height, radius);
ft_horizontal_stripes = fft2(image, width, height);
ft_horizontal_stripes = fftshift(ft_horizontal_stripes);
ft_spectrum = abs(ft_horizontal_stripes);

subplot(2,3,4), imshow(image), title('Original');
subplot(2,3,5), imagesc(ft_spectrum), colorbar, title('FT spectrum');
subplot(2,3,6), imagesc(log(abs(ft_spectrum))), colorbar, title('FT log spectrum');

% Stripes
figure, sgtitle('FT of stripes')
image = horizontal_stripes(width, height, stripe_width);
ft_horizontal_stripes = fft2(image, width, height);
ft_horizontal_stripes = fftshift(ft_horizontal_stripes);
ft_spectrum = abs(ft_horizontal_stripes);

subplot(2,2,1), imshow(image), title('Original');
subplot(2,2,2), imagesc(ft_spectrum), title('FT spectrum');

image = vertical_stripes(width, height, stripe_width);
ft_horizontal_stripes = fft2(image, width, height);
ft_horizontal_stripes = fftshift(ft_horizontal_stripes);
ft_spectrum = abs(ft_horizontal_stripes);

subplot(2,2,3), imshow(image), title('Original');
subplot(2,2,4), imagesc(log(abs(ft_spectrum))), title('FT spectrum');

%% Question 14: FT spectrum of blurred image
clc; clear; close all;

%% Question 15: Extraction of field
clc; clear all; close all;

im = imread('../BE1_IntroComputerVision/champs.bmp');
[width, height, channels] = size(im);

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

ft_filtered = ifftshift(ft_filtered);
im_mask= ifft2(ft_filtered);
im_mask = abs(im_mask);

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


% apply some kind of blur, actually dilation and then erode


% Plot Stripes
figure Name 'Part1_Question15' FileName 'Part1_Question15'
sgtitle('Extract field from Champs.jpg')
subplot(1,3,1)
imshow(im)
title("Original Image")

subplot(1,3,2)
imagesc(spectrum_filtered)
title("Filtered Image")

subplot(1,3,3)
imagesc(uint8(im_filtered))
title("Extracted Field")
