%% Question 1
clc; clear; close all;

% Load colour image and convert to B&W 
cargo = imread('../cargo.jpg');
cargo_gray = im2gray(cargo);

% Display images
figure Name 'Question1' FileName 'Question1'
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
figure Name 'Question2' FileName 'Question2'
imshow(A)
title('Grayscale Illusion')


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

% F = repmat(0,500,500);
% for i = 1:10
%     numb = i*50 -25;
%     numb2 = numb+25;
%     F(numb:numb2,:) = 1;
% end


% Plot Stripes
figure Name 'Question3a' FileName 'Question3a'
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
figure Name 'Question3b' FileName 'Question3b'
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
teinte = imread('../Teinte.jpg');
oeil = imread('../oeil.jpg');
cargo = imread('../cargo.jpg');
CoulAdd = imread('../CoulAdd.png');

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
figure Name 'Question4a' FileName 'Question4a'
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
figure Name 'Question4' FileName 'Question4'
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

figure Name Question6c FileName Question6c
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


%% Question 7
% What are the values of \alpha, \beta and \gamma?


%% Question 8
% Load ans display SpainBeach.jpg and isolate the beach.
clc; clear; close all;

beach = imread('../SpainBeach.jpg');

% DO STUFF

% Plot the results
figure Name Question8 FileName Question8
sgtitle('Isolating a beach')
subplot(1,2,1), imshow(beach), title('RGB')
subplot(1,2,2), imshow(rgb2hsv(beach)), title('HSV')


%% Question 13
clc; clear; close all;

% Stripes
B = repmat(0,500,500);
for i = 1:10
    numb = i*50-25;
    numb2 = numb+25;
    B(:, numb:numb2) = 1;
end


% Plot Stripes
figure Name 'Question13' 
sgtitle('Stripes')
subplot(2,2,1)
imshow(B)
title('Original')

subplot(2,2,2)
title('FFT')

subplot(2,2,3)
imshow(B')
title('Original')

subplot(2,2,4)
title('FFT')
