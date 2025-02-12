%% Neural Networks
clear, clc, close all 

% load image
im = imread('BE1_IntroComputerVision\SpainBeach.jpg');
[height, width, channels] = size(im);
figure, imshow(im), title('Original image')

im_train_beach = im(1:100, 1001:width, :);
im_train_other = im(501:height, 1:256, :);
imwrite(im_train_beach, 'training_set_beach.png', 'png');
imwrite(im_train_other, 'training_set_other.png', 'png');

figure
subplot(1,2,1), imshow(im_train_beach), title('Training set beach')
subplot(1,2,2), imshow(im_train_other), title('Training set other')

% Prepare the data and labels
data = [];
target = [];

data = reshape(im_train_beach, [], 3);
n = size(data, 1);
target = ones(n, 1);

data = [data; reshape(im_train_other, [], 3)];
m = size(data, 1);
target = [target; zeros(m-n, 1)];

% Train the network
net = feedforwardnet([5 5]);
net = train(net, double(data)', target');

% Apply the neural network to the image
im_data = reshape(im, [], 3);

% Run the original image through the network to classify it
output = net(double(im_data)');
classified_image = reshape(output', height, width);  
classified_image = classified_image > 0.5;  % Convert probabilities to binary values (0 or 1)

% Post-process to remove noise
se = strel('disk', 3);
mask = imopen(classified_image, se);

% Display results
figure
subplot(1,2,1), imshow(im .* uint8(classified_image)), title('Raw output')
subplot(1,2,2), imshow(im .* uint8(mask)), title('De-noised output');
imwrite(im .* uint8(classified_image), 'nn_classified_beach.png', 'png');
imwrite(im .* uint8(mask), 'nn_masked_beach.png', 'png');
