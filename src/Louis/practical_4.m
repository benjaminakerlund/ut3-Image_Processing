% Classification Assignment
clear, close all, clc

%%
clear, close all, clc

im = imread('BE1_IntroComputerVision\SpainBeach.jpg');
imshow(im)

%% Define ('Label') data 
[height, width, channels] = size(im);
scale = 7;
im_downsampled = im(1:scale:height, 1:scale:width, :);
[height, width, channels] = size(im_downsampled);
overlay = im_downsampled;
pixel = [10, 150];
pixels_beach = im_downsampled;% zeros(height, width);
pixels_land = im_downsampled;
pixels_ocean = im_downsampled;
pixels_foam = im_downsampled;

data_beach = [];
data_land = [];
data_ocean = [];
data_foam = [];
for i=1:18
    for j=145:width
        pixels_beach(i, j, :) = [255, 255, 0];
        data_beach = [data_beach im_downsampled(i, j, :)];
    end
end
for i=1:15
    for j=1:10
        pixels_land(i, j, :) = [255, 255, 0];
        data_land = [data_land im_downsampled(i, j, :)];
    end
end
for i=65:70
    for j=5:15
        pixels_land(i, j, :) = [255, 255, 0];
        data_land = [data_land im_downsampled(i, j, :)];
    end
end
for i=90:110
    for j=100:155
        pixels_ocean(i, j, :) = [255, 255, 0];
        data_ocean = [data_ocean im_downsampled(i, j, :)];
    end
end
for i=27:31
    for j=140:158
        pixels_ocean(i, j, :) = [255, 255, 0];
        data_ocean = [data_ocean im_downsampled(i, j, :)];
    end
end
for i=111:121
    for j=42:44
        pixels_foam(i, j, :) = [255, 255, 0];
        data_foam = [data_foam im_downsampled(i, j, :)];
    end
end
for i=34:37
    for j=160:164
        pixels_foam(i, j, :) = [255, 255, 0];
        data_foam = [data_foam im_downsampled(i, j, :)];
    end
end
for i=34:41
    for j=160:168
        pixels_foam(i, j, :) = [255, 255, 0];
        data_foam = [data_foam im_downsampled(i, j, :)];
    end
end
data_beach = squeeze(data_beach);
data_land = squeeze(data_land);
data_ocean = squeeze(data_ocean);
data_foam = squeeze(data_foam);
figure
subplot(1,4,1),  imagesc(pixels_beach), set(gca,'dataAspectRatio',[1 1 1])
subplot(1,4,2), imagesc(pixels_land), set(gca,'dataAspectRatio',[1 1 1])
subplot(1,4,3), imagesc(pixels_ocean), set(gca,'dataAspectRatio',[1 1 1])
subplot(1,4,4), imagesc(pixels_foam), set(gca,'dataAspectRatio',[1 1 1])
figure
subplot(2,1,1), imagesc(pixels_foam), set(gca,'dataAspectRatio',[1 1 1])
subplot(2,1,2), imagesc(im), set(gca,'dataAspectRatio',[1 1 1])

%% Apply the distance calculation to the entire image
[s_beach, mu_beach] = std(double(data_beach));
[s_land, mu_land] = std(double(data_land));
[s_ocean, mu_ocean] = std(double(data_ocean));
[s_foam, mu_foam] = std(double(data_foam));


ima = im;
[height, width, channels] = size(ima);
im_beach = zeros(height, width);
im_land = zeros(height, width);
im_ocean = zeros(height, width);
im_foam = zeros(height, width);

for i=1:height
    for j=1:width
        r = squeeze(double(ima(i, j, :)));
        a_beach = squeeze(abs(r - mu_beach(:)) ./ s_beach(:));
        a_land = squeeze(abs(r - mu_land(:)) ./ s_land(:));
        a_ocean = squeeze(abs(r - mu_ocean(:)) ./ s_ocean(:));
        a_foam = squeeze(abs(r - mu_foam(:)) ./ s_foam(:));
        d_beach = norm(a_beach);
        d_land = norm(a_land);
        d_ocean = norm(a_ocean);
        d_foam = norm(a_foam);
        d = [d_beach d_land d_ocean d_foam];
        [M, I] = min(d);
        if I == 1
            im_beach(i, j) = 1;
        end
        if I==2
            im_land(i, j) = 1;
        end
        if I==3
            im_ocean(i, j) = 1;
        end
        if I==4 
            im_foam(i, j) = 1;
        end
    end
end

%% Plotting go Brrrrr
close all
figure, 
subplot(2,3,1), imshow(im_beach), title('Beach');
subplot(2,3,2), imshow(im_land), title('Land');
subplot(2,3,3), imshow(im_ocean), title('Ocean');
subplot(2,3,4), imshow(im_foam), title('Foam');
subplot(2,3,5), imshow(ima), title('Original');

se = strel('disk', 6);
imf = imerode(im_beach, se);
se = strel('disk', 30);
imf = imdilate(imf, se);
imm = im .* uint8(imf);
figure, imshow(imm)

%% K-NN with K=1 
clear, clc, close all
%%
im = imread('BE1_IntroComputerVision\SpainBeach.jpg');
[height, width, channels] = size(im);

imshow(im)

%%
figure, imshow(im), hold on
rectangle('Position', [1027, 1, width-1027, 113], 'EdgeColor', 'r', 'LineWidth', 2), hold on

data_beach = im(1:113, 1027:width, :);
pix_beach = [];
n_pts = 10;
indices = [randi(size(data_beach, 1)-1, n_pts, 1) ];
indices = [indices randi(size(data_beach, 2)-1, n_pts, 1)];
indices(:, 2) = indices(:, 2) + 1026;
for i=1:n_pts
    h = indices(i, 1);
    w = indices(i, 2);
    new_pix = double(squeeze(im(h, w, :)));
    pix_beach = [pix_beach new_pix];
end
scatter(indices(:, 2), indices(:, 1), 250, '.r')

rectangle('Position', [1, 1, 300, 600], 'EdgeColor', 'r', 'LineWidth', 2)
data_land = im(1:600, 1:300, :);
pix_land = [];
indices = [randi(size(data_land, 1)-1, n_pts, 1) ];
indices = [indices randi(size(data_land, 2)-1, n_pts, 1)];
for i=1:n_pts
    h = indices(i, 1);
    w = indices(i, 2);
    new_pix = double(squeeze(im(h, w, :)));
    pix_land = [pix_land new_pix];
end
scatter(indices(:, 2), indices(:, 1), 250, '.r')

rectangle('Position', [width-49, height-49, 49, 49], 'EdgeColor', 'r', 'LineWidth', 2)
data_ocean = im(height-49:height, width-49:width, :);
pix_ocean = [];
indices = [randi(size(data_ocean, 1)-1, n_pts, 1) ];
indices = [indices randi(size(data_ocean, 2)-1, n_pts, 1)];
indices(:, 1) = indices(:, 1) + height-49;
indices(:, 2) = indices(:, 2) + width-49;
for i=1:n_pts
    h = indices(i, 1);
    w = indices(i, 2);
    new_pix = double(squeeze(im(h, w, :)));
    pix_ocean = [pix_ocean new_pix];
end
scatter(indices(:, 2), indices(:, 1), 250, '.r')

rectangle('Position', [790, 330, 10, 10], 'EdgeColor', 'r', 'LineWidth', 2)
data_foam = im(330:340, 790:800, :);
pix_foam = [];
indices = [randi(size(data_foam, 1)-1, n_pts, 1) ];
indices = [indices randi(size(data_foam, 2)-1, n_pts, 1)];
indices(:, 1) = indices(:, 1) + 330;
indices(:, 2) = indices(:, 2) + 790;
for i=1:n_pts
    h = indices(i, 1);
    w = indices(i, 2);
    new_pix = double(squeeze(im(h, w, :)));
    pix_foam = [pix_foam new_pix];
end
scatter(indices(:, 2), indices(:, 1), 250, '.r')

saveas(gcf, 'kNN_training_sets.png', 'png')

%%
im = double(im);

im_beach = zeros(height, width);
im_land = zeros(height, width);
im_ocean = zeros(height, width);
im_foam = zeros(height, width);
for i=1:height
    for j=1:width
        diff_beach = sum((squeeze(im(i, j, :)) - pix_beach(1:5)).^2, 2);
        diff_land  = sum((squeeze(im(i, j, :)) - pix_land(1:5)).^2);
        diff_ocean = sum((squeeze(im(i, j, :)) - pix_ocean(1:5)).^2);
        diff_foam = sum((squeeze(im(i, j, :)) - pix_foam(1:5)).^2);
        d_beach = norm(sqrt(diff_beach));
        d_land  = norm(sqrt(diff_land));
        d_ocean = norm(sqrt(diff_ocean));
        d_foam = norm(sqrt(diff_foam));
        d = [d_beach d_land d_ocean,d_foam];
        [M, I] = min(d);
        if I==1
            im_beach(i, j) = 1;
        end
        if I==2
            im_land(i, j) = 1;
        end
        if I==3
            im_ocean(i, j) = 1;
        end
        if I==4
            im_foam(i, j) = 1;
        end
    end
end

%%
figure, 
subplot(2,2,1), imshow(im_beach), title('Beach')
subplot(2,2,2), imshow(im_land), title('Land')
subplot(2,2,3), imshow(im_ocean), title('Ocean')
subplot(2,2,4), imshow(im_foam), title('Foam')

se = strel('disk', 10);
ime_beach = imopen(im_beach, se);

figure
subplot(1,2,1), imshow(im .* uint8(im_beach))
subplot(1,2,2), imshow(im .* uint8(ime_beach))
imwrite(im .* uint8(im_beach), 'kNN_classified_beach.png', 'png')
imwrite(im .* uint8(ime_beach), 'kNN_masked_beach.png', 'png')

%% Neural Networks
clear, clc, close all 
%%
% load image
im = imread('BE1_IntroComputerVision\SpainBeach.jpg');
[height, width, channels] = size(im);
imshow(im)

im_train_beach = im(1:100, 1001:width, :);
im_train_other = im(501:height, 1:256, :);
imwrite(im_train_beach, 'training_set_beach.png', 'png');
imwrite(im_train_other, 'training_set_other.png', 'png');
%imshow(im_beach)
%imshow(im_other)

%% Prepare the data and labels
data = [];
target = [];

data = reshape(im_train_beach, [], 3);
n = size(data, 1);
target = ones(n, 1);

data = [data; reshape(im_train_other, [], 3)];
m = size(data, 1);
target = [target; zeros(m-n, 1)];

%% Train the network
net = feedforwardnet([5 5]);
net = train(net, double(data)', target');

%% Apply the neural network to the image
im_data = reshape(im, [], 3);
output = net(double(im_data)');
classified_image = reshape(output', height, width);  
classified_image = classified_image > 0.5;  % Convert probabilities to binary values (0 or 1)
se = strel('disk', 3);
mask = imopen(classified_image, se);

figure;
imshow(im .* uint8(mask))% uint8(classified_image));
title('Classified Image');
imwrite(im .* uint8(classified_image), 'nn_classified_beach.png', 'png');
imwrite(im .* uint8(mask), 'nn_masked_beach.png', 'png');

%% k-means
clc, clear, close all

% data preparation
im = imread('BE1_IntroComputerVision\SpainBeach.jpg');
imRGB = imresize(im, [NaN 64]);
imHSV = rgb2hsv(imRGB);
figure, imagesc(imRGB)
figure, scatter(imHSV(:,:,1), imHSV(:,:,3), '.r')
[h, w, c] = size(imHSV);
%data = [reshape(imHSV(:,:,1), [h*w 1]) reshape(imHSV(:,:,3), [h*w 1])];

%%
% algorithm time
k = 4; % number of classes
centroids = rand(k, 3); % randomly generate centroids in HV

for kk=1:100
    cluster1 = [];
    cluster2 = [];
    cluster3 = [];
    cluster4 = [];

    for i=1:h
        for j=1:w

            % identify classes in image
            distances = zeros(k, 1);
            for ii=1:k
                distances(ii) = norm(squeeze(imHSV(i, j, :)) - centroids(ii,:), 2);
                [M, I] = min(distances); 
                if I==1
                    cluster1 = [cluster1 squeeze(imHSV(i, j, :))];
                end
                if I==2
                    cluster2 = [cluster2 squeeze(imHSV(i, j, :))];
                end
                if I==3
                    cluster3 = [cluster3 squeeze(imHSV(i, j, :))];
                end
                if I==4
                    cluster4 = [cluster4 squeeze(imHSV(i, j, :))];
                end
            end

            
        end
    end

    % compute new centroids
    centroids = [
        mean(cluster1, 2)'
        mean(cluster2, 2)'
        mean(cluster3, 2)'
        mean(cluster4, 2)'
    ];
    convergence = 0;
    
end

%%
figure, scatter3(cluster1(1,:), cluster1(2,:), cluster1(3,:))
figure, scatter3(cluster2(1,:), cluster2(2,:), cluster2(3,:))
figure, scatter3(cluster3(1,:), cluster3(2,:), cluster3(3,:))
figure, scatter3(cluster4(1,:), cluster4(2,:), cluster4(3,:))

