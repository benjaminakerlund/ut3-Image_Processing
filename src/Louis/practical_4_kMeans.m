%% k-means clustering
clear, clc, close all

% Load the image
im = imread('BE1_IntroComputerVision\SpainBeach.jpg');
figure, imshow(im), title('Original image')

% Prepare the data
imRGB = imresize(im, [NaN 250]);
imHSV = rgb2hsv(imRGB);

function imClusteredRGB = kMeans(imRGB, k)
    imHSV = rgb2hsv(imRGB);
    [height, width, channels] = size(imHSV);
    dataRGB = reshape(imRGB, height*width, channels);
    dataHSV = reshape(imHSV, height*width, channels);
    
    % Initialize k-means
    randId = randi(size(dataHSV, 1), k, 1);
    centroids = dataHSV(randId, :);
    
    nPixels = height*width;
    maxIter = 1000;
    
    for i=1:maxIter
        
        % Calculate the euclidean distance to each cluster
        distances = zeros(nPixels, k);
        for j=1:k
            diff = dataHSV - centroids(j, :);
            distances(:, j) = sum(diff.^2, 2);
        end
    
        % Assign cluster with the minimum distance
        [~, clusterId] = min(distances, [], 2);
    
        % Compute new centroids
        new_centroids = zeros(k, channels);
        for j=1:k
            clusterData = dataHSV(clusterId == j, :);
            if isempty(clusterData)
                new_centroids(j, :) = dataHSV(randi(size(dataHSV, 1)), :);
            else
                new_centroids(j, :) = mean(clusterData, 1);
            end
        end

        % Check convergence
        if isequal(new_centroids, centroids)
            disp('Converged!')
            dataClustered = centroids(clusterId, :);
            imClusteredHSV = reshape(dataClustered, [height, width, channels]);
            imClusteredRGB = hsv2rgb(imClusteredHSV);
            break;
        end
        centroids = new_centroids;
    end
end

% Display the image according to the clusters
imk3 = kMeans(im, 3);
imk4 = kMeans(im, 4);
imk5 = kMeans(im, 5);
imk6 = kMeans(im, 6);
imk7 = kMeans(im, 7);
imk8 = kMeans(im, 8);

figure
subplot(2,3,1), imshow(imk3), title('k = 3')
subplot(2,3,2), imshow(imk4), title('k = 4')
subplot(2,3,3), imshow(imk5), title('k = 5')
subplot(2,3,4), imshow(imk6), title('k = 6')
subplot(2,3,5), imshow(imk7), title('k = 7')
subplot(2,3,6), imshow(imk8), title('k = 8')