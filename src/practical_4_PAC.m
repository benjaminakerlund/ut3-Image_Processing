%% Principal Component Analysis
clear, clc, close all

im = imread('BE3_ShapeDetection\Images\lunar-eclipse-sep-28-2015-michelle-wood-1.jpg');
%im = imresize(im, [NaN 250]);

[h, w, c] = size(im);
data = double(reshape(im, h * w, c));

% Center the data around the mean
mean_data = mean(data);
data_centered = data - mean_data ;

% Calculate covariance of the data and check eigenvalues 
% The eigenvectors with the highest eigenvalues are the principal axes
covMat = cov(data_centered);
[eigVec, eigVal] = eig(covMat);

% identify max
[~, maxIdx] = max(diag(eigVal));
direction = eigVec(:, maxIdx);

% sort the values 
[eigValSorted, sortIdx] = sort(diag(eigVal), 'descend');
eigVecSorted = eigVec(:, sortIdx);

topComponents = eigVecSorted(:, 1:2);

% project the data onto the top components
compressedData = data_centered * topComponents;

% reconstruct
reconstructedData = (compressedData * topComponents') + mean_data;
im_reconstructed = uint8(reshape(reconstructedData, h, w, c));

% Plotting time
figure, imshow(im), title('Original image')
figure, scatter3(data(:,1), data(:,2), data(:,3)), title('RGB original')
figure, scatter3(reconstructedData(:,1), reconstructedData(:,2), reconstructedData(:,3)), title('RGB Reconstructed')
figure, imshow(im_reconstructed), title('Reconstructed image')

imwrite(im, 'PCA_im_original.png', 'png')
imwrite(im_reconstructed, 'PCA_im_reconstructed.png', 'png')