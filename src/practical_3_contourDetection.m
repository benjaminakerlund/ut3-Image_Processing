%% Contour-based
clear, clc, close all

% Load the images
im_moon            = imread('Images/lunar-eclipse-sep-28-2015-michelle-wood-1.jpg');
im_jupiter_earth   = imread('Images/Jupiter-dreams-meaning.jpg');
im_jupiter_partial = imread('Images/d52ed43e4a_106828_jupiter-pole-sud-juno.jpg');

function [x, y] = getEdges(im)
    im_gray = rgb2gray(im);
    threshold = 0.1;
    im_bw = imbinarize(im_gray, threshold);
    filter = fspecial("laplacian");
    im_filtered = imfilter(im_bw, filter);
    
    % figure
    % subplot(1,3,1), imshow(im_gray), title('Grayscale')
    % subplot(1,3,2), imshow(im_bw), title('Binary')
    % subplot(1,3,3), imshow(im_filtered), title('Edges')
     
    [row, col] = find(im_filtered);
    [h, w] = size(im_filtered);
    x = col;
    y = row; 
    %figure, scatter(x, y, '.'), axis('equal')
end

function findPlanetPseudoInverse(im, saveName)
    [x, y] = getEdges(im);
    [h, w] = size(im, [1 2]);

    % data matrix
    M = [(x.^2 + y.^2), x, y];
    params = pinv(M) * ones(size(M, 1), 1);
    A = params(1);
    B = params(2);
    C = params(3);
    
    center = [-B/(2*A), -C/(2*A)];
    radius = sqrt(1/A + center(1)^2 + center(2)^2);
    
    % reconstruct circle
    angles = linspace(0, 2*pi, 100);
    x_est = radius * cos(angles) + center(1);
    y_est = radius * sin(angles) + center(2);
    
    figure, xlim([0, w]), ylim([0, h]), axis('equal'), hold on 
    scatter(x, y, '.', 'SizeData', 150,'MarkerEdgeAlpha', 0.3)
    plot(x_est, y_est, '-', 'LineWidth', 2)
    axis off
    legend('Edge data', 'Contour fit')
    saveas(gcf, saveName);
end

function results = circleFitRANSAC(im, minPts, threshold, maxIter)
    [xe, ye] = getEdges(im);
    data = [xe ye];
    nPts = size(data, 1);
    disp(nPts)
    
    results = [];
    for i=1:maxIter
        % Select three points which uniquely define a circle
        idx = randperm(nPts, 3);
        xc = data(idx, 1);
        yc = data(idx, 2);
        A = [xc.^2+yc.^2, xc, yc];
        B = ones(3, 1);
        P = linsolve(A, B);
        center = [-P(2) / (2*P(1)), -P(3) / (2*P(1))];
        radius = sqrt(1/P(1) + center(1)^2 + center(2)^2);
        
        % Get the inliers to this initial circle
        diff = data-center;
        distances = vecnorm(diff, 2, 2);
        tolerances = abs(distances - radius);
        idxInliers = find(tolerances < threshold) ;
        nInliers = size(idxInliers, 1);
        if nInliers < minPts
            continue;
        end
        inliers = data(idxInliers, :);
        
        % Apply pseudo-inverse to find the best fit from the inliers
        x = inliers(:, 1);
        y = inliers(:, 2);
        M = [(x.^2 + y.^2), x, y];
        params = pinv(M) * ones(size(M, 1), 1);
        A = params(1);
        B = params(2);
        C = params(3);
        center = [-B/(2*A), -C/(2*A)];
        radius = sqrt(1/A + center(1)^2 + center(2)^2);

        % Get the inliers to this new circle
        diff = data-center;
        distances = vecnorm(diff, 2, 2);
        tolerances = abs(distances - radius);
        idxInliers = find(tolerances < threshold) ;
        nInliers = size(idxInliers, 1);
        if nInliers < minPts
            continue;
        end
        inliers = data(idxInliers, :);

        % calculate radial error from circle by converting to polar coords
        radii = sqrt(sum(inliers.^2, 2));
        error = sum((radii - radius).^2);

        % add the circle to the circle candidates
        results = [results; error, center(1), center(2), radius]; 
    end
    [M, id] = max(results(1,:));
    params = results(id, 2:4);
    x0 = params(1);
    y0 = params(2);
    r = params(3);
    
    angles = linspace(0, 2*pi, 100);
    xf = r * cos(angles) + x0;
    yf = r * sin(angles) + y0;

    figure, imshow(im), hold on
    scatter(xe, ye, '.')
    plot(xf, yf, '-', LineWidth=2)
    legend('Edges', 'Fitted circle')
end

%% Built-In edge detection
plotEdges(im_moon)
plotEdges(im_jupiter_earth)
plotEdges(im_jupiter_partial)

%% Pseudo-inverse
findPlanetPseudoInverse(im_moon, 'contourDection_pseudoInverse_moon.png');
findPlanetPseudoInverse(im_jupiter_earth, 'contourDection_pseudoInverse_jupiter_earth.png');
findPlanetPseudoInverse(im_jupiter_partial, 'contourDection_pseudoInverse_jupiter_partial.png');

%% Circle detect with optimization

function circleFitOptimize(im)
    % Pre-process images
    im_gray = rgb2gray(im);
    im_bw = imbinarize(im_gray, 0.1);
    [height, width] = size(im_bw);
    
    % Detect edges
    [B, L] = bwboundaries(im_bw, 'noholes');
    %figure, imshow(im), hold on
    for k = 1:length(B)
       boundary = B{k};
       %scatter(boundary(:,2), boundary(:,1), 5, 'filled', 'o')
    end
    
    global data_pts;
    data_pts = B{1};
    function d = cost_circle(x)
        x_c = x(1); 
        y_c = x(2); 
        r   = x(3);
        %radii = data_pts - [x_c; y_c];
        %diff = r - radii
        d = sum(abs(r - sqrt((data_pts(:,1) - x_c).^2 + (data_pts(:,2) - y_c).^2)));
    end
    
    
    x0 = [width/2, height/2, 100];
    xopt = fminunc(@cost_circle, x0, optimset('Display', 'iter', 'PlotFcns', @optimplotfval));
    
    angles = 1:1:360;
    y = xopt(3) .* cos(angles * pi/180) + xopt(1);
    x = xopt(3) .* sin(angles * pi/180) + xopt(2);
    figure, axis('equal'), xlim([0 width]), ylim([0 height]), hold on%imshow(im), hold on
    scatter(data_pts(:,2), height-data_pts(:,1), 5, 'filled', 'o')
    plot(x, height-y, LineWidth=2)
    legend('Edges', 'Optimized fit')
    axis off
end

% settings to scale the plots uniformly
x0 = 0;
y0 = 0;
width = 800;
height = 600;

circleFitOptimize(im_moon);
set(gcf,'position',[x0,y0,width,height])
saveas(gcf, 'Optimization_moon.png');

circleFitOptimize(im_jupiter_earth);
set(gcf,'position',[x0,y0,width,height])
saveas(gcf, 'Optimization_jupiter_earth.png');

circleFitOptimize(im_jupiter_partial);
set(gcf,'position',[x0,y0,width,height])
saveas(gcf, 'Optimization_jupiter_partial.png');

%% Circle detect RANSAC

% settings to scale the plots uniformly
x0 = 0;
y0 = 0;
width = 800;
height = 600;

res1 = circleFitRANSAC(im_moon, 3500, 3, 50);
set(gcf,'position',[x0,y0,width,height])
exportgraphics(gcf, 'RANSAC_moon.png');

res2 = circleFitRANSAC(im_jupiter_earth, 1000, 3, 1000);
set(gcf,'position',[x0,y0,width,height])
exportgraphics(gcf, 'RANSAC_jupiter_earth.png');

res3 = circleFitRANSAC(im_jupiter_partial, 5000, 5, 10000);
set(gcf,'position',[x0,y0,width,height])
exportgraphics(gcf, 'RANSAC_jupiter_partial.png');

%% Circle detect using Hough transform

im = im_jupiter_earth;
[h, w] = size(im);

im_gray = rgb2gray(im);
im_bw = imbinarize(im_gray, 0.1);

[centers, radii, metric] = imfindcircles(im, [10 2*w], 'Sensitivity', 0.85, 'Method', 'TwoStage');
figure, imshow(im)
viscircles(centers, radii,'EdgeColor','b');
exportgraphics(gcf, 'hough_rgb_S085_TwoStage.png')

[centers, radii, metric] = imfindcircles(im, [10 2*w], 'Sensitivity', 0.9, 'Method', 'TwoStage');
figure, imshow(im)
viscircles(centers, radii,'EdgeColor','b');
exportgraphics(gcf, 'hough_rgb_S09_TwoStage.png')


[centers, radii, metric] = imfindcircles(im, [10 2*w], 'Sensitivity', 0.9, 'Method', 'PhaseCode');
figure, imshow(im)
viscircles(centers, radii,'EdgeColor','b');
exportgraphics(gcf, 'hough_rgb_S09_PhaseCode.png')

[centers, radii, metric] = imfindcircles(im_bw, [10 w]);
figure, imshow(im_bw)
viscircles(centers, radii,'EdgeColor','b');
exportgraphics(gcf, 'hough_bw.png')
