close all, clc, clear
    
function find_planet(im)
    global im_bw; % need this for the optimization call
    
    % Process to black and white 
    im_gray = rgb2gray(im);
    
    % Convert to binary
    threshold = 0.15;
    im_bw = imbinarize(im_gray, threshold);
    [height, width] = size(im_bw);
    
    % Create model (simple circle)
    function im = im_circle(width, height, x_c, y_c, r)
        im = zeros(height, width, 'logical');
        for i = 1:width
            for j = 1:height
                if (i - x_c).^2 + (j - y_c).^2 <= r.^2
                    im(j, i) = 1;
                end
            end
        end
    end
    
    % Cost function to fit a circle
    function diff = cost(x)
        %global im_bw;
        [height, width] = size(im_bw);
        x_c = x(1);
        y_c = x(2);
        r   = x(3);
        im_c = im_circle(width, height, x_c, y_c, r);
        diff = sum(abs(im_bw - im_c), "all");
    end
    
    % optimization
    x0 = [height/2, width/2, 100]; % initial guess
    options          = optimset;
    %options.Display  = 'iter'; % show values at each iteration
    options.TolFun   = 1e-6;
    options.TolX     = 1e-6;
    options.PlotFcns = @optimplotfval; % plot the process
    xopt = fminsearch(@cost, x0, options); 
    
    disp(xopt);
    
    im_opt = im_circle(width, height, xopt(1), xopt(2), xopt(3));
    
    figure
    subplot(2, 2, 1), imshow(im)
    subplot(2, 2, 2), imshow(im_gray)
    subplot(2, 2, 3), imshow(im_bw)
    subplot(2, 2, 4), imshow(im_opt)
end

% Load the image
im_moon            = imread('../BE3_ShapeDetection/Images/lunar-eclipse-sep-28-2015-michelle-wood-1.jpg');
im_jupiter_earth   = imread('../BE3_ShapeDetection/Images/Jupiter-dreams-meaning.jpg');
im_jupiter_partial = imread('../BE3_ShapeDetection/Images/d52ed43e4a_106828_jupiter-pole-sud-juno.jpg');

%% Shape-based approach
find_planet(im_moon);
find_planet(im_jupiter_earth);
find_planet(im_jupiter_partial);

%% Contour-based
clc, close all

% Pre-process images
im = im_jupiter_earth;
im_gray = rgb2gray(im);
im_bw = imbinarize(im_gray, 0.1);
[height, width] = size(im_bw);

% Detect edges
[B, L] = bwboundaries(im_bw, 'noholes');
figure, imshow(im), hold on
for k = 1:length(B)
   boundary = B{k};
   scatter(boundary(:,2), boundary(:,1), 5, 'filled', 'o')
end

%% Circle detect with optimization
global data_pts;
data_pts = B{1};
function d = cost_circle(x)
    global data_pts
    x_c = x(1); 
    y_c = x(2); 
    r   = x(3);
    d = sum(abs(r - sqrt((data_pts(:,1) - x_c).^2 + (data_pts(:,2) - y_c).^2)));
end


x0 = [width/2, height/2, 100];
xopt = fminunc(@cost_circle, x0, optimset('Display', 'iter'));

angles = 1:1:360;
y = xopt(3) .* cos(angles * pi/180) + xopt(1);
x = xopt(3) .* sin(angles * pi/180) + xopt(2);
figure, imshow(im), hold on
scatter(data_pts(:,2), data_pts(:,1), 5, 'filled', 'o')
scatter(x, y)

%% Circle detect with pseudo inverse