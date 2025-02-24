%% Shape-based approach  

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
    subplot(2, 2, 1), imshow(im), title('Original')
    subplot(2, 2, 2), imshow(im_gray), title('Gray scale')
    subplot(2, 2, 3), imshow(im_bw), title('Binary')
    subplot(2, 2, 4), imshow(im_opt), title('Fit')
end

% Load the images
im_moon            = imread('Images/lunar-eclipse-sep-28-2015-michelle-wood-1.jpg');
im_jupiter_earth   = imread('Images/Jupiter-dreams-meaning.jpg');
im_jupiter_partial = imread('Images/d52ed43e4a_106828_jupiter-pole-sud-juno.jpg');

find_planet(im_moon);
find_planet(im_jupiter_earth);
find_planet(im_jupiter_partial);

