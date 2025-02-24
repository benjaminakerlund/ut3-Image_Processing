clc, clear, close all

% Generate the data

% randomly generate the ellipse's parameters
a = rand(1) + 2;                    % semi-major axis
b = rand(1) + 1;                    % semi-minor axis
center = 3 * (rand(2,1) - 0.5);     % focus of the ellipse
orientation = pi * (rand(1) - 0.5); % orientation of the ellipse in the x-y plane

% build the ellipse centered on the origin
angles = -pi:0.05:pi;
nPts = size(angles, 2);
x = a * cos(angles); 
y = b * sin(angles);

% rotate it and then offset it
x_th = x * cos(orientation) - y * sin(orientation) + center(1);
y_th = y * cos(orientation) + x * sin(orientation) + center(2);

% add noise to obtain imperfect data
level = 0.1;
noise = level * (rand(2, nPts) - 0.5);
data_x = x_th + noise(1, :);
data_y = y_th + noise(2, :);

% Apply the pseudo inverse approach
X = [data_x'.^2, 2*data_x'.*data_y', data_y'.^2, data_x', data_y'];
Y = pinv(X);
params = pinv(X) * ones(nPts, 1);
A = params(1); 
B = params(2);
C = params(3);
D = params(4);
E = params(5);

estimatedOrientation = pi/2 + 0.5 * atan2(2 * B, A-C);

% calculate the 'unrotated' ellipse parameters
A_ = A*cos(estimatedOrientation)^2 + B*cos(estimatedOrientation)*sin(estimatedOrientation) + C*sin(estimatedOrientation)^2;
B_ = 0;
C_ = A*sin(estimatedOrientation)^2 - B*cos(estimatedOrientation)*sin(estimatedOrientation) + C*cos(estimatedOrientation)^2;
D_ = D*cos(estimatedOrientation) + E*sin(estimatedOrientation);
E_ = -D*sin(estimatedOrientation) + E*cos(estimatedOrientation);

% calculate the center
x0 = -D_ / (2*A_);
y0 = -E_ / (2*C_);
estimatedCenter = [x0*cos(estimatedOrientation) - y0*sin(estimatedOrientation);
                   x0*sin(estimatedOrientation) + y0*cos(estimatedOrientation)];

% calculate axis
rhs = 1 + D_^2 / (4*A_) + E_^2 /(4*C_);
axis = [sqrt(rhs / A_), sqrt(rhs / C_)];
estimatedA = max(axis);
estimatedB = min(axis);

% sample the reconstructed ellipse
tmp_x = estimatedA * cos(angles);
tmp_y = estimatedB * sin(angles);
x_reconstructed = tmp_x * cos(estimatedOrientation) - tmp_y * sin(estimatedOrientation) + estimatedCenter(1);
y_reconstructed = tmp_y * cos(estimatedOrientation) + tmp_x * sin(estimatedOrientation) + estimatedCenter(2);


% plot go brrrrrr
figure, plot(x_th, y_th, '-', data_x, data_y, '.', x_reconstructed, y_reconstructed, '-')
legend('Original ellipse', 'Noisy ellipse', 'Reconstructed ellipse')
exportgraphics(gcf, 'ReconstructedAsteroidTrajectory.png');
