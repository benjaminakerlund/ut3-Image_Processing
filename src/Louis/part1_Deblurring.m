%% Question 17
clc, clear, close all

im = imread('toulouse.bmp');
[height, width] = size(im);
T = 3;
h = ones(2*T+1) / (2*T+1)^2; % square kernel
imb = uint8(conv2(im, h, "same"));
noiseLevel = 2;
noise = noiseLevel * randn(height, width);
imn = imb + uint8(noise);

% for convenience
images = {im, imb, imn};
titles = {'Original', 'Blurred', 'Blurred and Noisy'};

figure, imshow(im), exportgraphics(gca, '../Q17_Original.png')
figure, imshow(imb), exportgraphics(gca, '../Q17_Blurred.png')
figure, imshow(imn), exportgraphics(gca, '../Q17_Noisy.png')

% Fourier transform go brrrr
ft = {};
spectrum = {};
for i=1:length(images)
    ft{i} = fft2(images{i});
    ft{i} = fftshift(ft{i});
    spectrum{i} = abs(ft{i});
    figure, imagesc(log(spectrum{i})), axis('equal'), axis off
    exportgraphics(gca, strcat('../Q17_FT_Spectrum_', titles{i}, '.png'));
end

%% Question 18
diff = spectrum{2} - spectrum{1};
logdiff = log(spectrum{2}) - log(spectrum{1});
figure, imagesc(diff), axis equal off
exportgraphics(gca, '../Q18_FT_Spectrum.png');
figure, imagesc(logdiff), axis equal off
exportgraphics(gca, '../Q18_FT_Spectrum_Log.png');

%% Question 19
clc, close all

data = {};
data{1} = sum(log(spectrum{2}), 1);
data{2} = sum(log(spectrum{2}), 2);
% data{1} = data{1} - mean(data{1});
% data{1} = data / max(data{1})*1.5;

figure, plot(data{1}), xlim([0 512]), saveas(gcf, '../Q19_SumX.png')
figure, plot(data{2}), xlim([0 512]), saveas(gcf, '../Q19_SumY.png')

%%
x = 1:512;
x0 = 257;
t = [2.7, 2.53, 2.0];
y1 = sinc((x-x0)/256*2*(t(1)+0.5)) -0.45;
y2 = sinc((x-x0)/256*2*(t(2)+0.5)) -0.4;
y3 = sinc((x-x0)/256*2*(t(3)+0.5)) -0.15;
plot(x, y1, x, y2, x, y3)
legend('Data', strcat('t = ', string(t(1))), strcat('t = ', string(t(2))), strcat('t = ', string(t(3))))
exportgraphics(gca, '../Q19_EstimatedT.png')

%% Question 20
clc, clear, close all

im = imread('toulouse.bmp');


T = 3;
SeuilMax = 11 ;
hh = zeros(size(im));
center = [1 1] + floor(size(im)/2);

ext = (T-[1 1])/2;
ligs = center(1) + [-ext(1):ext(1)];
cols = center(2) + [-ext(2):ext(2)];
h = ones(T)/prod(T);

hh(ligs,cols) = h;
hh = ifftshift(hh);
H = fft2(hh);
ind = find(abs(H)<(1/SeuilMax));
H(ind) = (1/SeuilMax)*exp(j*angle(H(ind)));
G = ones(size(H))./H;

%% Question 21
clc, clear, close all