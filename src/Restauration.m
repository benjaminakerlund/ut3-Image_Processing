close all; clc;

SeuilMax = 11;
T=3;
Hubble = 0;

% Estimation :
N = 512 ;
u = [-300 : 0.1 : 300] ;
Hu = sin(2*pi*u/N*(T+0.5))./sin(pi*u/N) ;
sincu = sin(2*pi*u/N*(T+0.5))./(2*pi*u/N*(T+0.5)) ;
figure, plot(u,Hu,u,sincu,u,0), legend('H(u)','sinc(u)');


if (Hubble == 0)
    I = imread('../BE1_IntroComputerVision/Toulouse.bmp');
    %I = rgb2gray(I) ;
    TailleImage = size(I);
    figure;imshow(I);title('Image originale');
    %imwrite(I,'Deconv1_Or.png','png') ;

    h=fspecial('average',2*T+1)
    %h=fspecial('disk',2*T+1);
    % h = [1 1 1 1 1;1 3 3 3 1;1 3 8 3 1;1 3 3 3 1;1 1 1 1 1] / 48; 
    TailleFiltre = size(h);
    If=imfilter(I,h);
    figure,imshow(If);title('Image floue sans bruit');
    %imwrite(If,'Deconv2_b1.png','png') ;

    pcb = 0 / 100 ; % pourcentage de bruit
    B = double(max(max(If))) * pcb * rand(size(If)) ;
    Ifb = uint8(double(If) + B);
    figure,imshow(Ifb);title('Image floue bruitée');
    %imwrite(Ifb,'Deconv3_b2.png','png') ;
    
    moy = fspecial('average',3);
    Ifb = imfilter(Ifb,moy);
    figure,imshow(Ifb);title('Image floue bruitée débruitée');
end


if(Hubble == 1)
    If = imread('hubble.jpg') ;
    If = rgb2gray(If) ;
    TailleImage = size(If);
    figure, imshow(If), title('Image de Hubble floue');
    TailleFiltre = [5 10] ;
    Ifb=If;
end

% marcheur
if(Hubble == 2)
    I=imread('../BE1_IntroComputerVision/marcheur.jpg');
    If = rgb2gray(I);
    figure, imshow(If), title('Marcheur');
    Ifb = If ;
    TailleImage = size(If);
end


TFdIf = fft2(If);
TFdIfbis = fftshift(TFdIf);
logTFdIfbis = log(abs(TFdIfbis));
figure, imagesc(logTFdIfbis), colormap(gray);

TFdIfb = fft2(Ifb);
TFdIfbbis = fftshift(TFdIfb);
logTFdIfbbis = log(abs(TFdIfbbis));
figure, imagesc(logTFdIfbbis), colormap(gray);

logfft = log(abs(fftshift(TFdIf))+eps);
figure, imagesc(logfft)
colormap gray, colorbar,title('TFR de l image dégradée (échelle log)');
sumligfft = sum(logfft')';
sumcolfft = sum(logfft);
figure
subplot(1,2,1),plot(sumligfft),title('somme de la TFR selon les lignes');
subplot(1,2,2),plot(sumcolfft),title('somme de la TFR selon les colonnes');


%% Filtrage inverse
%%%%%%%%%%%%%%%%%%
hh = zeros(TailleImage);
centre = [1 1] + floor(TailleImage/2) ;
ext = (TailleFiltre-[1 1])/2;
ligs = centre(1) + [-ext(1):ext(1)];
cols = centre(2) + [-ext(2):ext(2)];

h = ones(TailleFiltre)/prod(TailleFiltre);
hh(ligs,cols) = h;
hh = ifftshift(hh);

H = fft2(hh);

ind = find(abs(H)<(1/SeuilMax));
H(ind) = (1/SeuilMax)*exp(j*angle(H(ind)));

G = ones(size(H))./H;


% Restauration

TFdIf = fft2(If);
TFdIrest = G.*TFdIf;
Irest = real(ifft2(TFdIrest));
Irest = uint8(Irest);
figure, imshowpair(If, Irest, "montage"), title('Image non bruitée restaurée par filtrage inverse');

    %imwrite(Irest,'Deconv4_fin.png','png') ;

if (Hubble == 0)
    TFdIfb = fft2(Ifb);
    TFdIrestb = G.*TFdIfb;
    Irestb = real(ifft2(TFdIrestb));
    Irestb = uint8(Irestb);
    figure, imshowpair(Ifb, Irestb, "montage"), title('Image bruitée restaurée par filtrage inverse');
end

% Filtrage de Wiener
%%%%%%%%%%%%%%%%%%%%

% IrestW = deconvwnr(If,h);
% figure;imshow(IrestW);
% title('Image restaurée à l aide du filtre de Wiener');

