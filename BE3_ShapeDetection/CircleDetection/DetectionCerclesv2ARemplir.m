%%%%%%%%%%%%%
%% (c) Emmanuel Zenou
%%
%% Algorithme d'approximation de cercles
%%
%%%%%%%%%%%%%


clear all, close all, clc ;
global ListePoints TrajCercles

NbPts = 50 ;


%% Points theoriques :

thfull = 0 : 1 : 360 ;

% Point sur un cercle avec bruit sur le rayon
Tx = 0.05 ;
th = floor (360 * rand(NbPts,1) ) ;
xth = 10 * (rand - .5) ;
yth = 10 * (rand - .5) ;
rth = max(5 * rand,1) ;
Ptsth(:,1) = xth + rth*(1+Tx*randn(NbPts,1)) .* cos(th*pi/180) ;
Ptsth(:,2) = yth + rth*(1+Tx*randn(NbPts,1)) .* sin(th*pi/180) ;

% Points particuliers :
% Pts = [1 1 ; 1 3 ; 3 1] ;

ListePoints = Ptsth ;


disp(['Centre théorique : (' num2str(xth) ',' num2str(yth) '), Rayon théorique = ' num2str(rth)]) ;

figure,
plot(Ptsth(:,1),Ptsth(:,2),'xb') ;
axis equal ;
title('Points théoriques') ;
saveas(gcf,'im_PointsTheoriques.png','png')


%% Approche Optimisation




%% Approche pseudo-inverse

