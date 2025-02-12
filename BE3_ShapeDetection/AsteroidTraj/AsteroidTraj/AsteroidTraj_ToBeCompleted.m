clear all, close all, clc ;


% Génération
t=-pi:0.03:pi ;

Tx = 0.01 ; % Taux de bruit



% Ellipse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T = 1.5 ;
Fenetre = [-T T -T T] ; % [xmin xmax ymin ymax]

% Equation
ae = monrand(5) ;
be = monrand(5) ;
centre = monrandn(2,3) ; % centre
orient = pi*(rand-0.5) ;
xe = (ae*cos(t))*cos(orient) - (be*sin(t))*sin(orient) + centre(1) ;
ye = (ae*cos(t))*sin(orient) + (be*sin(t))*cos(orient) + centre(2) ;
xeb = xe+Tx*monrandn(size(t,2),3) ;
yeb = ye+Tx*monrandn(size(t,2),3) ;

% Selection of observable points
Pef = (xeb>Fenetre(1))&(xeb<Fenetre(2))&(yeb>Fenetre(3))&(yeb<Fenetre(4)) ;
[i,j,xef] = find(xeb.*Pef) ;
[i,j,yef] = find(yeb.*Pef) ;
figure, plot(xe,ye,'-',xeb,yeb,'.',xef,yef,'.') ; 
legend('Ellipse originale','Ellipse bruitée','Sélection') ;

% Conic recovery :
X = ??? ;
A = ??? ;
disp(['Equation de la conique : ' num2str(A(1)) '*x^2+' num2str(A(2)) '*x*y+' num2str(A(3)) '*y^2+' num2str(A(4)) '*x+' num2str(A(5)) '*y+1=0']) ;
figure, ezplot([num2str(A(1)) '*x^2+' num2str(A(2)) '*x*y+' num2str(A(3)) '*y^2+' num2str(A(4)) '*x+' num2str(A(5)) '*y+1']) ;
%axis([-1 1 -1 1]) ;
figure,
plot(xef,yef,'r.',xe,ye,'k-') ;
legend('Sélection','Ellipse originale') ; hold ;
ezplot([num2str(A(1)) '*x^2+' num2str(A(2)) '*x*y+' num2str(A(3)) '*y^2+' num2str(A(4)) '*x+' num2str(A(5)) '*y+1']) ;


