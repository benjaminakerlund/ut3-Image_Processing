clear all, close all, clc ;


%% Paramètres

mu1 = [-5 -5] ;
mu2 = [5 5] ;

sigma1 = [1 3] ;
sigma2 = [3 1] ;

NbPts = 20 ;

alph = 0.5 ;


%% Génération des données

for i = 1 : NbPts
    zedata(i,1) = mu1(1) + sigma1(1) * randn ;
    zedata(i,2) = mu1(2) + sigma1(2) * randn ;
    zedata(i,3) = 1 ;
    zedata(i+NbPts,1) = mu2(1) + sigma2(1) * randn ;
    zedata(i+NbPts,2) = mu2(2) + sigma2(2) * randn ;
    zedata(i+NbPts,3) = -1 ;
end

figure,
plot(zedata(1:NbPts,1),zedata(1:NbPts,2),'bo') ;
hold
plot(zedata(NbPts+1:end,1),zedata(NbPts+1:end,2),'ko') ;


% On mélange les données
permut = randperm(2*NbPts) ;
zedatam = zedata(permut,:) ;

w = rand(1,2) ;
w0 = 1 ;
xh = -15 : .1 : 15 ;

yh = -(w0 + w(1) * xh) / w(2) ; % w0 + w1.x + w2.y
plot(xh,yh,'y') ;

while 1
    flag = 0 ;
    for NoPt = 1 : NbPts * 2
        plot(zedatam(NoPt,1),zedatam(NoPt,2),'or') ;
        klass = zedatam(NoPt,3) ;
        hdx = w0 + w * zedatam(NoPt,1:2)' ;
        if (klass * hdx < 0)                % Mal classé
            w = w + klass * alph * zedatam(NoPt,1:2) ;
            flag = 1 ;
        end
        yh = -(w0 + w(1) * xh) / w(2) ; % w0 + w1.x + w2.y
        plot(xh,yh,'g') ;


        if klass == 1
            plot(zedatam(NoPt,1),zedatam(NoPt,2),'ob') ;
        else
            plot(zedatam(NoPt,1),zedatam(NoPt,2),'ok') ;
        end
    end

    if flag == 0
        break ;
    end
end


        yh = -(w0 + w(1) * xh) / w(2) ; % w0 + w1.x + w2.y
        plot(xh,yh,'r') ;

