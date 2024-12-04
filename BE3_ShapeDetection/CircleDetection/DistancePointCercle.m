function d = DistancePointCercle(C)
global ListePoints TrajCercles

P = ListePoints ;

d = sum (abs (C(3) - sqrt( (P(:,1)-C(1)).^2 + (P(:,2)-C(2)).^2 ) ) ) ;

TrajCercles = [TrajCercles ; C(1) C(2) C(3) d ] ;