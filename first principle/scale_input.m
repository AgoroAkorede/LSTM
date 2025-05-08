function y=scale_input(x,xmin,xmax,ymin,ymax)

y=ymin + ((x-xmin)*(ymax-ymin))/(xmax-xmin) ;

