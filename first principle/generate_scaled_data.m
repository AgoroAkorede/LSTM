load('training_data_simple')

[~, q3_init_ss] = get_init_op();
yss= 7 ;

u_train_un = train_dataset.u.signals(1).values(:,1);
y_train_un = train_dataset.x.signals(3).values(:,1);

xmin=min(u_train_un)  ; xmax=max(u_train_un)  ; 
ymin=min(y_train_un)  ; ymax=max(y_train_un)  ; 

uo = scale_input(q3_init_ss,xmin,xmax,0,1);
u = scale_input(u_train_un,xmin,xmax,0,1);

yo = scale_input(yss,ymin,ymax,0,1);
y = scale_input(y_train_un,ymin,ymax,0,1);

data_size = numel(u);
Ndata = data_size ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_2=[yo , yo, y(1:Ndata-2)'];  
y_1=[yo, y(1:Ndata-1)'];


u_2=[uo , uo,u(1:Ndata-2)'];  
u_1=[uo, u(1:Ndata-1)'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% REGRESSED INPUTS AND OUTPUTS FOR 3
X = [y_1 ;  y_2 ; u_1 ;  u_2 ];
Y = y(1:Ndata)';

X_trans=X' ;
Y_trans=Y'  ;