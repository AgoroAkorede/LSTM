function [u_first_move,fcost,Ecount,exitflag,y_pred,hk,ck,xPx, Tconst] = u_compute_pH(y_set,yo,ypast,uo,upast,d,tp,ts,e,ho,Co,P,alpha)
    % e = 0;
    data = struct();
    data.upast = upast;
    data.ypast = ypast;
    data.uo = uo;
    data.yo = yo;
    data.ts = ts;
    data.d = d;
    data.e = e;
    data.y_set = y_set;
    data.Nu = tp(1);
    data.Np = tp(2);
    data.W_y = tp(3);
    data.W_du = tp(4);
    data.y_min = 1;
    data.y_max = 14;
    data.du_min = -50;
    data.du_max = 50;
    data.u_min = 0;
    data.u_max = 50;
    data.ho=ho ;
    data.Co=Co ;
    data.alpha=alpha ;
    data.P = P ;


    
    %Converts u_min and u_max to a column vector with a dimension equal
    %to the magnitude of the control horizon. These vectors will be used
    %by the optimization function
    data.u_min_colvector = data.u_min*ones(data.Nu,1);
    data.u_max_colvector = data.u_max*ones(data.Nu,1);

    data.u_initial_colvector = uo*ones(data.Nu,1);
    
  % OPTIONS = optimset('Algorithm','sqp','Display','final');
OPTIONS = optimoptions('fmincon','Algorithm','sqp','Display','final', 'FiniteDifferenceStepSize', 1e-3,'FiniteDifferenceType', 'central');
% OPTIONS = optimoptions('fmincon','Algorithm','sqp','Display','final', 'FiniteDifferenceStepSize', 1e-5);
% OPTIONS = optimoptions('fmincon','Algorithm','sqp','Display','final', 'FiniteDifferenceStepSize', 1e-1);
% OPTIONS = optimset('Algorithm','active-set');
% OPTIONS = optimset('Algorithm','interior-point');

tic

    [uopt,fval,exitflag] = fmincon(@(u)compute_cost(u,data),data.u_initial_colvector,[],[],[],[],data.u_min_colvector,data.u_max_colvector,@(u)nlcf(u,data),OPTIONS);

 Ecount = toc ;
 fcost=fval ;

%   Uncomment to allow for a more robust optimum search. Not needed though,
%   issue is with the faulty lstm model
%   Sadly parallel pool cannot be used due to race conditions when calling predictAndUpdateState
%
%     num_init_pts = 3;
%     spacing = round((data.du_max - data.du_min)/num_init_pts);
%     init_pts = data.du_min:spacing:data.du_max;
%     u_init_mat = zeros(num_init_pts,data.Nu);
%     for i = 1:numel(init_pts)
%        u_init_mat(i,:) = init_pts(i) + u_past; 
%     end
%     init_pts_obj = CustomStartPointSet(u_init_mat);
%     ms = MultiStart('UseParallel', false, 'Display', 'iter');
%     problem = createOptimProblem('fmincon','x0',data.u_initial_colvector,'objective',@(u)compute_cost(u,data),'lb',data.u_min_colvector,'ub',data.u_max_colvector,'nonlcon',@(u)nlcf(u,data),'options',OPTIONS);
%     [u_k,cost] = run(ms,problem,init_pts_obj);
%     if (numel(u_k) == 0)
%        u_k = u_past*ones(data.Nu,1);
%        cost = compute_cost(u_k,data);
%     end
    
    u_first_move  = uopt(1);


    
% [yk,h_update,C_update] = lstm_predict(u,upast,uo, Nu, Np,ho,Co)

[yk,h_update,C_update] = lstm_predict(u_first_move,upast,ypast,yo,uo, 1, 1,ho,Co)  ;
% [y_k,h_update,C_update] = lstm_predict(u,upast,ypast,yo,uo, Nu, Np,ho,Co)

hk=double(h_update) ; ck=double(C_update) ;
y_pred=double(yk) ;


[xPx, Tconst]=terminal_const(uopt,data);



   % yk = lstm_predict(u_k, Nu, Np,h1,C1, h2, C2) 

        
    % Compensate for error
    % y_pred = y_pred + data.e;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% cost function starts here
function cost = compute_cost(u,data)
    %This function returns the cost or performance index, J of implementing a
    %vector u of control moves on the plant.
   

    Nu = data.Nu;
    Np = data.Np;
    ho = data.ho;
    Co = data.Co;
    uo = data.uo;
    yo = data.yo;
    upast = data.upast  ;
    ypast = data.ypast ;


    %Input rate vector with a dimension of Nu
    % du = zeros(Nu,1);
    du(1) = u(1) - data.uo;
    for i = 2:Nu
        du(i) = u(i) - u(i-1);
    end
    

    y_k=LSTM_output_prediction(u,data,uo,upast,ypast, yo, Nu,Np,ho,Co) ;
    
    cost_du = du*data.W_du*du';
    

    cost = cost_du + sum(data.W_y*((y_k - data.y_set).^2));


end

function [c,ceq] = nlcf(u,data)
    %This evaluates the nonlinear inequality constraints which are bound
    %to the problem. nlcf returns two major parameters: c and ceq. c is nonlinear
    %inequality constraint function while ceq is the nonlinear equality constraint
    %function.

    Nu = data.Nu;
    Np = data.Np;
    ho = data.ho;
    Co = data.Co;
    uo = data.uo;
    yo = data.yo;
    upast = data.upast  ;
    ypast = data.ypast ;
    r1 = data.y_set;
    alpha = data.alpha ;
    P = data.P;
    


    y_min_colvector = data.y_min*ones(Np,1);
    y_max_colvector = data.y_max*ones(Np,1);
    du_min_colvector = data.du_min*ones(Nu,1);
    du_max_colvector = data.du_max*ones(Nu,1);
    u_min_colvector = data.u_min_colvector(1:Nu);
    u_max_colvector = data.u_max_colvector(1:Nu);

    y_k=LSTM_output_prediction(u,data,uo,upast,ypast, yo, Nu,Np,ho,Co)  ;
    
    %Input rate vector with a dimension of Nu
    % du = zeros(Nu,1);
    du(1) = u(1) - uo;
    for i = 2:Nu
        du(i) = u(i) - u(i-1);
    end
    
    ineq_du_min = du_min_colvector - du';
    ineq_du_max = du' - du_max_colvector;
    ineq_y_min = y_min_colvector - y_k';
    ineq_y_max = y_k' - y_max_colvector;
    ineq_u_min = u_min_colvector - u;
    ineq_u_max = u - u_max_colvector;
    
    ceq = [];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % % LYAPUNOV CONSTRAINTS
    % % u_first_move=uo ;
    % % [yk,h_update,C_update] = lstm_predict(u_first_move,upast,uo, 1, 1,ho,Co)  ;
    % % hk=double(h_update) ; ck=double(C_update) ;
    % g=[] ;
    % % rhomin=2 ;  %rhomin=2 ;372
    % rhomin = alpha ;
    % 
    % % rhomin=372 ;  %rhomin=2 ;372
    % % P=[1060,  22 ;
    % %    22,   0.52] ;   %  P=10    P=eye(20)
    % 
    % dt=0.01 ;
    % k=15 ;% k=0.15
    % 
    % xo=yo; y_set=[r1];  %  y_set=[r1;r2];
    % % xt=[y_set-xo] ;  %     xt=[ck;hk]% xt=[xo-y_set] 
    % xt=[yo-y_set] ;  %     xt=[ck;hk]% xt=[xo-y_set]
    %    % xo=[Co;ho]   %      xo=[Co;ho]
    % 
    % V=xt'*P*xt ;
    % 
    % y_k=y_k(end,:);
    % x_dot=(y_k'-yo)/dt ;
    % V_dot=2*xt'*P*x_dot ;
    % if V>rhomin
    %     g=[g;V_dot+k*abs(V/100)] ;
    %     % g=[V_dot+k*abs(V)]
    % elseif V<=rhomin
    %     xNN= [y_k'-y_set] ;
    %     V_xNN=xNN'*P*xNN  ;
    %     g=[g;V_xNN-rhomin] ;
    %     % g=[V_xNN-rhomin]
    % % x=[]
    % end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   c = [ineq_du_min; ineq_du_max; ineq_y_min; ineq_y_max; ineq_u_min; ineq_u_max ];   

    % c = [ineq_du_min; ineq_du_max; ineq_y_min; ineq_y_max; ineq_u_min; ineq_u_max ; g];
    % c = [ineq_du_min; ineq_du_max; ineq_y_min; ineq_y_max ; g]  ;
    
    % for i = 1:numel(c)
    %     if (c(i) == inf)
    %         c(i) = 1e23;
    %     end
    %     if (c(i) == -inf)
    %         c(i) = -1e23;
    %     end
    % end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xPx, Tconst]=terminal_const(u,data);

    Nu = data.Nu;
    Np = data.Np;
    ho = data.ho;
    Co = data.Co;
    uo = data.uo;
    yo = data.yo;
    upast = data.upast  ;
    ypast = data.ypast ;
    r1 = data.y_set;
    alpha = data.alpha ;
    P = data.P;

yk=LSTM_output_prediction(u,data,uo,upast,ypast, yo, Nu,Np,ho,Co)  ;

% yk=output_prediction(u,xo,uo,upast,ypast,Np,Nu,dy,ho,Co)

% [xk]=output_prediction(xo,uo,ko,du,hstep,Nsteps,Np,Nu);


xPx = [yk(Np)-[r1]]*P*[yk(Np)-[r1]]'   ;
Tconst=[yk(Np)-[r1]]*P*[yk(Np)-[r1]]' -  alpha    ;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function y_k=LSTM_output_prediction(u,data,uo,upast,ypast, yo, Nu,Np,ho,Co) ;

% yk = lstm_predict(u,uo,upast,ypast, yo, Nu,Np,false)  ;
yk = lstm_predict(u,upast,ypast,yo,uo, Nu, Np,ho,Co) ;

 y_k = yk + data.e*ones(1,Np);


end


function [y_k,h_update,C_update] = lstm_predict(u,upast,ypast,yo,uo, Nu, Np,ho,Co) ;


    y_k = zeros(1, Np);


    for i = 1:Np


       % if (i == 1)
       %      u_NN = [uo,upast] ;
       %  end
       % 
       %  if (i == 2)
       %      u_NN = [u(1,:),uo]  ;
       %  end
       % 
       %  if (i >= 3 && i<Nu)
       %      u_NN = [u(i-1,:),u(i-2,:)] ;
       %  end
       % 
       %  if i>=Nu
       %      u_NN = [u(Nu,:),u(Nu,:)] ;
       %  end

         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (i == 1)
            u_NN = [u(1,:),uo]; % [u(1), u(0)]
        elseif (i <= Nu)
            u_NN = [u(i,:),u(i-1,:)]; % [u(2),u(1)] if i=2
        else
            u_NN = [u(Nu,:),u(Nu,:)];
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        if (i == 1)
            y_NN = [yo,ypast];
        elseif (i == 2)
            y_NN = [y_k(:,1),yo];
        else
            y_NN = [y_k(:,i-1),y_k(:,i-2)];
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % if (i == 1)
        %     y_NN = [y_k(:,1),yo];
        % else
        %     y_NN = [y_k(:,i),y_k(:,i-1)];
        % end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % scale_input(x,xmin,xmax,ymin,ymax)
        xmin =  0.0036 ;  xmax  =    35  ; ymin = 2.7489  ; ymax = 10.8798 ;
     
        y_1=scale_input2(y_NN(1),ymin,ymax,0,1) ;
        y_2=scale_input2(y_NN(2),ymin,ymax,0,1) ;

        u_1=scale_input2(u_NN(1),xmin,xmax,0,1) ;
        u_2=scale_input2(u_NN(2),xmin,xmax,0,1) ;

        uNN_scaled = [y_1 ;  y_2 ; u_1 ;  u_2 ];

        % uNN_scaled = [u_1 ;  u_2 ];

        [y_ki, h_update,C_update] = lstm_network(uNN_scaled,ho,Co)  ;
        ho=h_update ;Co=C_update ;
        y_k(:,i) = scale_input2(y_ki,0,1,ymin,ymax) ;
        
       
        % [y_k, h1_update,C1_update,h2_update,C2_update] = lstm_network(u,h1,C1, h2, C2) ;
        % 
        % h1=h1_update ;C1=C1_update; h2=h2_update ;C2=C2_update ;
        % 
        %  yk(:,i) = y_k ;

       % yk(i) = lstm_network(u(i),h1(:,i),C1(:,i), h2(:,i), C2(:,i))  ;
    end



end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y=scale_input2(x,xmin,xmax,ymin,ymax)

y=ymin + ((x-xmin)*(ymax-ymin))/(xmax-xmin) ;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ypred, h1_update,C1_update] = lstm_network(xt,ho,Co) ;

% load('training_data_simple.mat')
load('lstm_pH_data.mat')
% net_inputs=4;
num_hidden_units=10 ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the model weights and biases for each LSTM layer
% First LSTM Layer
Wi1 = Wu1(1:num_hidden_units,:);  % Forget gate input weights (layer 1)
Ui1 = WR1(1:num_hidden_units,:);  % Forget gate recurrent weights (layer 1)
bi1 = b1(1:num_hidden_units,:);  % Forget gate biases (layer 1)

Wf1 = Wu1(num_hidden_units+1:2*num_hidden_units,:);  % Input gate input weights (layer 1)
Uf1 = WR1(num_hidden_units+1:2*num_hidden_units,:);  % Input gate recurrent weights (layer 1)
bf1 = b1(num_hidden_units+1:2*num_hidden_units,:);  % Input gate biases (layer 1)

Wc1 = Wu1(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate input weights (layer 1)
Uc1 = WR1(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate recurrent weights (layer 1)
bc1 = b1(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate biases (layer 1)

Wo1 = Wu1(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state input weights (layer 1)
Uo1 = WR1(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state recurrent weights (layer 1)
bo1 = b1(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state biases (layer 1)



% % Second LSTM Layer
% Wi2 = Wu2(1:num_hidden_units,:);  % Forget gate input weights (layer 1)
% Ui2 = WR2(1:num_hidden_units,:);  % Forget gate recurrent weights (layer 1)
% bi2 = b2(1:num_hidden_units,:);  % Forget gate biases (layer 1)
% 
% Wf2 = Wu2(num_hidden_units+1:2*num_hidden_units,:);  % Input gate input weights (layer 1)
% Uf2 = WR2(num_hidden_units+1:2*num_hidden_units,:);  % Input gate recurrent weights (layer 1)
% bf2 = b2(num_hidden_units+1:2*num_hidden_units,:);  % Input gate biases (layer 1)
% 
% Wc2 = Wu2(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate input weights (layer 1)
% Uc2 = WR2(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate recurrent weights (layer 1)
% bc2 = b2(2*num_hidden_units+1:3*num_hidden_units,:);  % Output gate biases (layer 1)
% 
% Wo2 = Wu2(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state input weights (layer 1)
% Uo2 = WR2(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state recurrent weights (layer 1)
% bo2 = b2(3*num_hidden_units+1:4*num_hidden_units,:);  % Cell state biases (layer 1)


% Wy=poly_net.Layers(3,1).Weights ;
% by=poly_net.Layers(3,1).Bias ;

% % Initialize hidden and cell states for both LSTM layers
% h1 = zeros(num_hidden_units, 1);  % Hidden state for layer 1
% C1 = zeros(num_hidden_units, 1);  % Cell state for layer 1
% 
% h2 = zeros(num_hidden_units, 1);  % Hidden state for layer 2
% C2 = zeros(num_hidden_units, 1);  % Cell state for layer 2

% Assuming 'testData' is the input sequence you want to predict on

% testData=test_data_u;
% [num_features, num_time_steps] = size(testData);
% 
% % Initialize an array to store the outputs at each time step
% pH_pred = zeros(net_inputs, num_time_steps);
% outputSequence = zeros(num_hidden_units, num_time_steps);
    
    % ============================
    % First LSTM Layer Calculations
    % ============================
    
    % Forget gate (layer 1)
    f1 = sigmoid(Wf1 * xt + Uf1 * ho + bf1) ;
    
    % Input gate (layer 1)
    i1 = sigmoid(Wi1 * xt + Ui1 * ho + bi1);
    
    % Candidate cell state (layer 1)
    C1_candidate = tanh(Wc1 * xt + Uc1 * ho + bc1);
    
    % Update cell state (layer 1)
    C1 = f1 .* Co  + i1 .* C1_candidate;  % original
    % C1 = i1 .* C1  + f1 .* C1_candidate;
    
    % Output gate (layer 1)
    o1 = sigmoid(Wo1 * xt + Uo1 * ho + bo1);
    
    % Update hidden state (layer 1)
    h1 = o1 .* tanh(C1);
    
    % % The output of the first LSTM layer (h1) becomes the input for the second LSTM layer
    % 
    % % ============================
    % % Second LSTM Layer Calculations
    % % ============================
    % 
    % % Forget gate (layer 2)
    % f2 = sigmoid(Wf2 * h1 + Uf2 * h2 + bf2);
    % 
    % % Input gate (layer 2)
    % i2 = sigmoid(Wi2 * h1 + Ui2 * h2 + bi2);
    % 
    % % Candidate cell state (layer 2)
    % C2_candidate = tanh(Wc2 * h1 + Uc2 * h2 + bc2);
    % 
    % % Update cell state (layer 2)
    % C2 = f2 .* C2 + i2 .* C2_candidate;
    % 
    % % Output gate (layer 2)
    % o2 = sigmoid(Wo2 * h1 + Uo2 * h2 + bo2);
    % 
    % % Update hidden state (layer 2)
    % h2 = o2 .* tanh(C2)  ;

    % ypred=Wy*h2+by  ;

    ypred=Wy*h1+by  ;
    
    % Store the output of the second LSTM layer
    % outputSequence(:, t) = h2;
    % 
    % ypred=Wy*outputSequence+by 

h1_update=h1;
C1_update=C1 ; 
% h2_update=h2 ; 
% C2_update=C2 ;


% % The final output is the sequence of hidden states from the second LSTM layer
% disp('LSTM Model Output from First Principles (Two LSTM Layers):');
% disp(outputSequence);

% ============================
% Fully Connected Layer Calculation
% ============================

% % For regression, we take the output of the last time step (h2)
% finalHiddenState = h2;  % Output of the last time step from the second LSTM layer
% 
% % Apply fully connected layer
% fcOutput = Wy * finalHiddenState + by;
% 
% % ============================
% % Regression Layer
% % ============================
% % In a regression layer, the output is typically the direct prediction, 
% % so we assume 'fcOutput' is the predicted value
% 
% predictedValue = fcOutput;
% 
% % Display the final predicted value
% disp('Final Predicted Value:');
% disp(predictedValue);

% updatin cell and hidden states



end

% LSTM model prediction ends here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Sigmoid function definition
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end