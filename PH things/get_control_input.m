function [u,y_pred,cost] = get_control_input(y_set,y_present,y_past,u_past,d,tp,ts,e,nn_type)
    % e = 0;
    data = struct();
    data.nn_type = nn_type;
    data.u_past = u_past;
    data.y_past = y_past;
    data.y_present = y_present;
    data.ts = ts;
    data.d = d;
    data.e = e;
    data.y_set = y_set;
    data.Nu = tp(1);
    data.Np = tp(2);
    data.W_y = tp(3);
    data.W_du = tp(4);
    data.y_min = 3;
    data.y_max = 11;
    data.du_min = -5;
    data.du_max = 5;
    data.u_min = 0;
    data.u_max = 50;
    
    %Converts u_min and u_max to a column vector with a dimension equal
    %to the magnitude of the control horizon. These vectors will be used
    %by the optimization function
    data.u_min_colvector = data.u_min*ones(data.Nu,1);
    data.u_max_colvector = data.u_max*ones(data.Nu,1);

    data.u_initial_colvector = u_past*ones(data.Nu,1);
    
    OPTIONS = optimoptions('fmincon','Algorithm','sqp','Display','final', 'FiniteDifferenceStepSize', 1e-5);
    [u_k,cost] = fmincon(@(u)compute_cost(u,data),data.u_initial_colvector,[],[],[],[],data.u_min_colvector,data.u_max_colvector,@(u)nlcf(u,data),OPTIONS);

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
    
    u  = u_k(1);
    
%     disp("u");
%     disp(u);
%     ys_pred = nn_predict(data.y_past,data.y_present,data.u_past,u,data.Nu,data.Np,0,false);
%     disp("ys_pred");
%     disp(mat2str(ys_pred));

    
    % Update global network state
    if (data.nn_type == 1)
        y_pred = nn_simple_predict(u,1,1,0,true);
    else
        y_pred = nn_predict(data.y_past,data.y_present,data.u_past,u,1,1,0,true);
    end
        
    % Compensate for error
    % y_pred = y_pred + data.e;
end

function cost = compute_cost(u,data)
    %This function returns the cost or performance index, J of implementing a
    %vector u of control moves on the plant.
    Nu = data.Nu;
    Np = data.Np;
    %Input rate vector with a dimension of Nu
    du = zeros(Nu,1);
    du(1) = u(1) - data.u_past;
    for i = 2:Nu
        du(i) = u(i) - u(i-1);
    end
    
    cost_du = du'*data.W_du*du;
    
    if (data.nn_type == 1)
        y_k = nn_simple_predict(u,Nu,Np,0,false);
    else
        y_k = nn_predict(data.y_past,data.y_present,data.u_past,u,Nu,Np,0,false);
    end
    
    % Compensate for error
    y_k = y_k + data.e;
    
    cost = cost_du + sum(data.W_y*((y_k - data.y_set).^2));
end

function [c,ceq] = nlcf(u,data)
    %This evaluates the nonlinear inequality constraints which are bound
    %to the problem. nlcf returns two major parameters: c and ceq. c is nonlinear
    %inequality constraint function while ceq is the nonlinear equality constraint
    %function.

    Nu = data.Nu;
    Np = data.Np;

    y_min_colvector = data.y_min*ones(Np,1);
    y_max_colvector = data.y_max*ones(Np,1);
    du_min_colvector = data.du_min*ones(Nu,1);
    du_max_colvector = data.du_max*ones(Nu,1);
    u_min_colvector = data.u_min_colvector(1:Nu);
    u_max_colvector = data.u_max_colvector(1:Nu);

    if (data.nn_type == 1)
        y_k = nn_simple_predict(u,Nu,Np,0,false);
    else
        y_k = nn_predict(data.y_past,data.y_present,data.u_past,u,Nu,Np,0,false);
    end
    
    % Compensate for error
    y_k = y_k + data.e;
    
    %Input rate vector with a dimension of Nu
    du = zeros(Nu,1);
    du(1) = u(1) - data.u_past;
    for i = 2:Nu
        du(i) = u(i) - u(i-1);
    end
    
    ineq_du_min = du_min_colvector - du;
    ineq_du_max = du - du_max_colvector;
    ineq_y_min = y_min_colvector - y_k';
    ineq_y_max = y_k' - y_max_colvector;
    ineq_u_min = u_min_colvector - u;
    ineq_u_max = u - u_max_colvector;
    
    ceq = [];
    
    c = [ineq_du_min; ineq_du_max; ineq_y_min; ineq_y_max; ineq_u_min; ineq_u_max];
    
    for i = 1:numel(c)
        if (c(i) == inf)
            c(i) = 1e23;
        end
        if (c(i) == -inf)
            c(i) = -1e23;
        end
    end
end