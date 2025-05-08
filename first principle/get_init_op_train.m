function [x_0, varargout] = get_init_op_train()
   x_0 = [-4.32e-4; 5.28e-4];
   u_0 = 15.6;

   % x_0 = [0.001; 7.901e-4];
   % u_0 = 3.98;
   
   d_0 = 0.55;
   if (nargout > 1)
       varargout{1} = u_0;
   end
   if (nargout > 2)
       varargout{2} = d_0;
   end
   if (nargout > 3)
      [~, y_0] = ode_set(x_0, u_0, d_0);
      varargout{3} = y_0;
   end
end