function [dxdt, varargout] = ode_set_ph_sys(x, u, d)
    q3 = u(1);
    q2 = d(1);
    Wa4 = x(1);
    Wb4 = x(2);
    A = 207;
    h = 14;
    q1e = 16.6;
    Wa1 = 0.003;
    Wa2 = -0.03;
    Wa3 = -0.00305;
    Wb1 = 0;
    Wb2 = 0.03;
    Wb3 = 5e-5;
    pK1 = 6.35;
    pK2 = 10.25;
    dWa4dt = (1/(h*A))*((q1e*(Wa1 - Wa4)) + (q2*(Wa2 - Wa4)) + (q3*(Wa3 - Wa4)));
    dWb4dt = (1/(h*A))*((q1e*(Wb1 - Wb4)) + (q2*(Wb2 - Wb4)) + (q3*(Wb3 - Wb4)));
    dxdt = [dWa4dt; dWb4dt];
    if (nargout > 1)
        opeqn = @(pH) 1e15*(Wa4 + 10^(pH - 14) - 10^(-pH) + (Wb4*(1 + (2*10^(pH - pK2)))/(1 + 10^(pK1 - pH) + 10^(pH - pK2))));
        options = optimoptions('fsolve','Display','none');
        varargout{1} = fsolve(opeqn, 7, options);
        if (nargout > 2)
            varargout{2} = q3;
        end
    end
end

