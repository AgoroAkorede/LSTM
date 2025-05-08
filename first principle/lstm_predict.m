function yk = lstm_predict(u_k, Nu, Np,h1,C1, h2, C2)

 yk = zeros(1, Np);
 % disp(u_k)
    for i = 1:Np
        if (i == 1)
            u = u_k(1);
        elseif (i <= Nu)
            u = u_k(i);
        else
            u = u_k(Nu);
        end
        %disp(u)
       
        [y_k, h1_update,C1_update,h2_update,C2_update] = lstm_network(u,h1,C1, h2, C2) ;

        h1=h1_update ;C1=C1_update; h2=h2_update ;C2=C2_update ;

         yk(:,i) = y_k ;

       % yk(i) = lstm_network(u(i),h1(:,i),C1(:,i), h2(:,i), C2(:,i))  ;
    end



end


