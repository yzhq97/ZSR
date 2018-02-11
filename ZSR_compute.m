function [ Po, P1, Np, t, Q ] = ZSR_compute ( X, Y, T_old, Pm, sigma2, omega )
    
% -----------------------------------------------------------------------------
% input
% -----------------------------------------------------------------------------
% X: the original X
% Y: the original Y
% T_old: T from last iteration
% omega: the outliers weight
% sigma2: sigma squared as in the equation
% -----------------------------------------------------------------------------
%
% -----------------------------------------------------------------------------
% output
% -----------------------------------------------------------------------------
% Po: Po(m, n) is P_old(m|xn)
% P1: P1 = Po * ones(N, 1);
% Np: Np as in the paper
% T: new T
% t: t is a special useful term
% Q: Q value of the EM algorithm
% -----------------------------------------------------------------------------

    [ N, D ] = size(X);
    [ M, D ] = size(Y);
    T = T_old;

    % compute Po
    Po = zeros(M, N);

    Pmxn = zeros(M, N);
    for m = 1:M
        for n = 1:N
            xn = X(n, :);
            tm = T_old(m, :);
            Pmxn(m, n) = (1-omega) * Pm(m, n) * exp( -(1/(2*sigma2)) * sum( norm(xn-tm)^2 ) );
        end
    end

    Pxn = sum(Pmxn, 1) + omega/N;

    for m = 1:M
        for n = 1:N
            Po(m, n) = Pmxn(m, n)/Pxn(n);
        end
    end
    
    % compute Np
    Np = ones(1, M) * Po * ones(N, 1);
    
    % compute P1
    P1 = Po * ones(N, 1);

    % compute Q
    Px = diag(Po' * ones(M, 1));
    Py = diag(P1);
    t = trace(X' * Px * X) - 2 * trace(T' * Po * X) + trace(T' * Py * T);
    Q = Np * D * log(sigma2) / 2 + t/(2*sigma2);

end