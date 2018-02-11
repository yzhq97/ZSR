function [ T ] = ZSR(X, Y, MX, MY, U, V, opt)

    %%%%%%%%%%%%%%%%%%%%%%
    % set default values
    %%%%%%%%%%%%%%%%%%%%%%

    if ~isfield(opt,'beta') || isempty(opt.beta), opt.beta = 2; end;
    if ~isfield(opt,'omega') || isempty(opt.omega), opt.omega = 0.7; end;
    if ~isfield(opt,'lambda') || isempty(opt.lambda), opt.lambda = 3; end;
    if ~isfield(opt,'eta') || isempty(opt.eta), opt.eta = 3; end;
    if ~isfield(opt,'nu') || isempty(opt.nu), opt.nu = 2; end;
    if ~isfield(opt,'delta') || isempty(opt.delta), opt.delta = 0.1; end;
    if ~isfield(opt,'freq') || isempty(opt.freq), opt.freq = 5; end;
    if ~isfield(opt,'epsilon') || isempty(opt.epsilon), opt.epsilon = 0.4; end;
    if ~isfield(opt,'tau') || isempty(opt.tau), opt.tau = 1; end;
    if ~isfield(opt,'K') || isempty(opt.K), opt.K = 5; end;

    if ~isfield(opt,'sc_nbins_theta') || isempty(opt.sc_nbins_theta), opt.sc_nbins_theta = 12; end;
    if ~isfield(opt,'sc_nbins_r') || isempty(opt.sc_nbins_r), opt.sc_nbins_r = 5; end;
    if ~isfield(opt,'sc_r_inner') || isempty(opt.sc_r_inner), opt.sc_r_inner = 1/8; end;
    if ~isfield(opt,'sc_r_outer') || isempty(opt.sc_r_outer), opt.sc_r_outer = 2; end;

    if ~isfield(opt,'max_it') || isempty(opt.max_it), opt.max_it = 100; end;
    if ~isfield(opt,'tolerance') || isempty(opt.tolerance), opt.tolerance = 1e-3; end;
    if ~isfield(opt,'viz') || isempty(opt.viz), opt.viz = 0; end;

    %%%%%%%%%%%%%%%%%%%%%%
    % get parameter
    %%%%%%%%%%%%%%%%%%%%%%

    beta = opt.beta;
    omega = opt.omega;
    nu = opt.nu;
    delta = opt.delta;
    freq = opt.freq;
    epsilon = opt.epsilon;
    tau = opt.tau;
    K = opt.K;

    sc_nbins_theta = opt.sc_nbins_theta;
    sc_nbins_r = opt.sc_nbins_r;
    sc_r_inner = opt.sc_r_inner;
    sc_r_outer = opt.sc_r_outer;

    max_it = opt.max_it;
    tolerance = opt.tolerance;
    viz = opt.viz;

    %%%%%%%%%%%%%%%%%%%%%%
    % initialization
    %%%%%%%%%%%%%%%%%%%%%%

    N = size(X, 1);
    M = size(Y, 1);

    if (M~=N)
        error('ZSR: X and Y should be correspondent!');
    end

    T = Y;
    iter = 0;
    Gaus = GRB(Y, beta);
    A = zeros(M, 2);
    sigma2 = (M*trace(X'*X) + N*trace(Y'*Y) - 2*sum(X)*sum(Y)')...
     / (M*N*2);

    tsampX = atan2(X(:, 2), X(:, 1));
    SCX = sc_compute(X, tsampX...
        sc_nbins_theta, sc_nbins_r, sc_r_inner, sc_r_outer);
    L = pairwiseDistance2(U, V);

    Q = 0;
    dQ = tolerance + 1;

    %%%%%%%%%%%%%%%%%%%%%%
    % EM algorithm
    %%%%%%%%%%%%%%%%%%%%%%

    while (iter < max_it) && (abs(dQ) > epsilon)

        T_old = T;
        Q_old = Q;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % I. CORRESPONDENCE ESTIMATION
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if (mod(iter, freq) == 0)

            %%%%%%%%%%%%%%%%%%%%%%
            % compute refined L
            %%%%%%%%%%%%%%%%%%%%%%
            Lr = L;
            for m = 1:M
                for n = 1:N
                    if MY(m) < tau || MX(n) < tau
                        Lr(m, n) = 1;
                end
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % compute Shape Context Cost 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            meanT = mean(T);
            tsampT = atan2(T(:, 2)-meanT(2), T(:, 1)-meanT(1));
            SCT = sc_compute(T, tsampT...
                sc_nbins_theta, sc_nbins_r, sc_r_inner, sc_r_outer);
            SC_cost = hist_cost_2(SCT, SCX);

            %%%%%%%%%%%%%%%%%%%%%%
            % compute C_tau
            %%%%%%%%%%%%%%%%%%%%%%

            C_tau = Lr .* SC_cost;

            %%%%%%%%%%%%%%%%%%%%%%%%%
            % compute prior prob. Pm
            %%%%%%%%%%%%%%%%%%%%%%%%%

            [ Corr, hungarian_cost ] = hungarian(C_tau);
            Corr = Corr';
            Xp = X(Corr, :);

            Pm = zeros(M, N);
            for m = 1:M
                for n=1:N
                    if Corr(m) == n
                        Pm(m, n) = 1;
                    else
                        Pm(m, n) = (1 - epsilon)/N;
                    end
                end
            end

            tau = tau + delta;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % II. TRANSFORMATION UPDATING
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        T = Y + Gaus * A;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % compute posterior prob. Po
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Po = zeros(M, N);

        phi = zeros(M, N);
        for m = 1:M
            for n = 1:N
                xn = X(n, :);
                tm = T_old(m, :);
                phi(m, n) = (1-omega) * Pm(m, n) *...
                exp( -(1/(2*sigma2)) * sum( norm(xn-tm)^2 ) );
            end
        end

        Pxn = sum(phi, 1) + omega/N;

        for m = 1:M
            for n = 1:N
                Po(m, n) = phi(m, n)/Pxn(n);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % prepare WX and WY
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        WX = zeros(N, N);
        for i = 1:N
            for j = 1:N
                if MX(i) >= tau && MX(j) >= tau
                    WX(i, j) = ...
                    exp( -1/nu * norm( X(i, :) - X(j, :) )^2 );
                else
                    WX(i, j) = 0;
                end
            end
        end

        WY  =zeros(M, M);
        for i = 1:M
            for j = 1:M
                if MY(i) >= tau && MY(j) >= tau
                    WY(i, j) = ...
                    exp( -1/nu * norm( X(i, :) - X(j, :) )^2 );
                else
                    WX(i, j) = 0;
                end
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % compute intermediate vars, and Q
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Id = eye(M);
        Np = ones(1, M) * Po * ones(N, 1);
        dP = diag(Po * ones(N, 1));
        dPt = diag(Po' * ones(N, 1));
        H = WY - K*Id;
        G = (WX - K*Id) * X - (WY - K*Id) * Y;
        Lc = (WX - K*Id) * T - (WY - K*Id) * X;
        Q_gmm = trace(X'*dPt*X) +...
        2*trace(T'*Po*X) + trace(T'dP*T);

        Q = Q_gmm/(2*sigma2) +...
        -1*Np*log(1-omega) +...
        -1*(N-Np)*log(omega) +...
        Np * log(sigma2) +...
        (lambda/2) * trace(A'*Gaus*A) +...
        (eta/2) * 

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % now we can update the variables
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        A = (dP*Gaus + lambda*sigma2*Id + eta*sigma2*H'*H*Gaus)\...
        (Po*X - dP*Y + eta*sigma2*H'*G);
        sigma2 = Q_gmm / (2*Np);
        omega = 1 - (Np/N);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % visualization
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if (viz == 1)
            Q = 
        end

    end

end