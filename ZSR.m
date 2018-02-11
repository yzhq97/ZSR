function [ T ] = ZSR(X, Y, U, V, opt)

    %%%%%%%%%%%%%%%%%%%%%%
    % set default values
    %%%%%%%%%%%%%%%%%%%%%%

    if ~isfield(opt,'beta') || isempty(opt.beta), opt.beta = 2; end;
    if ~isfield(opt,'omega') || isempty(opt.omega), opt.omega = 0.7; end;
    if ~isfield(opt,'lambda') || isempty(opt.lambda), opt.lambda = 3; end;
    if ~isfield(opt,'eta') || isempty(opt.eta), opt.eta = 3; end;
    if ~isfield(opt,'nu') || isempty(opt.nu), opt.nu = 2; end;
    if ~isfield(opt,'delta') || isempty(opt.delta), opt.delta = 0.05; end;
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
    lambda = opt.lambda;
    nu = opt.nu;
    delta = opt.delta;
    freq = opt.freq;
    epsilon = opt.epsilon;
    tau = opt.tau - delta;
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
    iter = 1;
    Gaus = GRB(Y, beta);
    A = zeros(M, 2);
    sigma2 = (M*trace(X'*X) + N*trace(Y'*Y) - 2*sum(X)*sum(Y)')/(M*N*2);

    tsampX = atan2(X(:, 2), X(:, 1));
    SCX = sc_compute(X, tsampX, sc_nbins_theta, sc_nbins_r, sc_r_inner, sc_r_outer);

    Q = 0;
    dQ = tolerance + 1;

    %%%%%%%%%%%%%%%%%%%%%%
    % EM algorithm
    %%%%%%%%%%%%%%%%%%%%%%
    
    while (iter <= max_it) && (abs(dQ) > tolerance) && (sigma2 > 1e-4)

        T_old = T;
        Q_old = Q;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % I. CORRESPONDENCE ESTIMATION
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if mod(iter-1, freq) == 0

            %%%%%%%%%%%%%%%%%%%%%%
            % compute refined L
            %%%%%%%%%%%%%%%%%%%%%%
            [ L_tau, Corr ] = ZSR_refine(U, V, tau);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % compute Shape Context Cost 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            meanT = mean(T);
            tsampT = atan2(T(:, 2)-meanT(2), T(:, 1)-meanT(1));
            SCT = sc_compute(T, tsampT, sc_nbins_theta, sc_nbins_r, sc_r_inner, sc_r_outer);
            SC_cost = hist_cost_2(SCT, SCX);

            %%%%%%%%%%%%%%%%%%%%%%
            % compute C_tau
            %%%%%%%%%%%%%%%%%%%%%%

            C_tau = L_tau .* SC_cost;

            tau = tau + delta;

            %%%%%%%%%%%%%%%%%%%%%%%%%
            % compute prior prob. Pm
            %%%%%%%%%%%%%%%%%%%%%%%%%

            [ Hung, Cost ] = hungarian(C_tau);

            for m = 1:M
                for n = 1:N
                    if Hung(m) == n
                        Pm(m, n) = 1;
                    else
                        Pm(m, n) = (1 - epsilon)/N;
                    end
                end
            end

            Pm = Pm ./ sum(Pm);

        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % II. TRANSFORMATION UPDATING
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % use only global constraint
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [ Po, P1, Np, tmp, Q ] = ZSR_compute(X, Y, T_old, Pm, sigma2, omega);
        Q = Q + lambda/2 * trace(A'*Gaus*A);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % now we can update the variables
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        dP = spdiags(P1, 0 , M, M);
        A = (dP * Gaus + lambda * sigma2 * eye(M)) \ (Po * X - dP * Y);
        sigma2 = tmp / (2*Np);
        omega = 1 - (Np/N);
        if omega > 0.99, omega = 0.99; end;
        if omega < 0.01, omega = 0.01; end;
        T = Y + Gaus * A;
        lambda = lambda * 0.95;
        if lambda < 0.1, lambda = 0.1; end;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % visualization
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if (viz == 1)
            disp([ num2str(iter) ' Q: ' num2str(Q) ' dQ= ' num2str(dQ)  ' tau=' num2str(tau) ' sigma2= ' num2str(sigma2) ' lambda=' num2str(lambda) ' omega=' num2str(omega)]);
            CPD_plot(X, T);
        end

        dQ = Q - Q_old;
        iter = iter + 1;

    end

end