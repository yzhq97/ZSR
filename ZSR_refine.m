function [ L, Corr ] = ZSR_refine(U, V, thres)

    matches = vl_ubcmatch(U', V', thres);
    matches = matches';

    M = size(V, 1);
    N = size(U, 1);
    Ni = size(matches, 1);

    Corr = zeros(Ni, 2);
    for i = 1:Ni
        Corr(matches(i, 2), 1) = matches(i, 1);
        Corr(matches(i, 1), 2) = matches(i, 2);
    end

    L = zeros(M, N);
    for i = 1:M
        for j = 1:N
            if Corr(i, 1) ~= 0 && Corr(j, 2) ~= 0
                L(i, j) = norm(V(i, :) - U(j, :));
            else
                L(i, j) = 0;
            end 
        end
    end

    maxL = max(max(L));
    L = L / maxL;

    for i = 1:M
        for j = 1:N
            if L(i, j) == 0
                L(i, j) = 1;
        end
    end

end