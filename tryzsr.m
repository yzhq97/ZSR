clear;
clc;

disp('install vlfeat...')
tic
run('../vlfeat/toolbox/vl_setup');
toc

I1 = imread('/Users/yzhq/Code/MATLAB/data/RemoteSense/ANGLE/68/2.jpg');
I2 = imread('/Users/yzhq/Code/MATLAB/data/RemoteSense/ANGLE/68/13.jpg');

k = 1;
imheight = size(I1, 1)/k;
imlength = size(I1, 2)/k;
I1 = imresize(I1, [ imheight, imlength ] );
I2 = imresize(I2, [ imheight, imlength ] );

I1 = im2double(I1);
I2 = im2double(I2);

I1g = single(rgb2gray(I1));
I2g = single(rgb2gray(I2));

[ p1, d1 ] = vl_sift(I1g);
[ p2, d2 ] = vl_sift(I2g);

matches = vl_ubcmatch(d1, d2);
matches = matches';

p1 = p1'; p2 = p2'; d1 = d1'; d2 = d2';

X = p1(matches(:, 1), 1:2);
Y = p2(matches(:, 2), 1:2);
U = double(d1(matches(:, 1), :));
V = double(d2(matches(:, 2), :));

meanX = mean(X);
sigmaX = std2(X);
meanY = mean(Y);
sigmaY = std2(Y);
X = (X - meanX) / sigmaX;
Y = (Y - meanY) / sigmaY;


opt.omega = 0.2;
opt.lambda = 6;
opt.viz = 1;
opt.epsilon = 0.7;
opt.tau = 1;
opt.delta = 0.05;
opt.freq = 5;

T = ZSR(X, Y, U, V, opt);

X = X * sigmaX + meanX;
Y = Y * sigmaY + meanY;
T = T * sigmaX + meanX;

disp('warping...');
tic
C = findNeighbours(T, X);
P1 = [ Y(: ,2), Y(:, 1), zeros(size(Y, 1), 1) ];
P2 = [ X(C(:, 1), 2), X(C(:, 1), 1), zeros(size(X, 1), 1) ];
Iw = tps_warp(I2, P1, P2, 'bicubic');
toc
Ic = immontage(I1, Iw, 9);

subplot(1, 3, 1);
imshow(I1);

subplot(1, 3, 2);
imshow(Iw);

subplot(1, 3, 3);
imshow(Ic);


