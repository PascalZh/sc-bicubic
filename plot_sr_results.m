% 按照2x3矩阵依次画出HR图、LR图、两张SR图、两张SR图的绝对误差，按顺序下面标a,b,c,d,e,f
clear all;clc;
set(groot,'defaultLineLineWidth',1)
set(groot,'defaultAxesFontName','Times New Roman')
set(groot,'defaultAxesFontSize',12)

% 读取图片
hr = imread('Lena.bmp');
lr = imread('LR.bmp');

sr1 = imread('sc_bicubic_sr-stochastic-bsl_512.bmp');
sr2 = imread('sc_bicubic_sr-stochastic-bsl_1024.bmp');
sr3 = imread('sc_bicubic_sr-binary.bmp');

% 转换成灰度图
hr = rgb2gray(hr);
lr = rgb2gray(lr);
sr1 = rgb2gray(sr1);
sr2 = rgb2gray(sr2);
sr3 = rgb2gray(sr3);

abs_err1 = abs(sr1 - hr);
abs_err2 = abs(sr2 - hr);

% 画图
figure('Units', 'inches', 'Position', [1 1 12 8]);
t = tiledlayout(2,3);
nexttile(1)
imshow(hr)
title('(a)')

nexttile(2)
imshow(lr)
title('(b)')

nexttile(3)
imshow(sr1)
title('(c)')

nexttile(4)
imshow(sr2)
title('(d)')

ax = nexttile(5)
imshow(abs_err1)
colormap(ax, jet)
colorbar
title('(e)')

ax = nexttile(6)
imshow(abs_err2)
colormap(ax, jet)
colorbar
title('(f)')

t.TileSpacing = 'none';
t.Padding = 'none';

% 计算 PSNR
psnr1 = psnr(sr1, hr)
psnr2 = psnr(sr2, hr)
psnr3 = psnr(sr3, hr)

% 读取图片
sr1 = rgb2gray(imread('sc_bicubic_sr-hybrid-4high4low-bsl_64.bmp'));
sr2 = rgb2gray(imread('sc_bicubic_sr-hybrid-4high4low-bsl_128.bmp'));
sr3 = rgb2gray(imread('sc_bicubic_sr-hybrid-4high4low-bsl_256.bmp'));
sr4 = rgb2gray(imread('sc_bicubic_sr-hybrid-4high4low-bsl_512.bmp'));


% 画图
figure('Units', 'inches', 'Position', [1 1 12 8]);
t = tiledlayout(2,3);
nexttile(1)
imshow(hr)
title('(a)')

nexttile(2)
imshow(lr)
title('(b)')

nexttile(3)
imshow(sr1)
title('(c)')

nexttile(4)
imshow(sr2)
title('(d)')

nexttile(5)
imshow(sr3)
title('(e)')

ax = nexttile(6)
imshow(sr4)
title('(f)')

t.TileSpacing = 'none';
t.Padding = 'none';

% 计算 PSNR
psnr1 = psnr(sr1, hr)
psnr2 = psnr(sr2, hr)
psnr3 = psnr(sr3, hr)
psnr4 = psnr(sr4, hr)