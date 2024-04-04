%%%%DFB光注入数值模拟%%%%%
%数据出自2013年宋健师兄硕士毕业论文
clear all;clc;
set(groot,'defaultLineLineWidth',1)
set(groot,'defaultAxesFontName','Times New Roman')
set(groot,'defaultAxesFontSize',12)
%% 参数设置
w0=1.216e12;           %1550nm
beta0 = 4;             %线宽增强因子
epsilon = 1e-7;        %增益饱和系数
g = 2e4;               %增益系数
tauP = 4.2e-12;        %光子寿命
tauN = 2e-9;           %载流子寿命
N0 = 1.25e8;           %透明载流子浓度
e = 1.6021766208e-19;  %电子电荷
Jth=11e-3/e;         %阈值电流
J = 2.5*Jth;           %偏置电流
%% 时间窗口
h = 1e-12;             %步长，1 ps
nTot = 2^20;           %总取点数
tTot = h*(nTot-1);     %总时间，tTot = 1.048575 us
t = 0:h:tTot;          %总时间窗口
n = length(t);         %时间矢量长度
Fs = 1/h;              %采样频率,Fs = 1 THz
%% 初始值设置
E = zeros(1,n);        %慢变电场存储矩阵
N = zeros(1,n);        %载流子存储矩阵
E(1) = 1e-5;           %慢变电场初始值
N(1) = 0.01;           %载流子初始值
%% 反馈部分
tau = 100e-9;          %反馈时间
gamma = 15e9;           %反馈系数
delayNum=round(tau/h);  %延时tau等价的点数
%% 自发辐射噪声
beta = 1.5e3;            %自发辐射系数
kai = 1;                %噪声开关
Gnoise = randn(1,n);    %均值为0的高斯白噪声
%% 四阶龙格库塔法
for ii=1:n-1
    if ii>delayNum
        key=1;tt=ii-delayNum;
    else
        key=0;tt=1;
    end
    %第一次迭代
    Et = E(ii);  Nt = N(ii);  tn = t(ii);
    Ek1 = 1/2*(1+1i*beta0)*(g*(Nt-N0)/(1+epsilon*abs(Et)^2)-1/tauP)*Et+key*gamma*E(tt)*exp(-1i*w0*tau);
    Nk1 = J-Nt/tauN-(g*(Nt-N0)/(1+epsilon*abs(Et)^2))*(abs(Et)^2);
    %第二次迭代
    Et = E(ii)+h*Ek1/2;  Nt = N(ii)+h*Nk1/2;
    Ek2 = 1/2*(1+1i*beta0)*(g*(Nt-N0)/(1+epsilon*abs(Et)^2)-1/tauP)*Et+key*gamma*E(tt)*exp(-1i*w0*tau);
    Nk2 = J-Nt/tauN-(g*(Nt-N0)/(1+epsilon*abs(Et)^2))*(abs(Et)^2);
    %第三次迭代
    Et = E(ii)+h*Ek2/2;  Nt = N(ii)+h*Nk2/2;
    Ek3 = 1/2*(1+1i*beta0)*(g*(Nt-N0)/(1+epsilon*abs(Et)^2)-1/tauP)*Et+key*gamma*E(tt)*exp(-1i*w0*tau);
    Nk3 = J-Nt/tauN-(g*(Nt-N0)/(1+epsilon*abs(Et)^2))*(abs(Et)^2);
    %第四次迭代
    Et = E(ii)+h*Ek3;  Nt = N(ii)+h*Nk3;
    Ek4 = 1/2*(1+1i*beta0)*(g*(Nt-N0)/(1+epsilon*abs(Et)^2)-1/tauP)*Et+key*gamma*E(tt)*exp(-1i*w0*tau);
    Nk4 = J-Nt/tauN-(g*(Nt-N0)/(1+epsilon*abs(Et)^2))*(abs(Et)^2);
    %下一点值
    E(ii+1) = E(ii)+h*(Ek1+2*Ek2+2*Ek3+Ek4)/6+...
              kai*sqrt(h)*sqrt(2*beta*N(ii))*Gnoise(ii);
    N(ii+1) = N(ii)+h*(Nk1+2*Nk2+2*Nk3+Nk4)/6;
end

figure('Units', 'inches', 'Position', [1 1 8 3]);

%% 时序
I = abs(E).^2;
subplot(1,2,1)
plot(t(end-2^19:end)*1e9,I(end-2^19:end)/1e5,'linewidth',1, 'Color', 'b')
axis([800,820,0e5,12])
xlabel('Time (ns)');  ylabel('Intensity (a.u.)');
text('string', '(a)', 'Units', 'normalized', 'position', [0.05, 0.95], 'FontName', 'Times New Roman', 'FontWeight', 'bold');
%% 相图
% subplot(1,4,2)
% plot(N(end-2^19+1:end),I(end-2^19+1:end),'r');
% xlabel('Carrier number ');  ylabel('Intensity');  title('Phase diagram');
%% 功率谱
I1 = I(end-2^19+1:end);              
NN = length(I1);
Y = 2*abs(h*fft(I1)).^2/(NN*h);
Y(1) = Y(1)/2;
WA_esa = 10*log10(Y(1:NN/2));
freq = (0:NN-1)*Fs/NN;
freq_esa = freq(1:NN/2)*1e-9;
subplot(1,2,2)
plot(freq_esa(10:end),WA_esa(10:end), 'Color', 'b')
axis([0,50,-100,60])
xlabel('Frequency (GHz)'); ylabel('Power (dB)');
text('string', '(b)', 'Units', 'normalized', 'position', [0.05, 0.95], 'FontName', 'Times New Roman', 'FontWeight', 'bold');
%% 光谱
% E_os = E(end-2^19+1:end);             %FFT所用电场
% N_os = length(E_os);                  %电场矢量长度
% Y_os = abs(h*fft(E_os)).^2/(N_os*h);  %功率谱密度函数
% YShif_os = fftshift(Y_os);            %光谱中心化
% YLog_os = 10*log10(YShif_os);         %转化为Dbm
% FNum_os = (0:N_os-1)*Fs/N_os;         %频率
% FUnit_os = (FNum_os-Fs/2)*1e-9;       %所选频率
% subplot(1,4,4)
% plot(FUnit_os,YLog_os,'b');
% axis([-60,60,-130,0])
% xlabel('Frequency [GHz]');  ylabel('Power');  title('Optical Spectrum');
%% 生成随机数
% 8bit量化
I_8bit = I(end-2^19:end);
min_I = 0; max_I = 10e5;
I_8bit = I_8bit - min_I;
I_8bit = I_8bit / max_I;
I_8bit = I_8bit * 255;
I_8bit = round(I_8bit);
I_8bit = I_8bit(I_8bit >= 0);
I_8bit = I_8bit(I_8bit <= 255);
% 画出 histogram
figure;
histogram(I_8bit, 256, 'LineStyle', 'none', 'FaceAlpha', 1.0);

%% 画自相关函数
figure('Units', 'inches', 'Position', [1 1 4 3]);
maxlag = 400000;
[c, lags] = xcorr(I, I(1:2^19), maxlag, 'none');
c = (c-c(maxlag*2));
c = c/c(maxlag+1);
plot((0:maxlag)*1e-3, c(maxlag+1:end), 'b');
xlabel('Lag Time (ns)'); ylabel('ACF');