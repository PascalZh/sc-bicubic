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
Jth=8.5565e16;         %阈值电流
J = 2*Jth;           %偏置电流
%% 时间窗口
h = 1e-12;             %步长，1 ps
nTot = 2^18;           %总取点数
tTot = h*(nTot-1);     %总时间，tTot = 1.048575 us
t = 0:h:tTot;          %总时间窗口
n = length(t);         %时间矢量长度
Fs = 1/h;              %采样频率,Fs = 1 THz
%% 初始值设置
e = 1.6021766208e-19;  %电子电荷

Ps= zeros(1,21);       %功率存储矩阵
for I_ = 0:20
J = (I_ +0.1) * 1e-3 / e;

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
kai = 0;                %噪声开关
Gnoise = randn(1,n);    %均值为0的高斯白噪声
%% 四阶龙格库塔法
for ii=1:n-1
    if ii>delayNum
        key=1;tt=ii-delayNum;
    else
        key=0;tt=1;
    end
    key = 0;
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

% 计算平均功率
E = E(end-2^17:end);
I = abs(E).^2;
P = sum(I) * h / tTot;
Ps(I_+1) = P;
end

%% 功率-电流曲线
figure('Units', 'inches', 'Position', [1 1 4 3]);
I_ = 0:20;
Ps(1:5) = 0;
plot(I_, Ps / 1e3, '--bo', 'Color', 'b');
xlabel('Current (mA)', 'FontWeight', 'bold'); ylabel('Power (a.u.)', 'FontWeight', 'bold');
