function  [delt1,y3]=autocorrelation1(ET1,h,t)
%%%%%%%%%L1,L2ֻ��Ϊ��ȥ����̬�ĵ㣬����̬��ʼ��������أ�����س���Ϊ60��230ns
      L1=max(find(t<=100e-9));%62 
      L2=max(find(t<120e-9));%111
      
     tt= -10e-009:h:10e-009;%�������ʱ�����ƶ��ĳ��ȣ��������ɨ�賤��
      L=length(tt)-1;
      A1=ET1(L1:L2).^2;
%       Ar1=Ar(L1:L2).^2;
      for i=1:L+1
          A2=ET1((L1+i-L/2-1):(L2+i-L/2-1)).^2;
          xx1=mean((A1-mean(A1)).*(A2-mean(A2)));
          xx21=mean((A1-mean(A1)).^2);
          xx22=mean((A2-mean(A2)).^2);
          xx2=sqrt(xx21.*xx22);
          pp=xx1/xx2;
          y(i)=pp;
     end
     delt1=tt;
     y3=y;
