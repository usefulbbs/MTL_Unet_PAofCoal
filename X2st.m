function [Yxx] = X2st(data)
%X1ST 此处显示有关此函数的摘要
%   此处显示详细说明
m=size(data,1);
Yx=[];
Yxx=[];

for j=1:m

    h=6.194;
    a=908.1;b=1676.2;
    x=a:h:b;
    n=length(x);
    y=data(j,:);
    % hold on
    %grid on
    yx=zeros(1,n);
    yxx=zeros(1,n);
    for i=2:n-1
        yx(i-1)=(y(i+1)-y(i-1))/(2*h);
        yxx(i-1)=(y(i+1)+y(i-1)-2*y(i))/h^2;
    end
     %Yx=[Yx;yx];
     Yxx=[Yxx;yxx];
end
end