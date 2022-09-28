function [B2] = SG(data)
B1=[];
B2=[];
[m,n]=size(data);
for j=1:m
    
    N_window = 7;
    h=6.194;
    t=908.1:h:1676.2;
    A=data(j,:);
    b1 = smoothdata(A,'sgolay' ,N_window);
%     figure(1)
%     plot(t,A,t,b1)
    B1=[B1;b1];
    b2 = smooth_SG_hyh(A,7,1,2);  
%     figure(2)
%     plot(t,b1,t,b2)
    B2=[B2;b2];
end
end

