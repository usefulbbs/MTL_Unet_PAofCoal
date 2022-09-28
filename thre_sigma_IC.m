
clear;
clc;
load MTL_670.mat;
load x_axis.mat;
% load quanshui_5mm_742.mat;

data=MTL_670(1:670,:);

[m,n]=size(data);
% data = quanshui_5mm_742(1:742,:);
index=crossvalind('Kfold',m,670);
data1=data(index,:);
% new_data = data
T1 = [];
for t = 1:3
    ED = [];
    ED1 = [];
    for i = 1:m
        data1(all(data1==0,2),:)=[];
        data(all(data==0,2),:)=[];
        index(all(index==0,2),:)=[];
        xa=data1(i,:);
        xa1=data(i,:);
        sz1=size(xa1);
        sz=size(xa);
        xb=mean(data1);
        ed = dist_E(xa,xb);
        ed1 = dist_E(xa1,xb);
        ED = [ED ed];
        ED1 = [ED1 ed1];
    end
    ED_mean = mean(ED);
    ED_mean1 = mean(ED1);
    ED_sigma = std(ED);
    ED_sigma1 = std(ED1);
    for j=1:m
        if abs(ED(:,j)-ED_mean)>3 * ED_sigma
            data1(j,:)=0;
            r=index(j);
            index(j,:)=0;
            m=m-1;
            T1 = [T1 r];
        end
    end
    min = ED_mean-3 * ED_sigma;
    max = ED_mean+3 * ED_sigma;
    
%     if t == 1
%         x = 1:670;
% 
%         n=length(x);
%         ED11 = ED1(1,1:670);
%         for p = 1:n
%             line([x(p),x(p)], [0,ED11(1,p)])
%             hold on
%             plot([1,670],[max,max]);
%             hold on
% %             plot([1,670],[min,min]);
% %             hold on
%             grid on;
%             xlabel('Coal Sample Number','FontName','Calibri','FontSize',12);
%             ylabel('ED Distance','FontName','Calibri','FontSize',12);
%         end
%     end

    if t==1
        data(431,:)=0;
        data(637,:)=0;
        data(636,:)=0;
        data(613,:)=0;
        data(612,:)=0;
        data(372,:)=0;
        data(371,:)=0;
        data(507,:)=0;
    end
    
%     if t == 2
%         x = 1:662;
% 
%         n=length(x);
%         ED11 = ED1(1,1:662);
%         for p = 1:n
%             line([x(p),x(p)], [0,ED11(1,p)])
%             hold on
%             plot([1,662],[max,max]);
% %             hold on
% %             plot([1,662],[min,min]);
%             hold on
%             grid on;
%             xlabel('Coal Sample Number','FontName','Calibri','FontSize',12);
%             ylabel('ED Distance','FontName','Calibri','FontSize',12);
%         end
%     end
    
    if t==2
        data(505,:)=0;
        data(501,:)=0;
        data(621,:)=0;
        data(50,:)=0;
        data(506,:)=0;
    end
    
    if t == 3
        x = 1:657;
        n=length(x);
        ED11 = ED1(1,1:657);
        for p = 1:n
            line([x(p),x(p)], [0,ED11(1,p)])
            hold on
            plot([1,657],[max,max]);
            hold on
%             plot([1,657],[min,min]);
%             hold on
            grid on;
            xlabel('Coal Sample Number','FontName','Calibri','FontSize',12);
            ylabel('ED Distance','FontName','Calibri','FontSize',12);
        end
    end
    
end
% x = 1:670;
% n=length(x);
% for p = 1:n
%     line([x(p),x(p)], [0,ED1(1,p)], 'color', 'b')
%     hold on
%     plot([1,670],[ED_mean+3 * ED_sigma,ED_mean+3 * ED_sigma], 'color', 'y');
%     hold on
%     plot([1,670],[ED_mean-3 * ED_sigma,ED_mean-3 * ED_sigma], 'color', 'r');
%     hold on
% end
% x = 1:670;
% 
% n=length(x);
% ED11 = ED1(1,1:670);
% for p = 1:n
%     line([x(p),x(p)], [0,ED11(1,p)])
%     hold on
%     plot([1,670],[max,max]);
%     hold on
%     plot([1,670],[min,min]);
%     hold on
%     xlabel('Coal Sample Number','FontName','Calibri','FontSize',12);
%     ylabel('ED Distance','FontName','Calibri','FontSize',12);
% end
