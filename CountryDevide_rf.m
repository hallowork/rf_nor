clear all;
clc;warning off;
load("mda8O3_day.mat");
load("把TP中NAN变成0.mat");
load("trend_2.mat");
load("hangIndexArray.mat");
load("longti_lati.mat");

% 2. 随机产生训练集和测试集

hangIndexArray = (1:2028)';


%风速度
newDayArray(:,15,:)=nan;
%风向
newDayArray(:,16,:)=nan;


%计算风速
newDayArray(:,15,:) = (newDayArray(:,13,:).^2+ newDayArray(:,14,:).^2).^0.5;

%计算风向
newDayArray(:,16,:)=mod((atan2(newDayArray(:,13,:),newDayArray(:,14,:))*180/pi+360),360);

ArraytestTrueData = zeros(2906,2028)*nan;
ArraytestSimData = zeros(2906,2028)*nan;
Arraypcc = zeros(2028,1)*nan;
Arrayr2 = zeros(2028,1)*nan;
Arraymae = zeros(2028,1)*nan;
Arraymape = zeros(2028,1)*nan;
Arrayrmse = zeros(2028,1)*nan;
ArraytruePm = zeros(2906,2028)*nan;
ArraynorPm = zeros(2906,2028)*nan;
ArrayImportance= zeros(10,2028)*nan;

co_data =newDayArray(:,3,:);
co_data(co_data>=5) = nan;
co_data(co_data<=0) = nan;
co_data2D = squeeze(co_data);
newDayArray(:,3,:) = co_data;



PM25_data =newDayArray(:,6,:);
PM25_data(PM25_data>=800) = nan;
PM25_data(PM25_data<=0) = nan;
PM25_data2D = squeeze(PM25_data);
newDayArray(:,6,:) = PM25_data;


%O3改成mda8
newDayArray(:,4,:)=mda8O3Array;
mda8o3_data =newDayArray(:,4,:);
mda8o3_data(mda8o3_data>=800) = nan;
mda8o3_data(mda8o3_data<=0) = nan;
mda8o3_data2D = squeeze(mda8o3_data);
newDayArray(:,4,:) = mda8o3_data;

blh_data =newDayArray(:,7,:);
blh_data(blh_data>=3500) = nan;
blh_data(blh_data<=0) = nan;
blh_data2D = squeeze(blh_data);
newDayArray(:,7,:) = blh_data;

r_data =newDayArray(:,9,:);
r_data(r_data>100) = nan;
r_data(r_data<=0) = nan;
r_data2D = squeeze(r_data);
newDayArray(:,9,:) = r_data;

ssr_data =newDayArray(:,10,:);
ssr_data(ssr_data<0) = nan;
ssr_data2D = squeeze(ssr_data);
newDayArray(:,10,:) = ssr_data;


t2m_data =newDayArray(:,11,:);
t2m_data2D = squeeze(t2m_data);
newDayArray(:,11,:) = t2m_data;


tp_data =newDayArray(:,12,:);
% tp_data(tp_data>0.015) = nan;
tp_data(tp_data<0) = nan;
tp_data2D = squeeze(tp_data);
newDayArray(:,12,:) = tp_data;

%把经纬度放进3D数据中
threeD_longti_lati = zeros(2906,2,2028);
for longti_lati_index = 1:2906
   threeD_longti_lati(longti_lati_index,:,:)  = longti_lati';
    
end
newDayArray(:,17:18,:) = threeD_longti_lati;

%把trend时间放进3D数据中

threeD_trend = zeros(2906,size(trend,2),2028);
for trend_index = 1:2028
   threeD_trend(:,:,trend_index)  = trend;
    
end


newDayArray(:,19:21,:) = threeD_trend;
    



%通过函数把3维数据转化成二维数据
jjjtable2D  = threeDto2D3(hangIndexArray,newDayArray);



datanum = length(hangIndexArray);
%复制trend,多次
mutitrend = repmat(trend,datanum,1);
    %1	     2	3	4	5	     6          7+	8	 9+	 10+	11+	12+	13  14	15+	16+
    %SO2	NO2	CO	O3	PM10	PM2.5       blh	msl	 r	 ssr	t2m	tp	u10	v10	ws	wd
    %17  18  19  20  21
    %lon la   trend
inputIndex = [7 9 10 11 12 15 16 17 18 19 20 21 ];
outputIndex = [3];
%调用随机森林模型
[R2,PCC,mae,mape,rmse,rfnet,ps_input,ps_output]=rf(jjjtable2D,inputIndex, outputIndex);

trendline=input_parameters(:,8:10);

%构建一个可以装testinput 的数组
test_input_array=zeros(numpara,300,numdata);
t = test_input_array(:,:,1);
for timei=1:size(trendline,1)
    b=randperm(numdata,300);

    test_input=input_parameters(b,1:7);

    mutitrend = repmat(trendline(timei,:),300,1);

    test_input(:,8:10)=mutitrend;
    test_input=test_input';
    
    
    test_input_array(:,:,timei)=test_input;
% >> A = [1, 2, 3]
% 
% >> B = repmat(A, 10, 1)
% test_input = input_parameters(a(66914:end),:)';
% test_output = output_data(a(53532:end),:);



end
%构建300个随机数 范围在1-66914内

















%构建一个数组用来存归一化后的PM
nor_pm=zeros(size(trendline,1),1);


for timej=1:size(trendline,1)
    
test_input=test_input_array(:,:,timej);
N = size(test_input,2); 
test_input_nor = mapminmax('apply',test_input,ps_input);

[sim,scores] = predict(rfnet, test_input_nor');


sim_aver = mapminmax('reverse',sim,ps_output);

mean_pm=mean(sim_aver);   


nor_pm(timej)=mean_pm;

info=strcat(num2str(mean_pm),"$",num2str(timej));
% disp(info);





fileName = '三明-日均-nda8O3.txt';

fileID = fopen(fileName,'A');

fprintf(fileID,'%s\r\n',info); 

fclose(fileID);
end


 
function [R2,PCC,mae,mape,rmse,rfnet,ps_input,ps_output]=rf(trainedData,inputIndex, outputIndex)
rawdata = trainedData(:,([outputIndex inputIndex])) ;
    %1	     2	3	4	5	     6          7+	8	 9+	 10+	11+	12+	13  14	15+	16+
    %SO2	NO2	CO	O3	PM10	PM2.5       blh	msl	 r	 ssr	t2m	tp	u10	v10	ws	wd
    %17  18  19  20  21
    %lon la   trend
%  x表示数据无效的行
b=all(~isnan(rawdata),2);
[x,y]= find(b==0);


%构建了一个代表行号的数据列
line = zeros(length(b),1);
for linei = 1:length(b)
    line(linei) = linei;
end

%非空的数据的行号
nonullDataLine = setdiff(line,x);
nonullData = rawdata(nonullDataLine,:);

output_data=nonullData(:,1);
input_parameters=nonullData(:,2:end);
%temp = randperm(size(parameters,1));


numdata=size(input_parameters,1);
a=randperm(numdata);

% 训练集——50个样本
train_input= input_parameters(a(1:0.8*numdata),:)';
train_output= output_data(a(1:0.8*numdata),:)';

% 测试集——10个样本
test_input_accuracy = input_parameters(a(0.8*numdata+1:end),:)';
test_output_accuracy = output_data(a(0.8*numdata+1:end),:);
N_accuracy = size(test_input_accuracy,2);


[train_input_nor, ps_input] = mapminmax(train_input,0,1);
test_input_nor_accuracy = mapminmax('apply',test_input_accuracy,ps_input);
[train_output_nor,ps_output] = mapminmax(train_output,0,1);
%% IV. BP神经网络创建、训练及仿真测试
% 1. 创建网络
extra_options.importance=1;

% 随机森林1
% rfnet=classRF_train(train_input_nor',train_output_nor',10,2 ,extra_options);
% [sim, votes, prediction_per_tree] = classRF_predict(test_input_nor',rfnet);
% 随机森林2
rfnet = TreeBagger(200,train_input_nor',train_output_nor','OOBPredictorImportance','On','Method','regression','NumPrint','2');




[sim,scores] = predict(rfnet, test_input_nor_accuracy');
% ntree 树的数量
% train_data 训练样本数据
% train_label 训练样本数据对应的类别标签

sim_aver = mapminmax('reverse',sim,ps_output);

mae = mean(abs(sim_aver - test_output_accuracy));
mape = mean(abs(sim_aver' - test_output_accuracy')/test_output_accuracy');
rmse =sqrt(sum((test_output_accuracy-sim_aver).^2)/N_accuracy);
%%
% 2. 决定系数R^2
R2 =(N_accuracy * sum(sim_aver .* test_output_accuracy) - sum(sim_aver) * sum(test_output_accuracy))^2 / ((N_accuracy * sum((sim_aver).^2) - (sum(sim_aver))^2) * (N_accuracy * sum((test_output_accuracy).^2) - (sum(test_output_accuracy))^2)); 
PCC=corr(sim_aver,test_output_accuracy,'type','pearson');



end


function table2D  = threeDto2D3(hangIndexArray,targetArray)
    %二维后的表格  = threeDto2D3把第三维度展开(所有符合条件的行数的记号,被二维化的目标三维数组)
    
    length3D = length(hangIndexArray);
    %三维数组的层数
    
    table3D = targetArray(:,:,hangIndexArray);
    %获取3维数组范围
    
    timetrendLength = length(targetArray);
    %获取每层数据的时间项数据行行数
    
    for Index = 0:length3D-1
        table2D(Index*timetrendLength+1:Index*timetrendLength+timetrendLength,:)=table3D(:,:,Index+1);
        
    end
    
    

    
    
end



% leaf =5;
% ntrees = 200;
% fboot = 1;