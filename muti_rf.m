clear all;
clc;warning off;

load("把TP中NAN变成0.mat");
load("trend.mat");
% 2. 随机产生训练集和测试集


% ceshi = a;
% b = ceshi(:,13);
% c = ceshi(:,14);
% d = (b.^2+c.^2).^0.5;


%风速度
newDayArray(:,15,:)=nan;
%风向
newDayArray(:,16,:)=nan;


%计算风速
newDayArray(:,15,:) = (newDayArray(:,13,:).^2+ newDayArray(:,14,:).^2).^0.5;

%计算风向
newDayArray(:,16,:)=mod((atan2(newDayArray(:,13,:),newDayArray(:,14,:))*180/pi+360),360);

testTrueDataArray = zeros(2906,2028);
testSimData = zeros(2906,2028);


    %生成机器学习需要的数组
rawdata = [newDayArray(:,6,stationIndex) newDayArray(:,7,stationIndex) newDayArray(:,9,stationIndex) newDayArray(:,10,stationIndex) newDayArray(:,11,stationIndex) newDayArray(:,12,stationIndex) newDayArray(:,15,stationIndex)  newDayArray(:,16,stationIndex) trend];



%  x表示数据无效的行
b=all(~isnan(rawdata),2);

[x,y]= find(b==0);


%构建了一个代表行号的数据列
line = zeros(length(b),1);
for linei = 1:length(b)
    line(linei) = linei;
end

nonullDataLine = setdiff(line,x);


nonullData = rawdata(nonullDataLine,:);


output_data=nonullData(:,1);
input_parameters=nonullData(:,2:11);
%temp = randperm(size(parameters,1));

numpara=size(input_parameters,2);
numdata=size(input_parameters,1);




a=randperm(numdata);
% 训练集——50个样本
train_input= input_parameters(a(1:0.8*numdata),:)';
train_output= output_data(a(1:0.8*numdata),:)';



% 测试集——10个样本
test_input_accuracy = input_parameters(a(0.8*numdata+1:end),:)';
test_output_accuracy = output_data(a(0.8*numdata+1:end),:);
N_accuracy = size(test_input_accuracy,2);



trendline=input_parameters(:,8:10);

%构建一个可以装testinput 的数组
test_input_array=zeros(numpara,1000,numdata);
t = test_input_array(:,:,1);
for timei=1:size(trendline,1)
    b=randperm(numdata,1000);

    test_input=input_parameters(b,1:7);

    mutitrend = repmat(trendline(timei,:),1000,1);

    test_input(:,8:10)=mutitrend;
    test_input=test_input';
    
    
    test_input_array(:,:,timei)=test_input;
% >> A = [1, 2, 3]
% 
% >> B = repmat(A, 10, 1)
% test_input = input_parameters(a(66914:end),:)';
% test_output = output_data(a(53532:end),:);



end
%构建1000个随机数 范围在1-66914内



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
rfnet = TreeBagger(100,train_input_nor',train_output_nor','OOBPredictorImportance','On','Method','regression');




[sim,scores] = predict(rfnet, test_input_nor_accuracy');
% ntree 树的数量
% train_data 训练样本数据
% train_label 训练样本数据对应的类别标签

sim_aver = mapminmax('reverse',sim,ps_output);

error_mae = mean(abs(sim_aver - test_output_accuracy));
error_mape = mean(abs(sim_aver' - test_output_accuracy')/test_output_accuracy');
error_rmse =sqrt(sum((test_output_accuracy-sim_aver).^2)/N_accuracy);
%%
% 2. 决定系数R^2
R2 =(N_accuracy * sum(sim_aver .* test_output_accuracy) - sum(sim_aver) * sum(test_output_accuracy))^2 / ((N_accuracy * sum((sim_aver).^2) - (sum(sim_aver))^2) * (N_accuracy * sum((test_output_accuracy).^2) - (sum(test_output_accuracy))^2)); 
PCC=corr(sim_aver,test_output_accuracy,'type','pearson');

















% 模型性能
figure
plot(oobError(rfnet))
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Classification Error')
rfnet.DefaultYfit = 0;
figure
plot(oobError(rfnet))
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Error Excluding In-Bag Observations')



figure
bar(rfnet.OOBPermutedVarDeltaError)
xticks([1 2 3 4 5 6 7 8 9 10])
xticklabels({'blh','rh','ssr','t2m','tp','ws','wd','juday','week','unix'})
xlabel('Feature Index')
ylabel('Out-of-Bag Feature Importance')
title('Sanming-mda8O3-DAY-RF')



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
disp(info);
fileName = '三明-日均-nda8O3.txt';

fileID = fopen(fileName,'A');

fprintf(fileID,'%s\r\n',info); 

fclose(fileID);

end





    

