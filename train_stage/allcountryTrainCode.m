
%用途：用一个2d数据集训练rf全国模型，得出来的结果具有散点可以精度验证

clear all;
clc;
load('F:\数据\PM-nor-全国结果\B-全国rf训练好的模型\所有训练数据2d.mat');
%% 

inputIndex = [7 9 10 11 12 15 16 17 18 19 20 21 ];
outputIndex = [6];
%调用随机森林模型
[R2,PCC,mae,mape,rmse,rfnet,ps_input,ps_output,trueValue,simValue,nonullDataLine,a]=rf(jjjtable2D,inputIndex, outputIndex);


%这里的a是随机数列
function [R2,PCC,mae,mape,rmse,rfnet,ps_input,ps_output,trueValue,simValue,nonullDataLine,a]=rf(trainedData,inputIndex, outputIndex)
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
rfnet = TreeBagger(30,train_input_nor',train_output_nor','OOBPredictorImportance','On','Method','regression','NumPrint','2');




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

trueValue = test_output_accuracy;
simValue = sim_aver;

end