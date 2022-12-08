clear all;
clc;warning off;

load("把TP中NAN变成0.mat");
load("trend_2.mat");
load("StationName.mat");
load("mda8O3_day.mat");
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

% so2_data =newDayArray(:,1,:);
% so2_data(so2_data>=800) = nan;
% so2_data(so2_data<=0) = nan;
% so2_data2D = squeeze(so2_data);
% newDayArray(:,1,:) = so2_data;
% 
% 
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



for stationIndex = [704]
    %1	     2	3	4	5	     6          7+	8	 9+	 10+	11+	12+	13  14	15+	16+
    %SO2	NO2	CO	O3	PM10	PM2.5       blh	msl	 r	 ssr	t2m	tp	u10	v10	ws	wd

    %生成机器学习需要的数组
try    
rawdata = [newDayArray(:,3,stationIndex) newDayArray(:,7,stationIndex) newDayArray(:,9,stationIndex) newDayArray(:,10,stationIndex) newDayArray(:,12,stationIndex) newDayArray(:,12,stationIndex) newDayArray(:,15,stationIndex)  newDayArray(:,16,stationIndex) trend];




%  x表示数据无效的行
b=all(~isnan(rawdata),2);

[x,y]= find(b==0);


%构建了一个代表行号的数据列
aline = zeros(length(b),1);
for linei = 1:length(b)
    aline(linei) = linei;
end


%非空的数据的行号
nonullDataLine = setdiff(aline,x);


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
test_input_array=zeros(numpara,300,numdata);
t = test_input_array(:,:,1);
for timei=1:size(trendline,1)
    b=randperm(numdata,300);


%构建一个可以装testinput 的数组
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


ArraytestTrueData(1:size(test_output_accuracy,1),stationIndex) = test_output_accuracy;
ArraytestSimData(1:size(test_output_accuracy,1) ,stationIndex) = sim_aver;


Arraypcc(stationIndex) =PCC ;
Arrayr2(stationIndex) = R2;
Arraymae(stationIndex) = error_mae;
Arraymape(stationIndex) = error_mape;
Arrayrmse(stationIndex) = error_rmse;
ArrayImportance(:,stationIndex)=rfnet.OOBPermutedVarDeltaError;








% % % % % % % % % 
% % % % % % % % % 
% % % % % % % % % % 模型性能
% % % % % % % % % figure
% % % % % % % % % plot(oobError(rfnet))
% % % % % % % % % xlabel('Number of Grown Trees')
% % % % % % % % % ylabel('Out-of-Bag Classification Error')
% % % % % % % % % rfnet.DefaultYfit = 0;
% % % % % % % % % figure
% % % % % % % % % plot(oobError(rfnet))
% % % % % % % % % xlabel('Number of Grown Trees')
% % % % % % % % % ylabel('Out-of-Bag Error Excluding In-Bag Observations')
% % % % % % % % % 
% % % % % % % % % 
% % % % % % % % % 
% % % % % % % % % figure
% % % % % % % % % bar(rfnet.OOBPermutedVarDeltaError)
% % % % % % % % % xticks([1 2 3 4 5 6 7 8 9 10])
% % % % % % % % % xticklabels({'blh','rh','ssr','t2m','tp','ws','wd','juday','week','unix'})
% % % % % % % % % xlabel('Feature Index')
% % % % % % % % % ylabel('Out-of-Bag Feature Importance')
% % % % % % % % % title('Sanming-mda8O3-DAY-RF')



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

ArraytruePm(:,stationIndex) = rawdata(:,1);
ArraynorPm(nonullDataLine,stationIndex) = nor_pm;

 detectBegin = 720;
 detectEnd = 1200;
thisTime = time(detectBegin:detectEnd);
datenn = datenum(time);
datenn=datenn(detectBegin:detectEnd);
fillMissingArray = fillmissing(ArraynorPm(detectBegin:detectEnd,stationIndex),'movmean',10) ;


% huadongT(1:160,fillMissingArray);
% 
% 
% 
% 
% 
% 
% PettittTest(fillMissingArray);
% 
% [UF,UB,UB2] = MKbreak(fillMissingArray,datenn,thisTime,StationName,stationIndex);
% 
% n = length(fillMissingArray);

%% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 35;
figureHeight = 15;
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [10 10 figureWidth figureHeight]);
hold on



datenn = datenum(time);
datenn=datenn(detectBegin:detectEnd);
fillMissingArray = fillmissing(ArraynorPm(detectBegin:detectEnd,stationIndex),'movmean',10) ;


stepLength = 1;
rightstepLength = 15;
sigLength = length(fillMissingArray);
birdArray = zeros(sigLength-rightstepLength-stepLength,1);

for stepIndex = stepLength+1:sigLength-rightstepLength
    leftwindow = fillMissingArray(stepIndex);
    rightwindow = fillMissingArray(stepIndex+1:stepIndex+rightstepLength);
    birdArray(stepIndex-stepLength) = (0.3*max(rightwindow)+0.7*mean(rightwindow)-mean(leftwindow))/mean(leftwindow);
end

yyaxis left
birddatenn = datenn(stepLength+1:sigLength-rightstepLength);
[~,ind] = max(birdArray);
birdArray2=sort(birdArray);
ind2 = find(birdArray==birdArray2(end-1));

ind3 = find(birdArray==birdArray2(end-2));


thisTime = time(detectBegin+stepLength:detectEnd-rightstepLength);
changeDate = thisTime(ind);
changeDate2 = thisTime(ind2);
changeDate3 = thisTime(ind3);
text(birddatenn(ind),birdArray(ind),changeDate);
% text(birddatenn(ind2),birdArray(ind2),changeDate2);
% text(birddatenn(ind3),birdArray(ind3),changeDate3);
plot(birddatenn(ind),birdArray(ind),"ro-","Color","b","LineWidth", 5,"linestyle","-",  "LineWidth", 2);
plot(birddatenn(ind2),birdArray(ind2),"ro-","Color","r","LineWidth", 5,"linestyle","-",  "LineWidth", 2);
plot(birddatenn(ind3),birdArray(ind3),"ro-","Color","g","LineWidth", 5,"linestyle","-",  "LineWidth", 2);

plot(birddatenn,birdArray,'m-','linewidth',1.5);
datetick('x','yyyy-mm');
ylim([min(birdArray)-0.05 max(birdArray)+0.05]);

yyaxis right
plot(datenn,fillMissingArray,'r-','linewidth',1.5);
ylim([0.8*min(fillMissingArray) max(fillMissingArray)*1.2]);
line([birddatenn(ind) birddatenn(ind)],[min(fillMissingArray)-5 max(fillMissingArray)+5],"Color","b","LineWidth", 1.2);
line([birddatenn(ind2) birddatenn(ind2)],[min(fillMissingArray)-5 max(fillMissingArray)+5],"Color","b","LineWidth", 1.2);
line([birddatenn(ind3) birddatenn(ind3)],[min(fillMissingArray)-5 max(fillMissingArray)+5],"Color","b","LineWidth", 1.2);
legend('突变点1','突变点2','突变点3','突变指数','PM-NOR','垂线');
titleText = strcat(StationName(stationIndex,2),'-',StationName(stationIndex,5),'-',num2str(stationIndex)  ,'-突变检测');
title(titleText);





catch e
    fprintf(1,'ERROR: %s\n', e.message);
  
    
end
disp(stationIndex);



% plot(datenn,UF')
% datetick('x','yyyy-mm')
% hold on
% plot(datenn,UB2);

end
aaaresult = ArraynorPm(:,[152]);
aaaresult2 = tem(:,[ 152]);