
%%
clear all;
clc;warning off;
load('E:\ding_workspace\model\pm30Trees-有测试集散点的.mat');

load("把TP中NAN变成0.mat");
load("trend_2.mat");
load("StationName.mat");
load("mda8O3_day.mat");
load("longti_lati");
load("tem");
load("metoRange_allCountry");

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


Arraypcc = zeros(2028,1)*nan;

ArraytruePm = zeros(2906,2028)*nan;
ArraynorPm = zeros(2906,2028)*nan;
ArrayPCC_true = zeros(2028,1)*nan;
ArrayPCC_nor = zeros(2028,1)*nan;


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
    

inputIndex = [7 9 10 11 12 15 16 17 18 19 20 21 ];
%% 
outputIndex = [6];
metoRange = metoRange_allCountry;
%% 

for stationIndex = [1:1000]
    %1	     2	3	4	5	     6          7+	8	 9+	 10+	11+	12+	13  14	15+	16+
    %SO2	NO2	CO	O3	PM10	PM2.5       blh	msl	 r	 ssr	t2m	tp	u10	v10	ws	wd

    %生成机器学习需要的数组
try    
rawdata = newDayArray(:,[outputIndex inputIndex],stationIndex) ;
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

%通过非空行号获取非空的数据
nonullData = rawdata(nonullDataLine,:);


output_data=nonullData(:,1);
input_parameters=nonullData(:,2:end);
%temp = randperm(size(parameters,1));



%获取参数个数，非空数据条数
numpara=size(input_parameters,2);
numdata=size(input_parameters,1);




randemArray=randperm(numdata);
% 训练集——50个样本
train_input= input_parameters(randemArray(1:0.8*numdata),:)';
train_output= output_data(randemArray(1:0.8*numdata),:)';



% 测试集——10个样本
test_input_accuracy = input_parameters(randemArray(0.8*numdata+1:end),:)';
test_output_accuracy = output_data(randemArray(0.8*numdata+1:end),:);
N_accuracy = size(test_input_accuracy,2);



norMetoNum = 1000;
trendsize = 3;
longlaSize = 2;
trendline=trend;
mutilongla = repmat(longti_lati(stationIndex,:),norMetoNum,1);


% 气象场的选择范围--------------------------------------需要调整气象场的看这里
if isnan(metoRange)
    metoRange = input_parameters;
end
numMetoRange = size(metoRange,1);

%%构建气象场 随机抽取300条
%构建一个可以装testinput 的数组

%test_input_array这个数据是   10个输入参数*每条（每个时间点）300次重复*2906个时间点（非空的）
test_input_array=zeros(norMetoNum,numpara,size(trend,1));

for timei=1:size(trendline,1)
    
    %构造一个大小范围在numdata范围的，数量为300的随机数
    b=randperm(numMetoRange,norMetoNum);
    %随机数对应的300个随即气象作为输入
    test_input=metoRange(b,1:end-trendsize-longlaSize);
    %把时间项中的第一行复制300行作为第一个时间项的多次
    mutitrend = repmat(trendline(timei,:),norMetoNum,1);
    
    %把气象和longla项拼接起来
    test_input(:,end+1:end+longlaSize)=mutilongla;
    
    %把气象和时间项拼接起来
    test_input(:,end+1:end+trendsize)=mutitrend;
    
    
    %该时间点的气象300个随机构建完成
    test_input_array(:,:,timei)=test_input;

end

test_input_array2D  = threeDto2D3((1:size(trendline,1)),test_input_array);





%构建一个数组用来存归一化后的PM
nor_pm=zeros(size(trendline,1),1);

%逐个对每个

    
test_input=test_input_array2D;
N = size(test_input,2); 
test_input_nor = mapminmax('apply',test_input',ps_input);

[sim,scores] = predict(rfnet, test_input_nor');


sim_aver = mapminmax('reverse',sim,ps_output);




for timej = 1:size(trendline,1)
    nor_pm(timej)=mean(sim_aver((timej-1)*norMetoNum+1:(timej-1)*norMetoNum+norMetoNum));
end

ArraytruePm(:,stationIndex) = rawdata(:,1);
ArraynorPm(:,stationIndex) = nor_pm;
ArrayPCC_true(stationIndex,1)=corr(tem(:,stationIndex),rawdata(:,1),'type','pearson','rows','complete');
ArrayPCC_nor(stationIndex,1)=corr(tem(:,stationIndex),nor_pm,'type','pearson','rows','complete');




catch e
    fprintf(1,'ERROR: %s\n', e.message);
    fprintf(1,'ERROR: %s\n', num2str(e.stack.line));

end




disp(stationIndex);
end

aaaResult.ArraytruePm = ArraytruePm;
aaaResult.ArraynorPm = ArraynorPm;
aaaResult.PCC_true = ArrayPCC_true;
aaaResult.PCC_nor = ArrayPCC_nor;

function table2D  = threeDto2D3(hangIndexArray,targetArray)
    %二维后的表格  = threeDto2D3把第三维度展开(所有符合条件的行数的记号,被二维化的目标三维数组)
    
    length3D = length(hangIndexArray);
    %三维数组的层数
    
    table3D = targetArray(:,:,hangIndexArray);
    %获取3维数组范围
    
    timetrendLength = size(targetArray,1);
    %获取每层数据的时间项数据行行数
    
    for Index = 0:length3D-1
        table2D(Index*timetrendLength+1:Index*timetrendLength+timetrendLength,:)=table3D(:,:,Index+1);
        
    end
     
end

