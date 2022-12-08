
%%
clear all;
clc;warning off;
load('E:\ding_workspace\model\pm30Trees-�в��Լ�ɢ���.mat');

load("��TP��NAN���0.mat");
load("trend_2.mat");
load("StationName.mat");
load("mda8O3_day.mat");
load("longti_lati");
load("tem");
load("metoRange_allCountry");

% 2. �������ѵ�����Ͳ��Լ�


% ceshi = a;
% b = ceshi(:,13);
% c = ceshi(:,14);
% d = (b.^2+c.^2).^0.5;

 
%���ٶ�
newDayArray(:,15,:)=nan;
%����
newDayArray(:,16,:)=nan;


%�������
newDayArray(:,15,:) = (newDayArray(:,13,:).^2+ newDayArray(:,14,:).^2).^0.5;

%�������
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


%O3�ĳ�mda8
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


%�Ѿ�γ�ȷŽ�3D������
threeD_longti_lati = zeros(2906,2,2028);
for longti_lati_index = 1:2906
   threeD_longti_lati(longti_lati_index,:,:)  = longti_lati';
    
end
newDayArray(:,17:18,:) = threeD_longti_lati;

%��trendʱ��Ž�3D������

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

    %���ɻ���ѧϰ��Ҫ������
try    
rawdata = newDayArray(:,[outputIndex inputIndex],stationIndex) ;
%  x��ʾ������Ч����
b=all(~isnan(rawdata),2);

[x,y]= find(b==0);


%������һ�������кŵ�������
aline = zeros(length(b),1);
for linei = 1:length(b)
    aline(linei) = linei;
end


%�ǿյ����ݵ��к�
nonullDataLine = setdiff(aline,x);

%ͨ���ǿ��кŻ�ȡ�ǿյ�����
nonullData = rawdata(nonullDataLine,:);


output_data=nonullData(:,1);
input_parameters=nonullData(:,2:end);
%temp = randperm(size(parameters,1));



%��ȡ�����������ǿ���������
numpara=size(input_parameters,2);
numdata=size(input_parameters,1);




randemArray=randperm(numdata);
% ѵ��������50������
train_input= input_parameters(randemArray(1:0.8*numdata),:)';
train_output= output_data(randemArray(1:0.8*numdata),:)';



% ���Լ�����10������
test_input_accuracy = input_parameters(randemArray(0.8*numdata+1:end),:)';
test_output_accuracy = output_data(randemArray(0.8*numdata+1:end),:);
N_accuracy = size(test_input_accuracy,2);



norMetoNum = 1000;
trendsize = 3;
longlaSize = 2;
trendline=trend;
mutilongla = repmat(longti_lati(stationIndex,:),norMetoNum,1);


% ���󳡵�ѡ��Χ--------------------------------------��Ҫ�������󳡵Ŀ�����
if isnan(metoRange)
    metoRange = input_parameters;
end
numMetoRange = size(metoRange,1);

%%�������� �����ȡ300��
%����һ������װtestinput ������

%test_input_array���������   10���������*ÿ����ÿ��ʱ��㣩300���ظ�*2906��ʱ��㣨�ǿյģ�
test_input_array=zeros(norMetoNum,numpara,size(trend,1));

for timei=1:size(trendline,1)
    
    %����һ����С��Χ��numdata��Χ�ģ�����Ϊ300�������
    b=randperm(numMetoRange,norMetoNum);
    %�������Ӧ��300���漴������Ϊ����
    test_input=metoRange(b,1:end-trendsize-longlaSize);
    %��ʱ�����еĵ�һ�и���300����Ϊ��һ��ʱ����Ķ��
    mutitrend = repmat(trendline(timei,:),norMetoNum,1);
    
    %�������longla��ƴ������
    test_input(:,end+1:end+longlaSize)=mutilongla;
    
    %�������ʱ����ƴ������
    test_input(:,end+1:end+trendsize)=mutitrend;
    
    
    %��ʱ��������300������������
    test_input_array(:,:,timei)=test_input;

end

test_input_array2D  = threeDto2D3((1:size(trendline,1)),test_input_array);





%����һ�������������һ�����PM
nor_pm=zeros(size(trendline,1),1);

%�����ÿ��

    
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
    %��ά��ı��  = threeDto2D3�ѵ���ά��չ��(���з��������������ļǺ�,����ά����Ŀ����ά����)
    
    length3D = length(hangIndexArray);
    %��ά����Ĳ���
    
    table3D = targetArray(:,:,hangIndexArray);
    %��ȡ3ά���鷶Χ
    
    timetrendLength = size(targetArray,1);
    %��ȡÿ�����ݵ�ʱ��������������
    
    for Index = 0:length3D-1
        table2D(Index*timetrendLength+1:Index*timetrendLength+timetrendLength,:)=table3D(:,:,Index+1);
        
    end
     
end

