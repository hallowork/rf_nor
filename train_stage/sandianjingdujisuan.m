
%用途：把带精度验证的模型训练模型结果放进来，得到模型精度验证信息
%%%%%%a是一个随机数集合，映射非空行的所有行号
%%%%%%训练时取a(0.8*numdata+1:end)为测试集
load('F:\数据\PM-nor-全国结果\B-全国rf训练好的模型\rf模型-co30Trees-有测试集散点的');
%首先获取非空数据总量
a = a';
numdata = size(a,1);
testline = a(0.8*numdata+1:end,1);

alldataline = nonullDataLine(testline,1);
stationIdList  = zeros(length(alldataline),1)*nan;
for j = 1:length(alldataline)
   stationIdList(j,1)=  round((alldataline(j,1)+2906)/2906);
    
end




for i = 1:2028
    
    [x,y] = find(stationIdList==i);
    test_output_accuracy = trueValue(x,1);
    
    sim_aver = simValue(x,1);
    N_accuracy = size(x,1);
    if N_accuracy>=2
    aaresult.mae(i,1) = mean(abs(sim_aver - test_output_accuracy));
    aaresult.mape(i,1) = mean(abs(sim_aver' - test_output_accuracy')/test_output_accuracy');
    aaresult.rmse(i,1) =sqrt(sum((test_output_accuracy-sim_aver).^2)/N_accuracy);
    aaresult.R2(i,1) =(N_accuracy * sum(sim_aver .* test_output_accuracy) - sum(sim_aver) * sum(test_output_accuracy))^2 / ((N_accuracy * sum((sim_aver).^2) - (sum(sim_aver))^2) * (N_accuracy * sum((test_output_accuracy).^2) - (sum(test_output_accuracy))^2)); 
    aaresult.PCC(i,1)=corr(sim_aver,test_output_accuracy,'type','pearson','rows','complete');
   
    
    else
    aaresult.mae(i,1) = nan;
    aaresult.mape(i,1) = nan;
    aaresult.rmse(i,1) =nan;
    aaresult.R2(i,1) =nan; 
    aaresult.PCC(i,1)=nan; 
    end
end