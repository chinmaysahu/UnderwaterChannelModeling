% Chinmay Sahu
% sahuc@clarkson.edu
% Last Updated: 04/23/19
% LMS Channel Adaptation
load('data1');
load('data2');

BerEst2=zeros(10,1);
% Estimating current channel quality
for channelNumber=1:10
    dataIn =x2_all(channelNumber,:)'>0.5; % input channel data of nth channel
    dataOut= y2_all(channelNumber,:)'>0.5; % convolve channel with the input
    BerEst2(channelNumber)=biterr(dataIn,dataOut)/length(dataIn);
end

figure
grid on
box on
grid minor
hold on
semilogy(1:10,BerEst1,'--','LineWidth',2);
hold on
semilogy(1:10,BerEst2,'-.','LineWidth',2);
% pbaspect([1 1 1])
title('Evaluating BER of channels in Data','FontName','Times New Roman','FontSize',28,'FontWeight','bold','interpreter','latex');
xlabel('Channel number','FontName','Times New Roman','FontSize',28,'FontWeight','bold','interpreter','latex');
ylabel('Bit Error Rate','FontName','Times New Roman','FontSize',28,'FontWeight','bold','interpreter','latex');
% set(gca,'fontsize',20,'ticklength',[0.025 0.05])
set(gca,'FontName','Times New Roman','FontSize',20,'FontWeight','bold','linewidth',2,'ticklength',[0.025 0.05],'TickLabelInterpreter', 'latex');
legend( 'Data-1','Data-2')
xlim([1 10])
%% LMS & LevinDurbin Equalizer Design
clc
load('data1');
load('data2');
LmsBerEst1=zeros(10,1);
LevBerEst1=zeros(10,1);
LmsBerEst2=zeros(10,1);
LevBerEst2=zeros(10,1);
filter_order=100;
b1=zeros(filter_order,1);
b2=zeros(filter_order,1);
% LMS Adaptation
for channelNumber = 1:10
    x1 =x_all(channelNumber,:)'>0.5; % input channel data of nth channel
    d1 = y_all(channelNumber,:)'>0.5; % convolve channel with the input
    x2 =x2_all(channelNumber,:)'>0.5; % input channel data of nth channel
    d2 = y2_all(channelNumber,:)'>0.5; % convolve channel with the input
    
    lms=dsp.LMSFilter(filter_order,'StepSize',0.01,'WeightsOutputPort',true);
    [~,e,w1]=lms(double(x1),double(d1)); % LMS filter weights
    [~,e,w2]=lms(double(x2),double(d2)); % LMS filter weights
    
    lms_filter1=w1./w1(1); % estimater equalizer coefficients
    lms_filter2=w1./w1(1); % estimater equalizer coefficients
    
    dataOut1=(filter(1,lms_filter1,d1))>0.5; % information bits pass through equalizer
    dataOut2=(filter(1,lms_filter2,d2))>0.5; % information bits pass through equalizer
    LmsBerEst1(channelNumber)=biterr(x1,dataOut1)/length(x1); % BER of channel
    LmsBerEst2(channelNumber)=biterr(x2,dataOut2)/length(x2); % BER of channel
    
    %%%LevinsonDurbin
    [a1 ~]=levinson(autocorr(d1,filter_order),filter_order-1);
    [a2 ~]=levinson(autocorr(d1,filter_order),filter_order-1);
    b1=a1;
    b2=a2;
    dataOut1=(filter(b1,1,d1))>0.5; % information bits pass through equalizer
    dataOut2=(filter(b2,1,d2))>0.5; % information bits pass through equalizer
    LevBerEst1(channelNumber)=biterr(x1,dataOut1)/length(x1); % BER of channel
    LevBerEst2(channelNumber)=biterr(x2,dataOut2)/length(x2); % BER of channel
end
% Plot results
figure
grid on
box on
grid minor
hold on
semilogy(1:10,LmsBerEst1,'--','LineWidth',2);
hold on
semilogy(1:10,LevBerEst1,'-.','LineWidth',2);
hold on
semilogy(1:10,LmsBerEst2,'--','LineWidth',2);
hold on
semilogy(1:10,LevBerEst2,'-.','LineWidth',2);
hold on
% semilogy(1:10,NNBerEst1,'--','LineWidth',2);
% hold on
% semilogy(1:10,NNBerEst2,'-.','LineWidth',2);
% pbaspect([1 1 1])
title(['Evaluating BER of equalisers: FilterOrder=',num2str(filter_order)],'FontName','Times New Roman','FontSize',28,'FontWeight','bold','interpreter','latex');
xlabel('Channel number','FontName','Times New Roman','FontSize',28,'FontWeight','bold','interpreter','latex');
ylabel('Bit Error Rate','FontName','Times New Roman','FontSize',28,'FontWeight','bold','interpreter','latex');
% set(gca,'fontsize',20,'ticklength',[0.025 0.05])
set(gca,'FontName','Times New Roman','FontSize',20,'FontWeight','bold','linewidth',2,'ticklength',[0.025 0.05],'TickLabelInterpreter', 'latex');
legend( 'LMSDataset-1','LevinsonDataset-1','LMSDataset-2','LevinsonDataset-2')
xlim([1 10])
%% Neural Network based modeling
load data2.mat
%train network

number_of_layers=10;
neurons_per_layer=10;
training_fn='trainbfg';
NNBerEst2=zeros(10,1);

for channel_num=1:10
    
    number_of_layers=10;
    neurons_per_layer=10;
    training_fn='trainbfg';
    
    train_data_input=y2_all(channel_num,1:9000)>0.5;
    train_data_output=x2_all(channel_num,1:9000)>0.5;
    
    
    hiddensize=neurons_per_layer*ones(1,number_of_layers);
    net=feedforwardnet(hiddensize,training_fn);
    % view(net);
    net=configure(net,train_data_input,train_data_output);
    net=train(net,train_data_input,train_data_output);
    
    %%Test network
    test_data_input=y2_all(channel_num,9001:10000);
    test_data_output=sim(net,test_data_input);
    
    x_out=test_data_output>0.5;
    x_ver=x2_all(channel_num,9001:10000)>0.5;
    
    NNBerEst2(channel_num)=sum(x_out~=x_ver)/1000;
end

figure
grid on
box on
grid minor
hold on
semilogy(1:10,NNBerEst1,'--','LineWidth',2);
hold on
semilogy(1:10,NNBerEst2,'-.','LineWidth',2);
pbaspect([1 1 1])
title('Evaluating BER of equalisers (Neural Network)','FontName','Times New Roman','FontSize',28,'FontWeight','bold','interpreter','latex');
xlabel('Channel number','FontName','Times New Roman','FontSize',28,'FontWeight','bold','interpreter','latex');
ylabel('Bit Error Rate','FontName','Times New Roman','FontSize',28,'FontWeight','bold','interpreter','latex');
% set(gca,'fontsize',20,'ticklength',[0.025 0.05])
set(gca,'FontName','Times New Roman','FontSize',20,'FontWeight','bold','linewidth',2,'ticklength',[0.025 0.05],'TickLabelInterpreter', 'latex');
legend('NNData-1','NNData-2')
xlim([1 10])

%% LMS channel estimation and it's BER
clc
load('data1');
load('data2');
LmsBerEst1=zeros(10,1);
LevBerEst1=zeros(10,1);
LmsBerEst2=zeros(10,1);
LevBerEst2=zeros(10,1);
filter_order=20;
b1=zeros(filter_order,1);
b2=zeros(filter_order,1);
% LMS Adaptation
for channelNumber = 5:10
    x1 =x_all(channelNumber,:)'>0.5; % input channel data of nth channel
    d1 = y_all(channelNumber,:)'>0.5; % convolve channel with the input
    x2 =x2_all(channelNumber,:)'>0.5; % input channel data of nth channel
    d2 = y2_all(channelNumber,:)'>0.5; % convolve channel with the input
    
    lms=dsp.LMSFilter(filter_order,'StepSize',0.05,'WeightsOutputPort',true);
    [~,e,w1]=lms(double(x1),double(d1)); % LMS filter weights
    plot(e,'LineWidth',2);
    title(['Evaluating LMS convergence, Data-1,Channel=',num2str(channelNumber)],'FontName','Times New Roman','FontSize',28,'FontWeight','bold','interpreter','latex');
    xlabel('Interations','FontName','Times New Roman','FontSize',28,'FontWeight','bold','interpreter','latex');
    ylabel('Error Magnitude','FontName','Times New Roman','FontSize',28,'FontWeight','bold','interpreter','latex');
    
    [~,e,w2]=lms(double(x2),double(d2)); % LMS filter weights
    
    dataOut1=(filter(w1,1,x1))>0.5; % information bits pass through equalizer
    dataOut2=(filter(w2,1,x2))>0.5; % information bits pass through equalizer
    LmsBerEst1(channelNumber)=biterr(x1,dataOut1)/length(x1); % BER of channel
    LmsBerEst2(channelNumber)=biterr(x2,dataOut2)/length(x2); % BER of channel
    
    %%%LevinsonDurbin
    [a1,e]=levinson(autocorr(d1,filter_order),filter_order-1);
    [a2 ~]=levinson(autocorr(d1,filter_order),filter_order-1);
    b1=a1;
    b2=a2;
    dataOut1=(filter(b1,1,d1))>0.5; % information bits pass through equalizer
    dataOut2=(filter(b2,1,d2))>0.5; % information bits pass through equalizer
    LevBerEst1(channelNumber)=biterr(x1,dataOut1)/length(x1); % BER of channel
    LevBerEst2(channelNumber)=biterr(x2,dataOut2)/length(x2); % BER of channel
end
% Plot results
figure
grid on
box on
grid minor
hold on
semilogy(1:10,LmsBerEst1,'--','LineWidth',2);
hold on
semilogy(1:10,LevBerEst1,'-.','LineWidth',2);
hold on
semilogy(1:10,LmsBerEst2,'--','LineWidth',2);
hold on
semilogy(1:10,LevBerEst2,'-.','LineWidth',2);
hold on
% semilogy(1:10,NNBerEst1,'--','LineWidth',2);
% hold on
% semilogy(1:10,NNBerEst2,'-.','LineWidth',2);
% pbaspect([1 1 1])
title(['Evaluating BER of equalisers: FilterOrder=',num2str(filter_order)],'FontName','Times New Roman','FontSize',28,'FontWeight','bold','interpreter','latex');
xlabel('Channel number','FontName','Times New Roman','FontSize',28,'FontWeight','bold','interpreter','latex');
ylabel('Bit Error Rate','FontName','Times New Roman','FontSize',28,'FontWeight','bold','interpreter','latex');
% set(gca,'fontsize',20,'ticklength',[0.025 0.05])
set(gca,'FontName','Times New Roman','FontSize',20,'FontWeight','bold','linewidth',2,'ticklength',[0.025 0.05],'TickLabelInterpreter', 'latex');
legend( 'LMSDataset-1','LevinsonDataset-1','LMSDataset-2','LevinsonDataset-2')
xlim([1 10])
