%% FOr evaluation purpose 
%  BER/FRR evaluation based on testing data
%  Ruirong Chen 
%  University of pittsburgh

close all
clear
MODE_ORDER = 1;% BPSK = 1 QPSK =2 16QAM = 4 64QAM = 6

Modulation = 'BPSK';
%file names
%Testing set names

Inter = 'Zigbee/'; %Microwave %BabyMonitor %Whitenoise %OtherWiFi %Zigbee

%Testing file names
%/home/labuser/payload_reconstruction/test_results/BabyMonitor/QPSK_Origin
%/home/labuser/payload_reconstruction/test_results/BabyMonitor/QPSK_After
%/home/labuser/payload_reconstruction/test_results/OtherWiFi/16QAM_Origin/
%/home/labuser/payload_reconstruction/MAT_OUT_16QAM_Origin
%/home/labuser/payload_reconstruction/MAT_OUT_16QAM

load(['/home/labuser/payload_reconstruction/FRR/',Modulation,'.mat']);


for j = 1:100
    dataname_origin = ['/home/labuser/payload_reconstruction/test_results/',Inter,Modulation,'_Origin/data', num2str(j-1), '.mat'];
    snr_name = ['/home/labuser/payload_reconstruction/test_results/',Inter,Modulation,'_Origin/sinr', num2str(j-1), '.mat'];
    labelname_origin = ['/home/labuser/payload_reconstruction/test_results/',Inter,Modulation,'_Origin/label', num2str(j-1), '.mat'];
    dataname = ['/home/labuser/payload_reconstruction/test_results/',Inter,Modulation,'_After/data', num2str(j-1), '.mat'];
    labelname = ['/home/labuser/payload_reconstruction/test_results/',Inter,Modulation,'_After/label', num2str(j-1), '.mat'];
    
    
    %dataname_origin = ['/home/labuser/payload_reconstruction/MAT_OUT_',Modulation,'_Origin_full/data', num2str(j-1), '.mat'];
    %snr_name = ['/home/labuser/payload_reconstruction/MAT_OUT_',Modulation,'_Origin_full/sinr', num2str(j-1), '.mat'];
    %labelname_origin = ['/home/labuser/payload_reconstruction/MAT_OUT_',Modulation,'_Origin_full/label', num2str(j-1), '.mat'];
    %dataname = ['/home/labuser/payload_reconstruction/MAT_OUT_',Modulation,'_full/data', num2str(j-1), '.mat'];
    %labelname = ['/home/labuser/payload_reconstruction/MAT_OUT_',Modulation,'_full/label', num2str(j-1), '.mat'];
    
    load(dataname_origin)
    load(labelname_origin)
    load(dataname)
    load(labelname)
    load(snr_name)
    
    
    for frame_index = 1:100
        data_reshape = data(1+40*(frame_index-1):40+40*(frame_index-1),:);
        label_reshape = label(1+40*(frame_index-1):40+40*(frame_index-1),:);
        frame_bin = reshape(de2bi(data_reshape(:),MODE_ORDER)',[],1);
        label_bin = reshape(de2bi(label_reshape(:),MODE_ORDER)',[],1);
        frame_deinterleave = wlanBCCDeinterleave(double(frame_bin),'Non-HT',48);
        label_deinterleave = wlanBCCDeinterleave(double(label_bin),'Non-HT',48);
        %frame_deinterleave = frame_bin;
        %label_deinterleave = label_bin;
        decoded_frame = wlanBCCDecode(int8(frame_deinterleave),'1/2','hard');
        label_frame = wlanBCCDecode(int8(label_deinterleave),'1/2','hard');
        BER(frame_index,j) = sum(abs(decoded_frame- label_frame));
        BER_before(frame_index,j) = sum(abs(frame_bin- label_bin));

        data_reshape_origin = data_origin(1+40*(frame_index-1):40+40*(frame_index-1),:);
        label_reshape_origin = label_origin(1+40*(frame_index-1):40+40*(frame_index-1),:);
        frame_bin_origin = reshape(de2bi(data_reshape_origin(:),MODE_ORDER)',[],1);
        label_bin_origin = reshape(de2bi(label_reshape_origin(:),MODE_ORDER)',[],1);
        frame_deinterleave_origin = wlanBCCDeinterleave(double(frame_bin_origin),'Non-HT',48);
        label_deinterleave_origin = wlanBCCDeinterleave(double(label_bin_origin),'Non-HT',48);
        %frame_deinterleave_origin = frame_bin_origin;
        %label_deinterleave_origin = label_bin_origin;

        decoded_frame_origin = wlanBCCDecode(int8(frame_deinterleave_origin),'1/2','hard');
        label_frame_origin = wlanBCCDecode(int8(label_deinterleave_origin),'1/2','hard');
        BER_origin(frame_index,j) = sum(abs(decoded_frame_origin- label_frame_origin));
        BER_before_origin(frame_index,j) = sum(abs(frame_bin_origin- label_bin_origin));
        SINR(frame_index,j) = sinr(frame_index,1);

    end

end
Corrected_bits = (sum(sum(BER_origin))-sum(sum(BER)))./(sum(sum(BER_origin)));
%difference_map_mean = mean(difference_map(difference_map~= 0 & isfinite(difference_map)));

Corrected_bits_before = (sum(sum(BER_before_origin)) - sum(sum(BER_before)))./(sum(sum(BER_before_origin)));
%difference_before_mean= mean(difference_before(difference_before~= 0 & isfinite(difference_before)));

[BER;BER_origin];

length(find(BER <=2))
length(find(BER_origin <=1))
Average_BER_original = (sum(sum(BER_origin))/(10000*length(label_frame)));
Average_BER_original_before = (sum(sum(BER_before_origin))/(10000*length(label_frame)));
Average_BER_before = (sum(sum(BER_before))/(10000*length(label_frame)));

Average_BER = (sum(sum(BER))/(10000*length(label_frame)));

[a,b] = find(BER_origin(:) <=40);
BER_array = BER(:);
BER_origin_array = BER_origin(:);
BER_origin_array_compare = BER_array(a,1);
length(find(BER_origin_array_compare < 5));
Accepted_frame = zeros(30,3);
BER_SINR = zeros(30,2);
%% Find SINR VS FRR VS BER
for incremental = 1:30
    [a1,b1]= find((SINR(:) >= (-2 +incremental)) & (SINR(:) <= (-1 +incremental)));
    BER_origin_array1 = BER_origin_array(a1,1);
    BER_array1 = BER_array(a1,1);   
    if  (length(a1)<=50) %isempty(a1) %||
        Accepted_frame(incremental,3) = 0;
        BER_improve(incremental) = 0;
        BER_SINR(incremental,1) = 0;
        BER_SINR(incremental,2) = 0;
    else
        BER_SINR(incremental,1) = mean(BER_origin_array1);
        BER_SINR(incremental,2) = mean(BER_array1);
        Accepted_frame(incremental,3) = length(find(BER_array1<2))/length(a1);
        BER_improve(incremental) = (mean(BER_origin_array1)-mean(BER_array1))/mean(BER_origin_array1);

    end    
    Accepted_frame(incremental,1) = length(a1)/10000;
    x(incremental) = -1 +incremental;
end


FRR = ones(30,1);
FRR(1,1) = FRR_0;
FRR(2,1) = FRR_1;
FRR(3,1) = FRR_2;
FRR(4,1) = FRR_3;
FRR(5,1) = FRR_4;
FRR(6,1) = FRR_5;
FRR(7,1) = FRR_6;
FRR(8,1) = FRR_7;
FRR(9,1) = FRR_8;
FRR(10,1) = FRR_9;
FRR(11,1) = FRR_10;
FRR(12,1) = FRR_11;
FRR(13,1) = FRR_12;
FRR(14,1) = FRR_13;
FRR(15,1) = FRR_14;
FRR(16,1) = FRR_15;
FRR(17,1) = FRR_16;
FRR(18,1) = FRR_17;
FRR(19,1) = FRR_18;
FRR(20,1) = FRR_19;
FRR(21,1) = FRR_20;
FRR(22,1) = FRR_21;
FRR(23,1) = FRR_22;

FRR_after = (1-FRR).*Accepted_frame(:,3) + FRR;

FRR_compare = [FRR,FRR_after];
FRR_Improve = (FRR_after - FRR)./FRR;
% 
% loss = zeros(20,1);
% for k = 1:19
%     loss(k+1) = Accepted_frame(31-k,1) +  loss(k,1);
% end
% 
% Theory_Throughput = 9.5*10^4./(sqrt(loss*0.5));
% Theory_Throughput(8:end) = 0;
% Throughput = 9.5*10^4./(1.1*sqrt(loss*0.1.*(1-Accepted_frame(11:30,2))));

figure(1)
plot(x,BER_improve*100,'LineWidth',5)
xlabel('SINR(dB)','FontSize',24);
ylabel('BER Improvment(%)','FontSize',24);

xlim([0,25])%BPSK
%xticks([-2 -1 0 1 2 3 4 ])%BPSK
%xlim([13.5,20.5])%64QAM
%xticks([14 15 16 17 18 19 20])%64QAM
set(gca,'FontSize',24)
figure(2)
plot(x,Accepted_frame(:,3)*100,'LineWidth',5)
xlabel('SINR(dB)','FontSize',24);
ylabel('Failed frame corrected(%)','FontSize',24);
xlim([0,25])%BPSK
%xticks([-2 -1 0 1 2 3 4 ])%BPSK
%xlim([13.5,20.5]) %64QAM
%xticks([14 15 16 17 18 19 20])%64QAM
%xticks([-2 -1 0 1 2 3 4 ])

set(gca,'FontSize',24)

figure(3)
bar(x,FRR_compare*100)
xlabel('SINR(dB)','FontSize',24);
ylabel('FRR(%)','FontSize',24);

xlim([-1.5,25.5])%BPSK
%xticks([-2 -1 0 1 2 3 4]) %BPSK
%xlim([13.5,20.5])%64QAM
%xticks([14 15 16 17 18 19 20])%64QAM

legend('Before NN','After NN')
set(gca,'FontSize',24)


figure(4)
bar(x,BER_SINR/length(label_frame)*100)
xlabel('SINR(dB)','FontSize',24);
ylabel('BER(%)','FontSize',24);
xlim([-1.5,25.5])%BPSK
%xticks([-2 -1 0 1 2 3 4 ]) %BPSK
%xlim([13.5,20.5])% 64QAM
%xticks([14 15 16 17 18 19 20]) % 64QAM
legend('Before NN','After NN')

set(gca,'FontSize',24)

figure(5)
plot(x,FRR_Improve*100,'LineWidth',5)
xlabel('SINR(dB)','FontSize',24);
ylabel('FRR Improvment(%)','FontSize',24);
xlim([0,25])%BPSK
%xticks([-2 -1 0 1 2 3 4 ]) %BPSK
%xlim([13.5,20.5])% 64QAM
%xticks([14 15 16 17 18 19 20]) % 64QAM
set(gca,'FontSize',24)


% figure(5)
% plot(x(11:30),flip(Theory_Throughput),x(11:30),flip(Throughput),'LineWidth',5)
% xlabel('SINR','FontSize',24);
% ylabel('Throughput','FontSize',24);
% xlim([-2,5])
% legend('Before NN','After NN')
% set(gca,'FontSize',24)
% 
% dataset1.CSI = data_set.CSI;
% dataset1.Pilot = data_set.Pilots;
% dataset1.Constellation = data_set.Constallation;
