close all
clear
MODE_ORDER = 1;% BPSK = 1 QPSK =2 16QAM = 4

%file names
%Testing set names

%Microwave %BabyMonitor %Whitenoise %OtherWiFi

%Testing file names
%/home/labuser/payload_reconstruction/test_results/BabyMonitor/QPSK_Origin
%/home/labuser/payload_reconstruction/test_results/BabyMonitor/QPSK_After
%/home/labuser/payload_reconstruction/test_results/OtherWiFi/16QAM_Origin/
%/home/labuser/payload_reconstruction/MAT_OUT_16QAM_Origin
%/home/labuser/payload_reconstruction/MAT_OUT_16QAM
for j = 1:100
    dataname_origin = ['/home/labuser/payload_reconstruction/MAT_OUT_BPSK_Origin/data', num2str(j-1), '.mat'];
    snr_name = ['/home/labuser/payload_reconstruction/MAT_OUT_BPSK_Origin/sinr', num2str(j-1), '.mat'];

    labelname_origin = ['/home/labuser/payload_reconstruction/MAT_OUT_BPSK_Origin/label', num2str(j-1), '.mat'];
    dataname = ['/home/labuser/payload_reconstruction/MAT_OUT_BPSK/data', num2str(j-1), '.mat'];
    labelname = ['/home/labuser/payload_reconstruction/MAT_OUT_BPSK/label', num2str(j-1), '.mat'];
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
difference_map = (sum(sum(BER_origin))-sum(sum(BER)))./(sum(sum(BER_origin)));
%difference_map_mean = mean(difference_map(difference_map~= 0 & isfinite(difference_map)));

difference_before = (sum(sum(BER_before_origin)) - sum(sum(BER_before)))./(sum(sum(BER_before_origin)));
%difference_before_mean= mean(difference_before(difference_before~= 0 & isfinite(difference_before)));

[BER;BER_origin];

length(find(BER <=2))
length(find(BER_origin <=1))
Average_BER_original = (sum(sum(BER_origin))/(10000*3840));
Average_BER = (sum(sum(BER))/(10000*3840));

[a,b] = find(BER_origin(:) <=40);
BER_array = BER(:);
BER_origin_array = BER_origin(:);
BER_origin_array_compare = BER_array(a,1);
length(find(BER_origin_array_compare < 5));
Accepted_frame = zeros(30,3);
BER_SINR = zeros(30,2);
%% Find SINR VS FRR VS BER
for incremental = 1:30
    [a1,b1]= find((SINR(:) >= (floor(min(min(SINR)))+incremental+20)) & (SINR(:) <= (floor(min(min(SINR)))+incremental+1+20)));
    BER_origin_array1 = BER_origin_array(a1,1);
    BER_array1 = BER_array(a1,1);
    BER_SINR(incremental,1) = mean(BER_origin_array1);
    BER_SINR(incremental,2) = mean(BER_array1);    
    Accepted_frame(incremental,3) = length(find(BER_array1<3))/length(a1);
    Accepted_frame(incremental,1) = length(a1)/10000;
    BER_improve(incremental) = (mean(BER_origin_array1)-mean(BER_array1))/mean(BER_origin_array1);
    x(incremental) = floor(min(min(SINR)))+incremental+20;
end
loss = zeros(20,1);
for k = 1:19
    loss(k+1) = Accepted_frame(31-k,1) +  loss(k,1);
end

Theory_Throughput = 9.5*10^4./(sqrt(loss*0.5));
Theory_Throughput(8:end) = 0;
Throughput = 9.5*10^4./(1.1*sqrt(loss*0.1.*(1-Accepted_frame(11:30,2))));

figure(1)
plot(x,BER_improve,'LineWidth',5)
xlabel('SINR','FontSize',24);
ylabel('BER Improvment(%)','FontSize',24);
xlim([-2.5,6])
xticks([-2 -1 0 1 2 3 4 5 6])
set(gca,'FontSize',24)
figure(2)
plot(x,Accepted_frame(:,3),'LineWidth',5)
xlabel('SINR','FontSize',24);
ylabel('FRR Improvment(%)','FontSize',24);
xlim([-2.5,6])
xticks([-2 -1 0 1 2 3 4 5 6])
set(gca,'FontSize',24)

figure(3)
bar(x,Accepted_frame(:,2:3))
xlabel('SINR','FontSize',24);
ylabel('FRR','FontSize',24);
xlim([-2.5,6.5])
xticks([-2 -1 0 1 2 3 4 5 6])
legend('Before NN','After NN')
set(gca,'FontSize',24)


figure(4)
bar(x,BER_SINR)
xlabel('SINR','FontSize',24);
ylabel('BER','FontSize',24);
xlim([-2.5,6.5])
xticks([-2 -1 0 1 2 3 4 5 6])
legend('Before NN','After NN')

set(gca,'FontSize',24)

figure(5)
plot(x(11:30),flip(Theory_Throughput),x(11:30),flip(Throughput),'LineWidth',5)
xlabel('SINR','FontSize',24);
ylabel('Throughput','FontSize',24);
xlim([-2,5])
legend('Before NN','After NN')
set(gca,'FontSize',24)

dataset1.CSI = data_set.CSI;
dataset1.Pilot = data_set.Pilots;
dataset1.Constellation = data_set.Constallation;
