close all
MODE_ORDER = 4;% BPSK = 1 QPSK =2 16QAM = 4

%file names
%Testing set names

%Microwave %BabyMonitor %Whitenoise %OtherWiFi

%Testing file names
%/home/labuser/payload_reconstruction/test_results/BabyMonitor/QPSK_Origin
%/home/labuser/payload_reconstruction/test_results/BabyMonitor/QPSK_After

%/home/labuser/payload_reconstruction/MAT_OUT_16QAM_Origin
%/home/labuser/payload_reconstruction/MAT_OUT_16QAM
for j = 1:100
    dataname_origin = ['/home/labuser/payload_reconstruction/test_results/BabyMonitor/16QAM_Origin/data', num2str(j-1), '.mat'];
 
    labelname_origin = ['/home/labuser/payload_reconstruction/test_results/BabyMonitor/16QAM_Origin/label', num2str(j-1), '.mat'];
    dataname = ['/home/labuser/payload_reconstruction/test_results/BabyMonitor/16QAM_After/data', num2str(j-1), '.mat'];
    labelname = ['/home/labuser/payload_reconstruction/test_results/BabyMonitor/16QAM_After/label', num2str(j-1), '.mat'];
    load(dataname_origin)
    load(labelname_origin)
    load(dataname)
    load(labelname)

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
BER_origin_array_compare = BER_array(a,1);
length(find(BER_origin_array_compare < 5));
