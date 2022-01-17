%close all
close all
clear all
load('/home/labuser/payload_reconstruction/BPSK_NoInter/payload_0.mat')

data_ind = [2:7 9:21 23:27 39:43 45:57 59:64];
x = 1:48;
x_p = 1:40;
error_table = zeros(48,1);
error_table_64 = zeros(64,1);
error_table_t = zeros(40,1);
fft_pilot_sum = zeros(64,40);
CSI_threshold = 0;
for i = 1:5000
    error_mat = zeros(1920,1);
    CSI = data_set.CSI{i,1};  
    Pilot = data_set.Pilots{i,1};  
    Tx_data = data_set.Tx_dec{i,1};
    Rx_data = data_set.Rx_dec{i,1};
    [a,b] = find(Tx_data ~= Rx_data);
    error_sub = mod(a,48);
    error_t = floor(a/48);
    
    for j = 1:48
        error_table(j,1) = length(find(error_sub == j));
        error_table_64(data_ind(1,j),1) = length(find(error_sub == j));
    end
    
    for k = 1:40
        error_table_t(k,1) = length(find(error_t == k-1));
  
    end
    
    error_mat(a,1)= 1;
    error_mat_reshape = reshape(error_mat,48,40);
    angleCSI = angle(CSI);
    phase_diff = zeros(48,1);
    phase_diff(48,1) = angleCSI(1,48);
    for j = 2:48
        phase_diff(j-1,1) = abs(angleCSI(j)-angleCSI(j-1));
    end
%     figure(i)
%     
%     bar(x,error_table);
%     hold on
%     plot(x,abs(phase_diff));
    
    valid_phase_diff = find(phase_diff<3);
    CSI_threshold = CSI_threshold + mean(phase_diff(valid_phase_diff));
    fft_pilot = abs(fftshift(fft(Pilot.',64)));
    fft_pilot_sum = fft_pilot + fft_pilot_sum;
    cwt_csi = cwt(abs(CSI));    
    %figure(i+1)
    %plot(x_p,angle(Pilot)+10);

    %imagesc(abs(fft_pilot(data_ind,:))/max(max(abs(fft_pilot(data_ind,:)))))
    %legend('1-12','13-24','25-36','37-48')
    %hold on 
    %bar(x_p,error_table_t);
    %hold on 
    %barh(1:64,error_table_64)
    %figure(2*i)
   % plot(x,angle(CSI));
    %figure(i+2)
   % imagesc(error_mat_reshape);
   %figure(i+3)
   %imagesc(abs(cwt_csi(:,:,2)));
end
fft_pilot_mean = fft_pilot_sum/5000;
CSI_threshold_mean = CSI_threshold/5000;
load('/home/labuser/payload_reconstruction/16QAM/payload_1.mat')

error_table_intf = zeros(48,1);
error_table_64_intf = zeros(64,1);
error_table_t_intf = zeros(40,1);
corrected_find_error = zeros(5000,1);
corrected_find_error_CSI = zeros(5000,1);
wrong_find_error = zeros(5000,1);
wrong_find_error_CSI = zeros(5000,1);
Possible_find_error_CSI = zeros(5000,1);
FalseNegative_find_error_CSI = zeros(5000,1);

for i = 1:5000
    error_mat_intf = zeros(1920,1);
    decision_map = zeros(48,40);
    CSI_intf = data_set.CSI{i,1};  
    Pilot_intf = data_set.Pilots{i,1};  
    Tx_data_intf = data_set.Tx_dec{i,1};
    Rx_data_intf = data_set.Rx_dec{i,1};
    [a_intf,b1_intf] = find(Tx_data_intf ~= Rx_data_intf);
    error_sub_intf = mod(a_intf,48);
    error_t_intf = floor(a_intf/48);
    
    for j = 1:48
        error_table_intf(j,1) = length(find(error_sub_intf == j));
        error_table_64_intf(data_ind(1,j),1) = length(find(error_sub_intf == j));
    end
    
    for k = 1:40
        error_table_t_intf(k,1) = length(find(error_t_intf == k-1));
  
    end
    
    error_mat_intf(a_intf,1)= 1;
    error_mat_reshape_intf = reshape(error_mat_intf,48,40);
    angleCSI_intf = angle(CSI_intf);
    phase_diff_intf = zeros(48,1);
    phase_diff_intf(48,1) = angleCSI_intf(1,48);
    for j = 2:48
        phase_diff_intf(j-1,1) = abs(angleCSI_intf(j)-angleCSI_intf(j-1));
    end
    
    CSI_intefered = phase_diff_intf;
    CSI_indicator = find(CSI_intefered>(CSI_threshold_mean*2));
   % figure(i)
    
    %bar(x,error_table_intf);
    %hold on
    %plot(x,abs(phase_diff_intf));
   
    fft_pilot_intf = abs(fftshift(fft(Pilot_intf.',64)));
    cwt_csi_intf = cwt(abs(CSI_intf));    
    %figure(i+1)
    %plot(x_p,angle(Pilot)+10);
    difference_mat = abs(fft_pilot_mean(data_ind,:))/max(max(abs(fft_pilot_mean(data_ind,:))))-abs(fft_pilot_intf(data_ind,:))/max(max(abs(fft_pilot_intf(data_ind,:))));
    decision_map(difference_mat>0.10) = 2;
    error_difference_mapping = decision_map - error_mat_reshape_intf;
    corrected_find_error(i,1) = length(find(error_difference_mapping ==1))/length(a_intf);
    wrong_find_error(i,1) = length(find(error_difference_mapping == 2))/1920;

    
    
    decision_map_withCSI = decision_map;
    decision_map_withCSI(CSI_indicator,:) = decision_map_withCSI(CSI_indicator,:)+2;
    error_difference_mapping_CSI = decision_map_withCSI - error_mat_reshape_intf;
    corrected_find_error_CSI(i,1) = length(find(error_difference_mapping_CSI ==3))/length(a_intf);
    wrong_find_error_CSI(i,1) = length(find(error_difference_mapping_CSI ==4))/1920;
    Possible_find_error_CSI(i,1) = (length(find(error_difference_mapping_CSI ==2)) + length(find(error_difference_mapping_CSI ==1)))/1920;
    FalseNegative_find_error_CSI(i,1) = length(find(error_difference_mapping_CSI ==-1))/length(a_intf);
    %imagesc(abs(fft_pilot_intf(data_ind,:))/max(max(abs(fft_pilot_intf(data_ind,:)))))
    %legend('1-12','13-24','25-36','37-48')
    %hold on 
    %bar(x_p,error_table_t);
    %hold on 
    %barh(1:64,error_table_64)
    %figure(2*i)
   % plot(x,angle(CSI));
    if corrected_find_error_CSI(i,1) >0.60
        figure(i+2)
        subplot(1,4,1)
        imagesc(error_mat_reshape_intf);
        title('Constellation error','FontSize',24);
        subplot(1,4,2)
        imagesc(difference_mat);
        title('Pilot difference','FontSize',24);
        subplot(1,4,3)
        imagesc(decision_map);
        title('Error map from Pilot','FontSize',24);
        subplot(1,4,4)
        imagesc(error_difference_mapping_CSI);
        title('Error map from Pilot and CSI','FontSize',24);

    end
   %figure(i+3)
   %imagesc(abs(cwt_csi(:,:,2)));


end

valid_Error = rmmissing(corrected_find_error);
valid_Error_Removed = find(valid_Error~=0);
valid_Error_mean = mean(valid_Error(valid_Error_Removed));

valid_Error_CSI = rmmissing(corrected_find_error_CSI);
valid_Error_Removed_CSI = find(valid_Error_CSI~=0);
valid_Error_mean_CSI = mean(valid_Error_CSI(valid_Error_Removed_CSI));


wrong_find_error_mean = mean(wrong_find_error);
wrong_find_error_CSI_mean = mean(wrong_find_error_CSI);
Possible_find_error_CSI_mean = mean(Possible_find_error_CSI);


FalseNegative_CSI = rmmissing(FalseNegative_find_error_CSI);

FalseNegative_CSI_mean = mean(FalseNegative_CSI);

result = Possible_find_error_CSI_mean + valid_Error_mean_CSI;
% difference_mat = abs(fft_pilot(data_ind,:))/max(max(abs(fft_pilot(data_ind,:))))-abs(fft_pilot_intf(data_ind,:))/max(max(abs(fft_pilot_intf(data_ind,:))));
% figure(10000)
% imagesc(difference_mat)