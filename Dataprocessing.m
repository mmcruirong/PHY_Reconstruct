%close all

clear all
load('/home/labuser/payload_reconstruction/BPSK_NoInter/payload_0.mat')
%load('/home/labuser/payload_reconstruction/BPSK/payload_0.mat')

data_ind = [2:7 9:21 23:27 39:43 45:57 59:64];
x = 1:48;
x_p = 1:40;
error_table = zeros(48,1);
error_table_64 = zeros(64,1);
error_table_t = zeros(40,1);
for i = 5:100:1000
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
    phase_diff(48,1) = 0;
    for j = 2:48
        phase_diff(j-1,1) = angleCSI(j)-angleCSI(j-1);
    end
    figure(i)
    
    bar(x,error_table);
    hold on
    plot(x,abs(phase_diff));
   
    fft_pilot = fftshift(fft(Pilot.',64));
    cwt_csi = cwt(abs(CSI));    
    figure(i+1)
    %plot(x_p,angle(Pilot)+10);

    imagesc(abs(fft_pilot(data_ind,:))/max(max(abs(fft_pilot(data_ind,:)))))
    %legend('1-12','13-24','25-36','37-48')
    %hold on 
    %bar(x_p,error_table_t);
    %hold on 
    %barh(1:64,error_table_64)
    %figure(2*i)
   % plot(x,angle(CSI));
    figure(i+2)
    imagesc(error_mat_reshape);
   %figure(i+3)
   %imagesc(abs(cwt_csi(:,:,2)));
end