close all
load('/home/labuser/payload_reconstruction/BPSK_full/payload_1.mat')
x = 1:48;
x_p = 1:40;
error_table = zeros(48,1);
error_table_t = zeros(40,1);
for i = 400:450
    CSI = data_set.CSI{i,1};  
    Pilot = data_set.Pilots{i,1};  
    Tx_data = data_set.Tx_dec{i,1};
    Rx_data = data_set.Rx_dec{i,1};
    [a,b] = find(Tx_data ~= Rx_data);
    error_sub = mod(a,48);
    error_t = floor(a/48);
    for j = 1:48
        error_table(j,1) = length(find(error_sub == j));
    end
    
    for k = 1:40
        error_table_t(k,1) = length(find(error_t == k-1));
    end
    figure(i)
    plot(x,angle(CSI)+20);
    hold on
    bar(x,error_table);
    
    figure(2*i)
    plot(x_p,angle(Pilot)+10);
    legend('1-12','13-24','25-36','37-48')
    hold on 
    bar(x_p,error_table_t);
    %figure(2*i)
   % plot(x,angle(CSI));
end