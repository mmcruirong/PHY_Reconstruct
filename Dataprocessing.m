close all
load('/home/labuser/payload_reconstruction/BPSK_full1/payload_10.mat')
x = 1:48;
error_table = zeros(48,1);
for i = 100:150
    CSI = data_set.CSI{i,1};  
    Tx_data = data_set.Tx_dec{i,1};
    Rx_data = data_set.Rx_dec{i,1};
    [a,b] = find(Tx_data ~= Rx_data);
    error_sub = mod(a,48);
    for j = 1:48
        error_table(j,1) = length(find(error_sub == j));
    end
    figure(i)
    plot(x,angle(CSI)+20);
    hold on
    bar(x,error_table);
    %figure(2*i)
   % plot(x,angle(CSI));
end