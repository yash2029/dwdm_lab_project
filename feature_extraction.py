import os
import pandas as pd
import time
import warnings
warnings.filterwarnings( "ignore" )

src_addr = "ip.src"
dst_addr = "ip.dst"
frame_len = "frame.len"
protocol = "ip.proto"
tcp_flags = "tcp.flags"
frame_number = "frame.number"
tcp_sourceport = "tcp.srcport"
tcp_destport = "tcp.dstport"
udp_sourceport = "udp.srcport"
udp_destport = "udp.dstport"

start_time = time.time()

# CSV FILE FORMAT
# src_ip_addr, dst_ip_addr, src_port, dst_port, tcp_flags, protocol
os.system('find ./Ddos_Detection_Dataset/Ddos_benign/ -name *.pcap > file_list_benign.txt')
os.system('find ./Ddos_Detection_Dataset/Ddos_Attack_data/ -name *.pcap > file_list_malware.txt')
setup_time = time.time()-start_time
print("Setup time: " + str(setup_time))


os.system("mkdir CSV_FILES")
os.system("mkdir CSV_FILES/BENIGN")
os.system("mkdir CSV_FILES/MALWARE")


file = open("file_list_benign.txt").read().split("\n")
i =0
for line in file[0:-1]:
	i = i+1
	# foldername = line.split("/")[-2]
	# filename = foldername + line.split("/")[-1].split(".")[0] 
	os.system(f'tshark -r {line} -Y "tcp and not _ws.malformed and not icmp" -T fields -e {frame_number} -e {src_addr} -e {dst_addr} -e {tcp_sourceport} -e {tcp_destport} -e {tcp_flags} -e {protocol} -E header=y -E separator=, > ./CSV_FILES/BENIGN/{i}_tcp.csv')
	os.system(f'tshark -r {line} -Y "udp and not _ws.malformed and not icmp" -T fields -e {frame_number} -e {src_addr} -e {dst_addr} -e {udp_sourceport} -e {udp_destport} -e {tcp_flags} -e {protocol} -E header=y -E separator=, > ./CSV_FILES/BENIGN/{i}_udp.csv')
benign_read_time = time.time()-setup_time
print("Time to read benign files: " + str(benign_read_time))


file = open("file_list_malware.txt").read().split("\n")
i =0
for line in file[0:-1]:
	i = i+1
	# filename = line.split("/")[-1].split(".")[0]
	os.system(f'tshark -r {line} -Y "tcp&&!icmp" -T fields -e {frame_number} -e {src_addr} -e {dst_addr} -e {tcp_sourceport} -e {tcp_destport} -e {tcp_flags} -e {protocol} -E header=y -E separator=, > ./CSV_FILES/MALWARE/{i}_tcp.csv')
	os.system(f'tshark -r {line} -Y "udp&&!icmp" -T fields -e {frame_number} -e {src_addr} -e {dst_addr} -e {udp_sourceport} -e {udp_destport} -e {tcp_flags} -e {protocol} -E header=y -E separator=, > ./CSV_FILES/MALWARE/{i}_udp.csv')
total_time = time.time()-benign_read_time
print("Total run time: "+ str(total_time))