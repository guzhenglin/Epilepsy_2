import os

#chb_list = ['chb07','chb08','chb09','chb10','chb22']
chb_list = ['chb01','chb02','chb03','chb04','chb05']
#chb_list = ['chb07','chb08','chb09','chb10','chb22']
#chb_list = ['chb01','chb02','chb03','chb04','chb05','chb07','chb08','chb09','chb10','chb22']
# chb_list = ['chb01','chb02','chb03','chb04']
# chb_list = ['chb01','chb02','chb03','chb04']
# chb_list = ['chb22']
k_ = [0, 1, 2, 3, 4]
# k_ = [0]

epochs = 800

lstm_hidden = 16
out_channels1=8
out_channels2=8
out_channels3=16
out_channels4=16
fc1_out=32

with open("acc_log.txt", 'a') as acc_file:
    acc_file.write(f'epochs = {epochs}' + '\n')
    acc_file.write(f'lstm_hidden = {lstm_hidden}' + '\n')
    acc_file.write(f'out_channels1 = {out_channels1}' + '\n')
    acc_file.write(f'out_channels2 = {out_channels2}' + '\n')
    acc_file.write(f'out_channels3 = {out_channels3}' + '\n')
    acc_file.write(f'out_channels4 = {out_channels4}' + '\n')
    acc_file.write(f'fc1_out = {fc1_out}' + '\n')
    acc_file.close()

for chb in chb_list:
    with open("acc_log.txt", 'a') as acc_file:
        acc_file.write(str(chb) + '\n')
        acc_file.close()
    for k in k_:
        os.system(f"python main.py -chb {chb} -k {k} -epochs {epochs}\
                  -out_channels1 {out_channels1} -out_channels2 {out_channels2} -out_channels3 {out_channels3} -out_channels4 {out_channels4} \
                    -lstm_hidden {lstm_hidden} -fc1_out {fc1_out}")
    with open("acc_log.txt", 'a') as acc_file:
        acc_file.write('\n')
        acc_file.close()
