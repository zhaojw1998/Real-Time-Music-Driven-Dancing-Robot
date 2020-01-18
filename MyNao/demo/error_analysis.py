from matplotlib import pyplot as plt
import sys


with open('error_record\\error_record_6.txt', 'r') as f:
    record = f.readlines()
record = record
record_int = []
for i in record:
    record_int.append(int(i.strip('\n')))
plt.figure()
plt.plot([i + 1 for i in range(len(record_int))], record_int)
plt.xlabel('Beat Claim')
plt.ylabel('Frame-Beat Error')

plt.legend()
plt.ylim(-8, 8)
#plt.savefig('change_song.png')
plt.show()