import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

fig1 = plt.figure()

data = np.random.rand(2, 25)
# plt.plot([.006*50000,.012*50000,.018*50000,.024*50000,.03*50000,.036*50000,.042*50000], [0.0922226991913,
# 0.102343461843,
# 0.103603163727,
# 0.103652178445,
# 0.102468989917,
# 0.0929265881376,
# 0.0927392616271])
#
# plt.xlabel('No of training sample (300 increment,7 iteration)')
# plt.ylabel('EMD error')
# plt.title('Test error wrt training sample ')
# plt.show()

# plt.plot([.1,.2,.3,.4,.5,.6,.7,.8,.9],
#          [0.0773725615556,
# 0.0731282977333,
# 0.0693153756667,
# 0.081891874037,
# 0.113213898939,
# 0.127838971256,
# 0.122880512956,
# 0.117007545882,
# 0.114742176193
# ])

plt.plot([.95,.9,.85,.8,.75,.7,.65,.6,.55,.5,.45,.4,.35,.3,.25,.2,.15,.1,.05],
         [0.102571505141,
0.102642073118,
0.103293485105,
0.103447795125,
0.105418321257,
0.104035992881,
0.10435098989,
0.104746962675,
0.108142512414,
0.107596408515,
0.110203059668,
0.108833673166,
0.106076415274,
0.107139039406,
0.104649895973,
0.104427180946,
0.112601419925,
0.127409247442,
0.127848260227
])
plt.xlabel('percentage of training sample')
plt.ylabel('EMD error')
plt.title('Test error wrt training (Sample Size= 500) ')
plt.show()
plt.show()

#line_ani.save('lines.mp4')