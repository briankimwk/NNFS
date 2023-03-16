"""
Categorical cross entropy uses log so that all the probabilities (which are between 0 and 1) will be changed accrodingly.
It will be negative num * log, which will always be a positive number between 0 and 1 (as per the natural log)
"""

import math
softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]  # dummy variable so only one of the loss formula is a probability

# loss = -(math.log(softmax_output[0])*target_output[0] +
#          math.log(softmax_output[1])*target_output[1] +
#          math.log(softmax_output[2])*target_output[2])

loss = -math.log(softmax_output[0])
print(loss)
print(-math.log(0.5))
print(-math.log(0.99))
print(-math.log(0.2))  # as probability gets lower, the LOSS or the error is higher