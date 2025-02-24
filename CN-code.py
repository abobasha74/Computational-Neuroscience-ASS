def tanh(x):
    exp_pos = 2.718281828459 ** x
    exp_neg = 2.718281828459 ** (-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)

def random_weight(seed):
    return (seed * 0.4321 % 1.0) - 0.5

input1, input2 = 0.05, 0.1

b1, b2 = 0.5, 0.7

weights = {
    'w1': random_weight(1),
    'w2': random_weight(2),
    'w3': random_weight(3),
    'w4': random_weight(4),
    'w5': random_weight(5),
    'w6': random_weight(6),
    'w7': random_weight(7),  
    'w8': random_weight(8)  
}

hidden1 = tanh(input1 * weights['w1'] + input2 * weights['w2'] + b1)
hidden2 = tanh(input1 * weights['w3'] + input2 * weights['w4'] + b1)

output1 = tanh(hidden1 * weights['w5'] + hidden2 * weights['w6'] + b2)
output2 = tanh(hidden1 * weights['w7'] + hidden2 * weights['w8'] + b2)  

print(f"Input1: {input1}, Input2: {input2}")
print(f"Weights: w1={weights['w1']:.4f}, w2={weights['w2']:.4f}, w3={weights['w3']:.4f}, w4={weights['w4']:.4f}, w5={weights['w5']:.4f}, w6={weights['w6']:.4f}, w7={weights['w7']:.4f}, w8={weights['w8']:.4f}")
print(f"Hidden Layer: h1={hidden1:.4f}, h2={hidden2:.4f}")
print(f"Output 1: {output1:.4f}")
print(f"Output 2: {output2:.4f}")
