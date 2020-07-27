# 1/usr/bin/env python


import matplotlib.pyplot as plt


def plot(scores, name):
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(scores['train'])),
             scores['train'], label='train{}'.format(name))
    plt.title('{}plot'.format(name))
    plt.xlabel('Epoch')
    plt.ylabel('{}'.format(name))
    plt.legend()
    plt.show()
