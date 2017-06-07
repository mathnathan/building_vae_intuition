import numpy as np
import matplotlib.pyplot as plt
import tqdm

#np.random.seed(0)

def gauss2D(xpts, ypts, mean=(0.0,0.0), var=None, normed=True):

    mx, my = mean
    if var is None:
        sx, sy = [np.min(xpts.shape)]*2
    else:
        sx, sy = var

    coeff = np.log(2*np.pi*np.sqrt(sx)*np.sqrt(sy))
    vx = (xpts - mx)**2/(2*sx)
    vy = (ypts - my)**2/(2*sy)

    logOfGauss = -coeff-vx-vy
    r = np.exp(logOfGauss)

    if normed:
        return r/r.max()
    else:
        return r

def addHFNoise(data, amp):
    noise = np.random.normal(size=data.size).reshape(data.shape)
    return data + amp*noise

def addLFNoise(data, amp, scale):
    x = np.linspace(-scale/2,scale/2,scale)
    y = np.linspace(-scale/2,scale/2,scale)
    X, Y = np.meshgrid(x,y)
    for i in range(10):
        loc = np.random.randint(-(scale/2),(scale/2),2)
        data += (amp/2.0)*gauss2D(X, Y, mean=loc, var=(scale*20, scale*20))

    return data

def addNoise(data, amp, scale):
    """Add noise to the data"""
    lfnData = addLFNoise(data, amp, scale)
    noisyData = addHFNoise(hfnData, amp)

    return noisyData

def create_dataset():

    numelems = int(1e5)
    x = np.linspace(-14,14,28)
    y = np.linspace(-14,14,28)
    X, Y = np.meshgrid(x,y)
    dataset = np.ndarray((numelems,28*28))
    labels = np.ndarray((numelems))
    variances = [np.array((5,5)), np.array((30,30))]
    for i in tqdm.tqdm(range(numelems)):
        index = np.random.choice(range(len(variances)))
        var = variances[index] + np.random.normal()
        labels[i] = index
        dataset[i] = gauss2D(X, Y, var=var).flatten()
    np.save('single_gaussians_sizes=2', dataset)
    np.save('single_gaussians_sizes=2_labels', labels)


if __name__ == "__main__":

    create_dataset()
