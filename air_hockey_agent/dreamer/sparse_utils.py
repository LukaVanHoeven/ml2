import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

def initializeEpsilonWeightsMask(text,epsilon, noRows, noCols):
    # generate an epsilon based Erdos Renyi sparse weights mask
    mask_weights = np.random.rand(noRows, noCols)
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)
    mask_weights = np.random.rand(noRows, noCols)
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    sparsity = 1-noParameters/(noRows * noCols)
    print("Epsilon Sparse Initialization ",text,": Epsilon ",epsilon,"; Sparsity ",sparsity,"; NoParameters ",noParameters,"; NoRows ",noRows,"; NoCols ",noCols,"; NoDenseParam ",noRows*noCols)
    print (" OutDegreeBottomNeurons %.2f ± %.2f; InDegreeTopNeurons %.2f ± %.2f" % (np.mean(mask_weights.sum(axis=1)),np.std(mask_weights.sum(axis=1)),np.mean(mask_weights.sum(axis=0)),np.std(mask_weights.sum(axis=0))))
    return [noParameters, mask_weights.transpose()]


def initializeEpsilonWeightsMaskGPU(text, epsilon, noRows, noCols, device="cuda"):
    """
    Generates an epsilon-based Erdos-Renyi sparse weight mask using PyTorch.
    """
    # Generate random values on the specified device (GPU or CPU)
    mask_weights = torch.rand((noRows, noCols), device=device)

    # Compute probability threshold for sparsity
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)

    # Apply threshold to create the binary mask
    mask_weights = (mask_weights >= prob).float()

    # Count number of parameters
    noParameters = torch.sum(mask_weights).item()

    # Compute sparsity
    sparsity = 1 - (noParameters / (noRows * noCols))

    # Print statistics
    print(f"Epsilon Sparse Initialization {text}: Epsilon {epsilon}; Sparsity {sparsity}; NoParameters {noParameters}; NoRows {noRows}; NoCols {noCols}; NoDenseParam {noRows*noCols}")
    print(" OutDegreeBottomNeurons {:.2f} ± {:.2f}; InDegreeTopNeurons {:.2f} ± {:.2f}".format(
        torch.mean(mask_weights.sum(dim=1)).item(), torch.std(mask_weights.sum(dim=1)).item(),
        torch.mean(mask_weights.sum(dim=0)).item(), torch.std(mask_weights.sum(dim=0)).item()
    ))

    # Return the mask (transposed)
    return noParameters, mask_weights.T

def initializeSparsityLevelWeightMask(text,sparsityLevel,noRows, noCols):
    # generate an Erdos Renyi sparse weights mask
    prob=sparsityLevel
    mask_weights = np.random.rand(noRows, noCols)
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    sparsity = 1-noParameters/(noRows * noCols)
    epsilon = int((prob*(noRows * noCols)-1)/(noRows + noCols))
    print("Sparsity Level Initialization ",text,": Computed Epsilon ",epsilon,"; Sparsity ",sparsity,"; NoParameters ",noParameters,"; NoRows ",noRows,"; NoCols ",noCols,"; NoDenseParam ",noRows*noCols)
    print (" OutDegreeBottomNeurons %.2f ± %.2f; InDegreeTopNeurons %.2f ± %.2f" % (np.mean(mask_weights.sum(axis=1)),np.std(mask_weights.sum(axis=1)),np.mean(mask_weights.sum(axis=0)),np.std(mask_weights.sum(axis=0))))
    return [noParameters, mask_weights.transpose()]

def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx

def changeConnectivitySET(weights, noWeights, initMask, zeta, lastTopologyChange, iteration):
    # change Connectivity
    # remove zeta largest negative and smallest positive weights
    #print(f'change connectivity weight: {weights} noweights: {noWeights} iteration: {iteration}')
    weights = weights * initMask
    values = np.sort(weights.ravel())
    firstZeroPos = find_first_pos(values, 0)
    lastZeroPos = find_last_pos(values, 0)
    largestNegative = values[int((1 - zeta) * firstZeroPos)]
    smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos + zeta * (values.shape[0] - lastZeroPos)))]
    rewiredWeights = weights.copy();
    rewiredWeights[rewiredWeights > smallestPositive] = 1;
    rewiredWeights[rewiredWeights < largestNegative] = 1;
    rewiredWeights[rewiredWeights != 1] = 0;

    # add random weights
    nrAdd = 0
    if (lastTopologyChange==False):
        noRewires = noWeights - np.sum(rewiredWeights)
        while (nrAdd < noRewires):
            i = np.random.randint(0, rewiredWeights.shape[0])
            j = np.random.randint(0, rewiredWeights.shape[1])
            if (rewiredWeights[i, j] == 0):
                rewiredWeights[i, j] = 1
                nrAdd += 1

    ascStats=[iteration, nrAdd, noWeights, np.count_nonzero(rewiredWeights)]

    return [rewiredWeights,ascStats]


def changeConnectivitySETGPU(weights, noWeights, initMask, zeta, lastTopologyChange, iteration, device="cuda"):
    # Ensure tensors are on the correct device
    weights = weights.to(device) * initMask.to(device)
    
    # Sort weights
    values, _ = torch.sort(weights.view(-1))  # Flatten and sort
    firstZeroPos = torch.nonzero(values > 0, as_tuple=True)[0][0].item()  # First positive index
    lastZeroPos = torch.nonzero(values == 0, as_tuple=True)[0][-1].item()  # Last zero index

    # Get threshold values for sparsity
    largestNegative = values[int((1 - zeta) * firstZeroPos)]
    smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos + zeta * (values.shape[0] - lastZeroPos)))]

    # Rewire weights
    rewiredWeights = weights.clone()
    rewiredWeights[rewiredWeights > smallestPositive] = 1
    rewiredWeights[rewiredWeights < largestNegative] = 1
    rewiredWeights[rewiredWeights != 1] = 0

    # Add random connections if needed
    nrAdd = 0
    if not lastTopologyChange:
        noRewires = noWeights - torch.sum(rewiredWeights).item()
        while nrAdd < noRewires:
            i = torch.randint(0, rewiredWeights.shape[0], (1,), device=device).item()
            j = torch.randint(0, rewiredWeights.shape[1], (1,), device=device).item()
            if rewiredWeights[i, j] == 0:
                rewiredWeights[i, j] = 1
                nrAdd += 1

    ascStats = [iteration, nrAdd, noWeights, torch.count_nonzero(rewiredWeights).item()]

    return rewiredWeights, ascStats

def changeConnectivityXReLU(weights, noWeights, initMask, lastTopologyChange, iteration):
    weights = weights * initMask

    weightspos = weights.copy()
    weightspos[weightspos < 0] = 0
    strengthpos = np.sum(weightspos, axis=0)

    weightsneg = weights.copy()
    weightsneg[weightsneg > 0] = 0
    strengthneg = np.sum(weightsneg, axis=0)

    for j in range(strengthpos.shape[0]):
        if (strengthpos[j] + strengthneg[j] < 0):
            difference = strengthpos[j]
            iis = np.nonzero(weightsneg[:, j])[0]
            ww = weightsneg[iis, j]
            iisort = np.argsort(ww)
            for i in iisort:
                if (difference > 0):
                    difference += weightsneg[iis[i], j]
                else:
                    weights[iis[i], j] = 0
    rewiredWeights = weights.copy();
    rewiredWeights[rewiredWeights != 0] = 1;

    nrAdd = 0
    if (lastTopologyChange == False):
        noRewires = noWeights - np.sum(rewiredWeights)
        while (nrAdd < noRewires):
            i = np.random.randint(0, rewiredWeights.shape[0])
            j = np.random.randint(0, rewiredWeights.shape[1])
            if (rewiredWeights[i, j] == 0):
                rewiredWeights[i, j] = 1
                nrAdd += 1

    ascStats = [iteration, nrAdd, noWeights, np.count_nonzero(rewiredWeights)]
    return [rewiredWeights, ascStats]