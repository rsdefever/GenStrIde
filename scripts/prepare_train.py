# Ryan DeFever
# Sarupria Research Group
# Clemson University
# 2019 Jun 11

import sys
import os
import math
import numpy as np
import argparse
import MDAnalysis as mda

# For Cython nlist
# PROVIDE AN ABSOLUTE PATH TO MAKE THIS IMPORT WORK
# FROM WHEREVER YOU ARE EXECUTING THIS CODE
sys.path.append('/home/rdefeve/repos/stride-upload/mda_custom')
import nsgrid.nsgrid_rsd as nsgrid

def main():
    # Argument parsing
    args = get_args()

    # List of all classes
    classes = ['liq','fcc','hcp','bcc']
    nclass = len(classes)

    # Get list of all subdirectories
    dirs = [d for d in os.listdir(args.path) if os.path.isdir("%s/%s" % (args.path, d))]
    # Narrow directory list to specified classes
    # Assumes (T,P) conditions in directories named 'phase_pPPPtTTT'
    dirs = [d for d in dirs if d.split("_")[0] in classes]
  
    # Get all files
    files_gro = ["%s/prod.gro" % d for d in dirs]
    files_xtc = ["%s/prod.xtc" % d for d in dirs]

    assert len(files_gro) == len(files_xtc), \
            "Error, unequal number of .gro and .xtc files found" 
 
    # Initialize lists for samples and labels
    samples = []
    labels = []

    # Initialize values for mean/stdev of nneigh
    count = 0
    mean = 0
    m2 = 0


    # Read in data for each file
    for fcount in range(len(files_gro)):
        # Progress
        print("Reading file: %s" % files_gro[fcount])
        # Extract classid and create label
        classid = files_gro[fcount].split("_")[0]
        ndx = classes.index(classid)
        label = np.zeros(len(classes))
        label[ndx] = 1
        # Import topology
        u = mda.Universe(files_gro[fcount],
                         files_xtc[fcount])
        # Loop over trajectory
        for ts in u.trajectory:
            # Select atoms at random for samples
            sel = np.random.choice(u.atoms.n_atoms,size=args.n_select,replace=False)
            #sel = [13518]
            #print(sel)
            # Create neighbor list for atoms
            nlist = nsgrid.FastNS(args.cutoff*10.0,u.atoms.positions,ts.dimensions).search(u.atoms[sel].positions)
            ndxs = nlist.get_indices()
            dxs = nlist.get_dx()
            dists = nlist.get_distances()
            for i in range(len(sel)):
                np_dxs = np.asarray(dxs[i]).reshape(-1,3)
                sort = np.argsort(dists[i])
                # Remove the (0,0,0) atom
                vals = np_dxs[sort][1:]
                if np.linalg.norm(vals[0]) == 0.0:
                    print(vals)
                    print(files_gro[fcount])
                    print(ts.frame)
                    print(sel[i])
                nneigh = vals.shape[0]
                # Calc avg/stdev of nneigh with Welford Alg.
                count += 1
                delta = nneigh - mean
                mean += delta/count
                delta2 = nneigh - mean
                m2 += delta*delta2
                # Zero padding
                if nneigh > args.max_neigh:
                    sample = np.resize(vals,(args.max_neigh,3))
                elif nneigh < args.max_neigh:
                    npad = args.max_neigh  - nneigh
                    sample = np.vstack((vals,np.zeros((npad,3))))
                else:
                    sample = vals
                # Append sample and label
                samples.append(sample)
                labels.append(label)

    # convert samples and labels to numpy 
    samples = np.asarray(samples)
    labels = np.asarray(labels)

    ## Here is some extra pre-processing of training data ##
    # Extract idxs for different classes
    idxs = [np.where(labels[:,i] == 1)[0] for i in range(nclass)]
    # Extract samples/labels for each class
    samples_list = [samples[idx] for idx in idxs]
    labels_list = [labels[idx] for idx in idxs]
    # 1. In place shuffle WITHIN each class -- returns [None,none,none,none]
    [np.random.shuffle(samples_list[i]) for i in range(nclass)]
    # 2. Confirm equal number of samples per phase
    for i in range(nclass):
        print("Total number of available samples for class %s: %d" % (classes[i],samples_list[i].shape[0]))
        assert samples_list[i].shape[0] >= args.n_samples, \
            "Error, only %d samples in class %s but requested %d." \
            % (samples_list[i].shape[0],classes[i],args.n_samples)
    samples_list = [samples_list[i][:args.n_samples] for i in range(nclass)]
    labels_list = [labels_list[i][:args.n_samples] for i in range(nclass)]
    # 3. Restack
    samples = np.vstack(samples_list)
    labels = np.vstack(labels_list)
    # 4. Normalize each sample such that distance to closest atom is 1.0 units
    for k in range(samples.shape[0]):
        samples[k,...] = samples[k,...]/np.linalg.norm(samples[k][0])

    # save output files
    np.save(args.out_name + '_scaled_shuffled_equal_samples.npy', samples)
    np.save(args.out_name + '_scaled_shuffled_equal_labels.npy', labels)

    # Finalize and print avg/stdev nneigh
    print("Mean nneigh: %f" % mean)
    print("Stdev nneigh: %f" % ((math.sqrt(m2/count))))

def get_args():

    #Parse Arguments
    parser = argparse.ArgumentParser(description='Create datasets for pointnet training for crystal structure ID')
    parser.add_argument('--path', help='path to simulations for generating training data', type=str, required=True)
    parser.add_argument('--out_name', help='file prefix to save training dataset', type=str, required=True)
    parser.add_argument('--cutoff', help='neighbor cutoff for point clouds', type=float, required=True)
    parser.add_argument('--max_neigh',help='max neighbors in point clouds',type=int,required=True)
    parser.add_argument('--n_select', help='number training samples to extract from each frame of a simulation', type=int, required=False,default=5)
    parser.add_argument('--n_samples', help='total number training samples per phase', type=int, required=False,default=100000)

    args = parser.parse_args()

    return args

# Boilerplate notation to run main fxn
if __name__ == "__main__":
    main()

