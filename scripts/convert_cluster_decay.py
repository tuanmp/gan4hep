#!/usr/bin/env python
# %% 
"""This script reads the original cluster decay files 
and boost the two hadron decay prodcuts to the cluster frame in which
they are back-to-back.
Save the output to a new file for training the model.
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_dataframe(filename, sep=",", engine=None):
    if type(filename) == list:
        print(filename)
        df_list = [
            pd.read_csv(f, sep=sep, header=None, names=None, engine=engine)
                for f in filename
        ]
        df = pd.concat(df_list, ignore_index=True)
        filename = filename[0]
    else:
        df = pd.read_csv(filename, sep=sep, 
                    header=None, names=None, engine=engine)
    return df

def calculate_mass(lorentz_vector):
    sum_p2 = sum([lorentz_vector[idx]**2 for idx in range(1,4)])
    return np.sqrt(lorentz_vector[0]**2 - sum_p2)


def create_boost_fn(cluster_4vec: np.ndarray):
    mass = calculate_mass(cluster_4vec)
    E0, p0 = cluster_4vec[0], cluster_4vec[1:]
    gamma = E0 / mass

    velocity = p0 / gamma / mass
    v_mag = np.sqrt(sum([velocity[idx]**2 for idx in range(3)]))
    n = velocity / v_mag

    def boost_fn(lab_4vec: np.ndarray):
        """4vector [E, px, py, pz] in lab frame"""
        E = lab_4vec[0]
        p = lab_4vec[1:]
        n_dot_p = np.sum((n * p))
        E_prime = gamma * (E - v_mag * n_dot_p)
        P_prime = p + (gamma - 1) * n_dot_p * n - gamma * E * v_mag * n
        return np.array([E_prime]+ P_prime.tolist())
    
    def inv_boost_fn(boost_4vec: np.ndarray):
        """4vecot [E, px, py, pz] in boost frame (aka cluster frame)"""
        E_prime = boost_4vec[0]
        P_prime = boost_4vec[1:]
        n_dot_p = np.sum((n * P_prime))
        E = gamma * (E_prime + v_mag * n_dot_p)
        p = P_prime + (gamma - 1) * n_dot_p * n + gamma * E_prime * v_mag * n
        return np.array([E]+ p.tolist())

    return boost_fn, inv_boost_fn


def boost(a_row: np.ndarray):
    """boost hadron one and two to the cluster framework"""
    boost_fn, _ = create_boost_fn(a_row[:4])
    b_cluster, b_h1, b_h2 = [boost_fn(a_row[4*x: 4*(x+1)]) for x in range(3)]
    return b_cluster.tolist() + b_h1.tolist() + b_h2.tolist()

def read(filename, outname, mode=2, example=False):
    """
    This Herwig dataset is for the "ClusterDecayer" study.
    Each event has q1, q1, cluster, h1, h2.
    I define 3 modes:
    0) both q1, q2 are with Pert=1
    1) only one of q1 and q2 is with Pert=1
    2) neither q1 nor q2 are with Pert=1
    3) at least one quark with Pert=1
    """
    print(f'reading from {filename}')
    df = read_dataframe(filename, ";", 'python')

    def split_to_float(df, sep=','):
        out = df
        if type(df.iloc[0]) == str:
            out = df.str.split(sep, expand=True).astype(np.float32)
        return out

    q1,q2,c,h1,h2 = [split_to_float(df[idx]) for idx in range(5)]
    if mode not in [0, 1, 2, 3]:
        print(f"mode {mode} is not known! use mode 2")
        mode = 2

    if mode == 0:
        selections = (q1[5] == 1) & (q2[5] == 1)
    elif mode == 1:
        selections = ((q1[5] == 1) & (q2[5] == 0)) | ((q1[5] == 0) & (q2[5] == 1))
    elif mode == 2:
        selections = (q1[5] == 0) & (q2[5] == 0)
        print("both quarks with Pert=0")
    elif mode == 3:
        selections = ~(q1[5] == 0) & (q2[5] == 0)
        print("at least one quark with Pert=1")
    else: pass

    outname = outname+f"_mode{mode}"
        
    cluster = c[[1, 2, 3, 4]][selections].values
    h1 = h1[[1, 2, 3, 4]][selections]
    h2 = h2[[1, 2, 3, 4]][selections]

    org_inputs = np.concatenate([cluster, h1, h2], axis=1)

    if example:
        print(org_inputs[0])
        return 
    # org_inputs = org_inputs[:3]

    # print("origin", org_inputs.shape, org_inputs)
    new_inputs = np.array([boost(row) for row in org_inputs])
    # print("boosted", new_inputs.shape, new_inputs)


    out_4vec = new_inputs[:, -4:]
    _,px,py,pz = [out_4vec[:, idx] for idx in range(4)]
    pT = np.sqrt(px**2 + py**2)
    phi = np.arctan(px/py)
    theta = np.arctan(pT/pz)
    out_truth = np.stack([phi, theta], axis=1)

    input_4vec = cluster

    scaler = MinMaxScaler(feature_range=(-1,1))
    # the input 4vector is the cluster 4vector
    input_4vec = scaler.fit_transform(input_4vec)
    pickle.dump(scaler, open(outname+"_scalar_input4vec.pkl", "wb"))

    out_truth = scaler.fit_transform(out_truth)
    pickle.dump(scaler, open(outname+"_scalar_outtruth.pkl", "wb"))

    np.savez(outname, input_4vec=input_4vec, out_truth=out_truth)


def check(outname, mode):
    import matplotlib.pyplot as plt
    outname = outname+f"_mode{mode}"

    arrays = np.load(outname+".npz")
    truth_in = arrays['out_truth']
    plt.hist(truth_in[:, 0], bins=100, histtype='step', label='phi')
    plt.hist(truth_in[:, 1], bins=100, histtype='step', label='theta')
    plt.savefig("angles.png")

    scaler_input = pickle.load(open(outname+"_scalar_input4vec.pkl", 'rb'))
    scaler_output = pickle.load(open(outname+"_scalar_outtruth.pkl", 'rb'))

    def dump_scaler(scaler):
        print("Min and Max for inputs: {",\
            ", ".join(["{:.6f}".format(x) for x in scaler.data_min_]),\
            ", ".join(["{:.6f}".format(x) for x in scaler.data_max_]), "}")

    print("//---- inputs ----")
    dump_scaler(scaler_input)
    print("//---- output ----")
    dump_scaler(scaler_output)
    print("Total entries:", truth_in.shape[0])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='convert herwig decayer')
    add_arg = parser.add_argument
    add_arg('inname', help='input filename')
    add_arg('outname', help='output filename')
    add_arg('-m', '--mode', help='mode', type=int, default=2)
    add_arg("-c", '--check', action='store_true', help="check outputs")
    add_arg("-e", '--example', action='store_true', help='print an example event')
    args = parser.parse_args()
    
    if args.check:
        check(args.outname, args.mode)
    else:
        read(args.inname, args.outname, args.mode, args.example)