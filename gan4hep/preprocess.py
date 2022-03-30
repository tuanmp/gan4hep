import os
import pandas as pd
import numpy as np
import vector
from sklearn.preprocessing import MinMaxScaler
import sklearn
import joblib


def convert_lorentz_vector(data):
    """
    Take data as a numpy array of shape (n_events, 4) of 4 momenta, (E, px, py, pz) 
    and convert to scikit-hep vector for manipulation
    """
    data = vector.arr({
        'E': data[:, 0],
        'px': data[:, 1],
        'py': data[:, 2],
        'pz': data[:, 3],
    })
    return data

def shuffle(array: np.ndarray):
    from numpy.random import MT19937
    from numpy.random import RandomState, SeedSequence
    np_rs = RandomState(MT19937(SeedSequence(123456789)))
    np_rs.shuffle(array)


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
   
def herwig_angles(filename,
        max_evts=None, testing_frac=0.1):
    """
    This reads the Herwig dataset where one cluster decays
    into two particles.
    In this case, we ask the GAN to predict the theta and phi
    angle of one of the particles
    """
    df = read_dataframe(filename, engine='python')

    event = None
    with open(filename, 'r') as f:
        for line in f:
            event = line
            break
    particles = event[:-2].split(';')

    input_4vec = df[0].str.split(",", expand=True)[[4, 5, 6, 7]].to_numpy().astype(np.float32)
    out_particles = []
    for idx in range(1, len(particles)):
        out_4vec = df[idx].str.split(",", expand=True).to_numpy()[:, -4:].astype(np.float32)
        out_particles.append(out_4vec)

    # ======================================
    # Calculate the theta and phi angle 
    # of the first outgoing particle
    # ======================================
    out_4vec = out_particles[0]
    px = out_4vec[:, 1].astype(np.float32)
    py = out_4vec[:, 2].astype(np.float32)
    pz = out_4vec[:, 3].astype(np.float32)
    pT = np.sqrt(px**2 + py**2)
    phi = np.arctan(px/py)
    theta = np.arctan(pT/pz)

    # <NOTE, inputs and outputs are scaled to be [-1, 1]>
    max_phi = np.max(np.abs(phi))
    max_theta = np.max(np.abs(theta))
    scales = np.array([max_phi, max_theta], np.float32)

    truth_in = np.stack([phi, theta], axis=1) / scales

    shuffle(truth_in)
    shuffle(input_4vec)


    # Split the data into training and testing
    # <HACK, FIXME, NOTE>
    # <HACK, For now a maximum of 10,000 events are used for testing, xju>
    num_test_evts = int(input_4vec.shape[0]*testing_frac)
    if num_test_evts < 10_000: num_test_evts = 10_000

    # <NOTE, https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html>



    test_in, train_in = input_4vec[:num_test_evts], input_4vec[num_test_evts:max_evts]
    test_truth, train_truth = truth_in[:num_test_evts], truth_in[num_test_evts:max_evts]

    xlabels = ['phi', 'theta']

    return (train_in, train_truth, test_in, test_truth, xlabels)

def herwig_angles2(filename,
        max_evts=None, testing_frac=0.1, mode=2):
    """
    This Herwig dataset is for the "ClusterDecayer" study.
    Each event has q1, q1, cluster, h1, h2.
    I define 3 modes:
    0) both q1, q2 are with Pert=1
    1) only one of q1 and q2 is with Pert=1
    2) neither q1 nor q2 are with Pert=1
    3) at least one quark with Pert=1
    """
    if type(filename) == list:
        filename = filename[0]
    arrays = np.load(filename)
    truth_in = arrays['out_truth']
    input_4vec = arrays['input_4vec']

    shuffle(truth_in)
    shuffle(input_4vec)
    print(truth_in.shape, input_4vec.shape)


    # Split the data into training and testing
    # <HACK, FIXME, NOTE>
    # <HACK, For now a maximum of 10,000 events are used for testing, xju>
    num_test_evts = int(input_4vec.shape[0]*testing_frac)
    if num_test_evts < 10_000: num_test_evts = 10_000

    test_in, train_in = input_4vec[:num_test_evts], input_4vec[num_test_evts:max_evts]
    test_truth, train_truth = truth_in[:num_test_evts], truth_in[num_test_evts:max_evts]
    xlabels = ['phi', 'theta']

    return (train_in, train_truth, test_in, test_truth, xlabels)


def dimuon_inclusive(filename, max_evts=None, testing_frac=0.1):
    
    df = read_dataframe(filename, " ", None)
    truth_data = df.to_numpy().astype(np.float32)
    print(f"reading dimuon {df.shape[0]} events from file {filename}")

    scaler = MinMaxScaler(feature_range=(-1,1))
    truth_data = scaler.fit_transform(truth_data)
    # scales = np.array([10, 1, 1, 10, 1, 1], np.float32)
    # truth_data = truth_data / scales

    shuffle(truth_data)

    num_test_evts = int(truth_data.shape[0]*testing_frac)
    if num_test_evts > 10_000: num_test_evts = 10_000


    test_truth, train_truth = truth_data[:num_test_evts], truth_data[num_test_evts:max_evts]

    xlabels = ['leading Muon {}'.format(name) for name in ['pT', 'eta', 'phi']] +\
              ['subleading Muon {}'.format(name) for name in ['pT', 'eta', 'phi']]
    
    return (None, train_truth, None, test_truth, xlabels)

def geant4_leading_products(filename, testing_frac=0.1, train_particle_type=False, max_evts=None):
    df = read_dataframe(filename, sep=' ')
    data = df.to_numpy()
    
    data = data.reshape((data.shape[0], -1, 5))

    n_particles = data.shape[1]

    # from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    # encoder = OneHotEncoder(sparse=False)
    # energy_scaler = MinMaxScaler()
    momentum_scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = MinMaxScaler(feature_range=(-1,1))

    if not train_particle_type:
        data=data[:,:, 1:]
        # energy = data[:,:,0]
        # energy /= np.max(energy)
        # momentum = data[:,:,1:]
        # momentum = momentum.reshape((momentum.shape[0], -1))
        # momentum = momentum_scaler.fit_transform(momentum).reshape((momentum.shape[0], 3, -1))

    data = data.reshape((data.shape[0], -1))
    scaled_data = scaler.fit_transform(data).reshape(data.shape[0], n_particles, -1) 
    # scaled_data = data / 30000       

    # scaled_data = np.concatenate((energy[...,np.newaxis], momentum), axis=-1)
    scaled_input = scaled_data[:, 0]
    scaled_truth = scaled_data[:, 1:]
    scaled_truth = scaled_truth.reshape((scaled_truth.shape[0], -1))

    cutoff = int(data.shape[0] * testing_frac)

    train_input, test_input = scaled_input[cutoff:], scaled_input[:cutoff]
    train_truth, test_truth = scaled_truth[cutoff:], scaled_truth[:cutoff]

    print(f'Train truth shape: {train_truth.shape}')
    print(f'Test truth shape: {test_truth.shape}')


    xlabels = [ "leading_" + i for i in ['E', 'px', 'py', 'pz'] ] + [ "subleading_" + i for i in ['E', 'px', 'py', 'pz'] ]

    return train_input, train_truth, test_input, test_truth, xlabels

def geant4_COM_frame(filename, testing_frac=0.1, train_particle_type=False, max_evts=None, print_info=True, save_transformer=None):
    
    df = read_dataframe(filename, sep=' ')
    data = df.to_numpy()
    data = data.reshape((data.shape[0], -1, 5))

    scaler = MinMaxScaler(feature_range=(-1,1))

    if not train_particle_type:
        data=data[:,:, 1:]
        # energy = data[:,:,0]
        # energy /= np.max(energy)
        # momentum = data[:,:,1:]
        # momentum = momentum.reshape((momentum.shape[0], -1))
        # momentum = momentum_scaler.fit_transform(momentum).reshape((momentum.shape[0], 3, -1))

    condition = data[:, :1, :]
    outcome = data[:, 1:, :]

    outcome[:, :, 3] /= np.pi
    outcome[:, :, 2] /= np.pi

    n_particles = outcome.shape[1]
    n_evts = outcome.shape[0]

    outcome = scaler.fit_transform(outcome.reshape(( n_evts , -1))).reshape((n_evts, n_particles, -1))
    # outcome[:, :, 1] = scaled_outcome[:, :, 1]
    # outcome[:, 0, 0] = 0.139
    # outcome[:, 1, 0] = 0.938
    # scaled_data = data / 30000       

    # scaled_data = np.concatenate((energy[...,np.newaxis], momentum), axis=-1)
    scaled_input = np.array([ [0.139, 0.938, 0] ] * n_evts)
    scaled_truth = outcome.reshape((n_evts, -1))

    cutoff = int(data.shape[0] * testing_frac)

    train_input, test_input = scaled_input[cutoff:], scaled_input[:cutoff]
    train_truth, test_truth = scaled_truth[cutoff:], scaled_truth[:cutoff]

    if print_info: 
        print(f'Train truth shape: {train_truth.shape}')
        print(f'Test truth shape: {test_truth.shape}')

    variables = ['M', 'pT', 'Theta', 'Phi']

    xlabels = [ "leading_" + i for i in variables ] + [ "subleading_" + i for i in variables ]

    return train_input, train_truth, test_input, test_truth, xlabels

def geant4_momentum_transfer(filename, testing_frac=0.1, train_particle_type=False, max_evts=None):

    train_input, train_truth, test_input, test_truth, xlabels = geant4_COM_frame(filename, testing_frac, train_particle_type, max_evts)

    train_truth = train_truth.reshape((train_truth.shape[0], -1, 4))[:, 1:, :].reshape((train_truth.shape[0], -1))
    test_truth = test_truth.reshape((test_truth.shape[0], -1, 4))[:, 1:, :].reshape((test_truth.shape[0], -1))

    xlabels = np.array(xlabels).reshape(( -1, 4))[1:].reshape((-1,))

    return train_input, train_truth, test_input, test_truth, xlabels


def geant4_momentum_transfer_delta_phi(filename, testing_frac=0.1, train_particle_type=False, max_evts=None):

    train_input, train_truth, test_input, test_truth, xlabels = geant4_COM_frame(filename, testing_frac, train_particle_type, max_evts)
    
    def convert_delta_phi(data):
        for idx in range(1, data.shape[1]):
            data[:, idx, -1] -= data[:, 0, -1]
        return data

    train_truth = train_truth.reshape((train_truth.shape[0], -1, 4))
    train_truth = convert_delta_phi(train_truth)[:, 1:, :].reshape((train_truth.shape[0], -1))

    test_truth = test_truth.reshape((test_truth.shape[0], -1, 4))
    test_truth = convert_delta_phi(test_truth)[:, 1:, :]
    n_particles = test_truth.shape[1]
    test_truth = test_truth.reshape((test_truth.shape[0], -1))

    xlabels = []
    variables = ['M', 'pT', 'Theta', 'Phi']
    for i in range(n_particles):
        xlabels += [ f"{j}_{i}" for j in variables ]

    return train_input, train_truth, test_input, test_truth, xlabels


def geant4_momentum_transfer_delta_phi_eta(filename, testing_frac=0.1, train_particle_type=False, max_evts=None):

    train_input, train_truth, test_input, test_truth, xlabels = geant4_COM_frame(filename, testing_frac, train_particle_type, max_evts, False)
    
    def convert_delta_phi(data):
        for idx in range(1, data.shape[1]):
            data[:, idx, -1] -= data[:, 0, -1]
        return data
    
    def convert_eta(data):
        for idx in range(0, data.shape[1]):
            data[:, idx, -2] = -np.log( np.tan( data[:, idx, -2] / 2 * np.pi ) ) / 10
        return data


    train_truth = train_truth.reshape((train_truth.shape[0], -1, 4))
    train_truth = convert_eta(convert_delta_phi(train_truth))[:, 1:, :].reshape((train_truth.shape[0], -1))

    test_truth = test_truth.reshape((test_truth.shape[0], -1, 4))
    test_truth = convert_eta(convert_delta_phi(test_truth))[:, 1:, :]
    n_particles = test_truth.shape[1]
    test_truth = test_truth.reshape((test_truth.shape[0], -1))

    xlabels = []
    variables = ['M', 'pT', 'Eta', 'Delta_Phi']
    for i in range(n_particles):
        xlabels += [ f"{j}_{i}" for j in variables ]

    return train_input, train_truth, test_input, test_truth, xlabels


def geant4_COM_delta_phi_eta_E_pT(filename, testing_frac=0.1, train_particle_type=False, print_info=True, max_evts=None, save_transformer=None, read_transformer=None):
    df = read_dataframe(filename, sep=' ')
    data = df.to_numpy()[:, :-5]
    transformations = df.to_numpy()[:, -5:]
    data, transformations = sklearn.utils.shuffle(data, transformations, random_state=0)
    data = data.reshape((data.shape[0], -1, 5))

    if not train_particle_type:
        data=data[:,:, 1:]

    condition = data[:, :2, :]
    outcome = data[:, 2:, :]

    n_particles = outcome.shape[1]
    n_evts = outcome.shape[0]

    # convert cartesian momenta pT, eta, phi 
    for idx in range(n_particles):
        momentum = outcome[:, idx, :].copy()
        momentum = convert_lorentz_vector(momentum)
        outcome[:, idx, 0] = momentum.E.to_numpy()
        outcome[:, idx, 1] = momentum.pt.to_numpy()
        outcome[:, idx, 2] = momentum.eta.to_numpy()
        outcome[:, idx, 3] = momentum.phi.to_numpy() if idx==0 else momentum.phi.to_numpy() - outcome[:, 0, 3]

    outcome = outcome[:, 1:, :]

    p1 = convert_lorentz_vector(condition[:, 0, :].copy())
    p2 = convert_lorentz_vector(condition[:, 1, :].copy())
    p = p1+p2
    condition = p.tau.to_numpy()[:, np.newaxis]

    scaler = MinMaxScaler(feature_range=(-1,1))
    data = np.concatenate([condition, outcome.reshape((n_evts, -1))], axis=1)
    scaler.fit(data)
    
    if isinstance(read_transformer, str):
        scaler = joblib.load(read_transformer)

    data = scaler.transform(data)
    outcome = data[:, 1:]
    condition = data[:, :1]

    if isinstance(save_transformer, str):
        joblib.dump(scaler, save_transformer) 

    cutoff = int(n_evts * testing_frac)

    train_input, test_input = condition[cutoff:], condition[:cutoff]
    train_truth, test_truth = outcome[cutoff:], outcome[:cutoff]

    xlabels = []
    variables = ['E', 'pT', 'Eta', 'Delta_Phi']
    for i in range(n_particles):
        xlabels += [ f"{j}_{i}" for j in variables ]

    if print_info: 
        print(f'Train truth shape: {train_truth.shape}')
        print(f'Test truth shape: {test_truth.shape}')

    return train_input, train_truth, test_input, test_truth, xlabels

def geant4_COM_eta_E_pT(filename, testing_frac=0.1, train_particle_type=False, print_info=True, max_evts=None, save_transformer=None, read_transformer=None):

    df = read_dataframe(filename, sep=' ')
    data = df.to_numpy()[:, :-5]
    data = data.reshape((data.shape[0], -1, 5))

    if not train_particle_type:
        data=data[:,:, 1:]

    condition = data[:, :2, :]
    outcome = data[:, 2:, :]

    n_particles = outcome.shape[1]
    n_evts = outcome.shape[0]

    # convert cartesian momenta pT, eta, phi 
    for idx in range(n_particles):
        momentum = outcome[:, idx, :].copy()
        momentum = convert_lorentz_vector(momentum)
        outcome[:, idx, 0] = momentum.E.to_numpy()
        outcome[:, idx, 1] = momentum.pt.to_numpy()
        outcome[:, idx, 2] = momentum.eta.to_numpy()
        outcome[:, idx, 3] = momentum.phi.to_numpy() if idx==0 else momentum.phi.to_numpy() - outcome[:, 0, 3]

    outcome = outcome[:, 1:, :-1]
    n_particles = outcome.shape[1]
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(outcome.reshape(( n_evts , -1 )))
    
    if isinstance(read_transformer, str):
        scaler = joblib.load(read_transformer)
    outcome = scaler.transform(outcome.reshape(( n_evts , -1 ))).reshape((n_evts, n_particles, -1))

    if isinstance(save_transformer, str):
        joblib.dump(scaler, save_transformer)
    
    scaled_input = np.array([ [0.139, 0.938, 0] ] * n_evts)
    scaled_truth = outcome.reshape((n_evts, -1))

    cutoff = int(data.shape[0] * testing_frac)

    train_input, test_input = scaled_input[cutoff:], scaled_input[:cutoff]
    train_truth, test_truth = scaled_truth[cutoff:], scaled_truth[:cutoff]

    xlabels = []
    variables = ['E', 'pT', 'Eta']
    for i in range(n_particles):
        xlabels += [ f"{j}_{i}" for j in variables ]

    if print_info: 
        print(f'Train truth shape: {train_truth.shape}')
        print(f'Test truth shape: {test_truth.shape}')

    return train_input, train_truth, test_input, test_truth, xlabels
    



