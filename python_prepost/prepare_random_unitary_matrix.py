import numpy as np
import h5py
from qiskit.quantum_info import random_unitary
import sys
import yaml


if __name__ == '__main__':
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    n_qt = config["qpe_params"]["nq_matrix"]
    h5_name = config["input_hdf5"]

    # n_qt = 3
    # h5_name = 'matrix_and_eigens.h5'

    n_size = 2 ** n_qt

    u_target = random_unitary(n_size)

    w, vec = np.linalg.eig(u_target.data)
    nond_phase_list = np.zeros(n_size)

    for i, eig in enumerate(w):
        eigv = vec[:, i]
        phase_rad = np.atan2(eig.imag, eig.real)
        if phase_rad < 0: phase_rad += 2.0*np.pi
        phase_norm = phase_rad / 2.0 / np.pi
        nond_phase_list[i] = phase_norm

        print('='*64)
        print('Eigen value : {0:.6f} {1:+.6f}i (nond-phase : {2:.8f})'.format(eig.real, eig.imag, phase_norm))
        print('Eigen vector: ')
        for j in range(n_size):
            print('\t{0: .6f} {1:+.6f}i'.format(eigv[j].real, eigv[j].imag))


    print('='*64)
    print('Target U')
    for arr in u_target.data:
        for x in arr:
            print(' {0: .2f}{1:+.2f}i'.format(x.real, x.imag), end='')
        print('/n')


    with h5py.File(h5_name, 'w') as f5:
        f5.create_dataset('matrix_real', data=u_target.data.real)
        f5.create_dataset('matrix_imag', data=u_target.data.imag)
        grp = f5.create_group('eigens')
        for i, eig in enumerate(w):
            grp_eig = grp.create_group('eig_{0:09d}'.format(i))
            grp_eig.create_dataset('eigen_vector_real', data=vec[:,i].real)
            grp_eig.create_dataset('eigen_vector_imag', data=vec[:,i].imag)
            grp_eig.attrs['eigen_val_real'] = eig.real
            grp_eig.attrs['eigen_val_imag'] = eig.imag
            grp_eig.attrs['normalized_phase'] = nond_phase_list[i]




