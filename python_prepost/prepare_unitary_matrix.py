import numpy as np
import h5py
import yaml


if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    n_qt = config["qpe_params"]["nq_matrix"]

    h5_name = config["input_hdf5"]
    n_size = 2 ** n_qt

    phis = np.arange(n_size) / n_size

    eigs = np.exp(1j * 2.0 * np.pi * phis)
    u_diag = np.diag(eigs)

    P = np.zeros_like(u_diag)
    v_base = np.zeros(n_size)

    for i in range(n_size):
        v_base[i] = 1.0
        P[:, i] = v_base[:]

    print(P)

    u_target = (P @ u_diag) @ np.linalg.inv(P)

    w, vec = np.linalg.eig(u_target)

    for i, eig in enumerate(w):
        eigv = vec[:, i]
        phase = np.atan2(eig.imag, eig.real)
        if phase < 0: phase += 2.0*np.pi
        norm_phase = phase / 2.0 / np.pi
        print('='*64)
        print('Eigen value : {0:.6f} {1:+.6f}i'.format(eig.real, eig.imag))
        print('Eigen vector [{0:d}]: {1:.6f} {2:+.6f}i'.format(i, eigv[0].real, eigv[0].imag))
        print('Ref. phase : {:.8f}'.format(norm_phase))


    if n_qt < 5:
        print('='*64)
        print('Target U')
        for arr in u_target:
            print(arr)


    with h5py.File(h5_name, 'w') as f5:
        f5.create_dataset('matrix_real', data=u_target.real)
        f5.create_dataset('matrix_imag', data=u_target.imag)
        grp = f5.create_group('eigens')
        for i, eig in enumerate(w):
            grp_eig = grp.create_group('eig_{0:09d}'.format(i))
            grp_eig.create_dataset('eigen_vector_real', data=vec[:,i].real)
            grp_eig.create_dataset('eigen_vector_imag', data=vec[:,i].imag)
            grp_eig.attrs['eigen_val_real'] = eig.real
            grp_eig.attrs['eigen_val_imag'] = eig.imag
            grp_eig.attrs['normalized_phase'] = phis[i]




