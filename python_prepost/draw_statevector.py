import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


# input_file = './matrix_and_eigens.h5'
# sv_file = './statevector_qpe.h5'
# dim_eigvec = 4
# target_index = 0


if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    input_file = config["input_hdf5"]
    sv_file = config["output_hdf5"]
    dim_eigvec = config["qpe_params"]["nq_matrix"]
    target_index = config["qpe_params"]["target_eigen_index"]

    phase_list = []
    with h5py.File(input_file, 'r') as f5:
        root_grp = "eigens"
        for grp in f5[root_grp]:
            phase_list.append(f5["{0}/{1}".format(root_grp, grp)].attrs['normalized_phase'])
    phase_list = np.array(phase_list)

    with h5py.File(sv_file, 'r') as f5:
        sv_x_ini = f5['x-real-initial'][()]
        sv_y_ini = f5['y-imaginary-initial'][()]
        sv_x = f5['x-real'][()]
        sv_y = f5['y-imaginary'][()]

    norm_sv_ini = np.sqrt(np.power(sv_x_ini, 2) + np.power(sv_y_ini, 2))
    norm_sv = np.sqrt(np.power(sv_x, 2) + np.power(sv_y, 2))

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(sv_x_ini, label='real')
    ax1.plot(sv_y_ini, label='imag')
    ax1.plot(norm_sv_ini, label='norm')
    ax1.set_title('Initial coefs for real part of statevector')

    ax2 = fig.add_subplot(212)
    ax2.plot(sv_x, label='real')
    ax2.plot(sv_y, label='imag')
    ax2.plot(norm_sv, label='norm')
    ax2.legend()
    ax2.set_title('Coefs of statevector after QPE circuit')

    plt.tight_layout()
    plt.savefig("sv_norm_qpe.png")
    plt.clf()

    initial_dict = {format(i, 'b') : norm for i, norm in enumerate(norm_sv_ini) if norm > 1e-10}
    final_dict = {format(i, 'b') : norm for i, norm in enumerate(norm_sv) if norm > 1e-10}
    final_dict_extract = sorted(final_dict.items(), key=lambda x:x[1], reverse=True)

    max_len = int( np.log2(sv_x_ini.size) )

    print("{0:{width}} {1:{width}} Norm".format("in", "out", width=max_len))
    for (k1, v1), (k2, v2) in zip(initial_dict.items(), final_dict_extract):
        print(k1.zfill(max_len), k2.zfill(max_len), "({0:.4f})".format(v2))


    target_bit_text = final_dict_extract[0][0]
    measured_bit_text = target_bit_text.zfill(max_len)[:-dim_eigvec]
    # print(target_bit_text.zfill(max_len)[-1::-1])
    print('Measured bit order : \n{0:{width}} {1}'.format(" ", measured_bit_text, width=max_len))
    print('Estimated eigen phase : {0:.12f}'.format(int(measured_bit_text, 2) / 2**(max_len - dim_eigvec)))
    print('Ref. eigen phase      : {0:.12f}'.format(phase_list[target_index]))
    




