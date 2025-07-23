import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
import os
import yaml


# input_file = './matrix_and_eigens.h5'
# sv_file = './statevector_qpe.h5'
# dim_eigvec = 2
# nq_measure = 23

logfile = 'eval_qpe.log'


if __name__ == '__main__':
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    input_file = config["input_hdf5"]
    sv_file = config["output_hdf5"]
    dim_eigvec = config["qpe_params"]["nq_matrix"]
    nq_measure = config["qpe_params"]["nq_measure"]
    
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.basename(__file__)
    data_path = os.path.abspath(os.path.join(file_path, 'data', file_name.split(".")[0]))
    if not os.path.isdir(data_path): os.makedirs(data_path)

    phase_list = []
    with h5py.File(input_file, 'r') as f5:
        root_grp = "eigens"
        for grp in f5[root_grp]:
            phase_list.append(f5["{0}/{1}".format(root_grp, grp)].attrs['normalized_phase'])
    if len(phase_list) > 16:
        phase_list = np.array(phase_list[:16])
    else:
        phase_list = np.array(phase_list)

    error_list_max = np.zeros_like(phase_list)
    error_list_ave = np.zeros_like(phase_list)
    phase_meas_max = np.zeros_like(phase_list)
    phase_meas_ave = np.zeros_like(phase_list)
    elapsed_times = np.zeros_like(phase_list)
    # print(phase_list)

    f = open(logfile, 'w')

    for i, phi in enumerate(phase_list):
        # print('{0}th eigenvalue'.format(i))
        f.write('{0}th eigenvalue {1}\n'.format(i, "="*40))
        f.flush()

        config["qpe_params"]["target_eigen_index"] = i
        with open("config.yaml", 'w') as fw:
            yaml.dump(config, fw, default_flow_style=False)

        res = subprocess.run(
            ['./qpe.out'],
            capture_output = True,
            text = True
        )

        with h5py.File(sv_file, 'r') as f5:
            sv_x_ini = f5['x-real-initial'][()]
            sv_y_ini = f5['y-imaginary-initial'][()]
            sv_x = f5['x-real'][()]
            sv_y = f5['y-imaginary'][()]
            elapsed_times[i] = f5.attrs['Elapsed time [ms]'][()]

        with open("qpe_eval_{0:03d}.log".format(i), "w") as fl:
            fl.write(res.stdout)
            fl.write("{0}\n".format(" ERROR ".center(60, "=")))
            fl.write(res.stderr)

        norm_sv_ini = np.sqrt(np.power(sv_x_ini, 2) + np.power(sv_y_ini, 2))
        norm_sv = np.sqrt(np.power(sv_x, 2) + np.power(sv_y, 2))

        initial_dict = {format(i, 'b') : norm for i, norm in enumerate(norm_sv_ini) if norm > 1e-10}
        initial_len = len(initial_dict)
        final_dict = {format(i, 'b') : norm for i, norm in enumerate(norm_sv)}
        final_dict_extract = sorted(final_dict.items(), key=lambda x:x[1], reverse=True)

        max_len = dim_eigvec + nq_measure

        # print("{0:{width}} {1}".format("in", "out", width=max_len))
        f.write("{0:{width}} {1}\n".format("in", "out", width=max_len))
        f.flush()
        for (k1, v1), (k2, v2) in zip(initial_dict.items(), final_dict_extract):
            # print(k1.zfill(max_len), k2.zfill(max_len))
            f.write("{0} {1} (norm {2:.4f})\n".format(k1.zfill(max_len), k2.zfill(max_len), v2**2))
            f.flush()

        ave_phi_meas = 0.0
        for k, v in final_dict.items():
            meas_bit_txt = k.zfill(max_len)[:-dim_eigvec]
            ave_phi_meas += int(meas_bit_txt, 2) / 2**(max_len - dim_eigvec) * v**2
        phase_meas_ave[i] = ave_phi_meas

        target_bit_text = final_dict_extract[0][0]
        measured_bit_text = target_bit_text.zfill(max_len)[:-dim_eigvec]
        # print(target_bit_text.zfill(max_len)[-1::-1])
        phase_meas_max[i] = int(measured_bit_text, 2) / 2**(max_len - dim_eigvec)
        error_list_max[i] = phase_meas_max[i] - phi
        error_list_ave[i] = phase_meas_ave[i] - phi
        '''
        print('Measured bit order : ', measured_bit_text)
        print('Elapsed time [ms]  : ', elapsed_times[i])
        print('Estimated eigen phase by max : {0:.10f}'.format(phase_meas_max[i]))
        print('Estimated eigen phase by ave : {0:.10f}'.format(phase_meas_ave[i]))
        print('Ref. eigen phase             : {0:.10f}'.format(phi))
        print('Error (max) [%]              : {0:.4}'.format(error_list_max[i] * 100.0))
        print('Error (ave) [%]              : {0:.4}'.format(error_list_ave[i] * 100.0))
        '''
        f.write('Measured bit order : \n{0:<{w}} {1}\n'.format(" ", measured_bit_text, w=max_len))
        f.write('Elapsed time [ms]  : {0}\n'.format(elapsed_times[i]))
        f.write('Estimated eigen phase by max : {0:.10f}\n'.format(phase_meas_max[i]))
        f.write('Estimated eigen phase by ave : {0:.10f}\n'.format(phase_meas_ave[i]))
        f.write('Ref. eigen phase             : {0:.10f}\n'.format(phi))
        f.write('Error (max) [%]              : {0:.4}\n'.format(error_list_max[i] * 100.0))
        f.write('Error (ave) [%]              : {0:.4}\n'.format(error_list_ave[i] * 100.0))
        f.flush()

    with h5py.File('{0}/summary_vecbit_{1}_measbit_{2}.h5'.format(data_path, dim_eigvec, nq_measure), 'w') as f5:
        f5.create_dataset('Reference eigen phases', data=phase_list)
        f5.create_dataset('Estimated eigen phases by maximum norm', data=phase_meas_max)
        f5.create_dataset('Estimated eigen phases by averaging', data=phase_meas_ave)
        f5.create_dataset('Error eigen phases by maximum norm', data=error_list_max)
        f5.create_dataset('Error eigen phases by averaging', data=error_list_ave)
        f5.create_dataset('Elapsed times [ms]', data=elapsed_times)

        f5.attrs['Ave. elapsed time [ms]'] = np.mean(elapsed_times)
        f5.attrs['Ave. error by max'] = np.mean(np.abs(error_list_max))
        f5.attrs['Ave. error by ave'] = np.mean(np.abs(error_list_ave))

    '''
    print(">"*20, "Summary", ">"*20)
    print('Ave. elapsed time [ms]: {0:.4e}'.format(np.mean(elapsed_times)))
    print('Ave. error by max     : {0:.4e}'.format(np.mean(np.abs(error_list_max))))
    print('Ave. error by ave     : {0:.4e}'.format(np.mean(np.abs(error_list_ave))))
    '''
    f.write("{0}\n".format(" Summary ".center(60, '>')))
    f.write('Ave. elapsed time [ms]: {0:.4e}\n'.format(np.mean(elapsed_times)))
    f.write('Ave. error by max     : {0:.4e}\n'.format(np.mean(np.abs(error_list_max))))
    f.write('Ave. error by ave     : {0:.4e}\n'.format(np.mean(np.abs(error_list_ave))))
    f.flush()


    f.close()



