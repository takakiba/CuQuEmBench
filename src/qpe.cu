#include <H5Cpp.h>            // HDF5 interface
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include "helper.hpp"

constexpr double PI = 3.14159265359;
constexpr double kMillisecondsPerSecond = 1000.0;
constexpr double kMillisecondsPerHour   = 3600.0 * 1000.0;


/*
std::string get_config_param(const std::string& key) {
    YAML::Node config = YAML::LoadFile("config.yaml");
    if (!config[key]) {
        std::cerr << "Key not found in YAML: " << key << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return config[key].as<std::string>();
}
*/


bool loadMatrixAndEigensFromHDF5(
    const std::string& filename,
    std::vector<double>& mat_real,
    std::vector<double>& mat_imag,
    std::vector<double>& eigval_real,
    std::vector<double>& eigval_imag,
    std::vector<double>& vec_real,
    std::vector<double>& vec_imag,
    size_t& ndim) {
    /*
       function to load HDF5 containing,
            - Unitary matrix to get eigenvalue
            - Eigenvalues and eigenvectors for the unitary matrix
    */
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        // names for datasets and groups
        const std::string h5_mat_real = "matrix_real";
        const std::string h5_mat_imag = "matrix_imag";
        const std::string h5_grp      = "eigens";
        const std::string h5_eig_pre  = "eig_";
        const std::string h5_eig_real = "eigen_vector_real";
        const std::string h5_eig_imag = "eigen_vector_imag";
        const std::string h5_eigval_real = "eigen_val_real";
        const std::string h5_eigval_imag = "eigen_val_imag";

        // get problem size
        auto dset_mat_real = file.openDataSet(h5_mat_real);
        auto dspace = dset_mat_real.getSpace();
        std::vector<hsize_t> dims(dspace.getSimpleExtentNdims());
        dspace.getSimpleExtentDims(dims.data());

        ndim = dims[0];

        // allocate array size
        mat_real.resize(ndim * ndim);
        mat_imag.resize(ndim * ndim);
        eigval_real.resize(ndim);
        eigval_imag.resize(ndim);
        vec_real.resize(ndim * ndim);
        vec_imag.resize(ndim * ndim);

        // read matrix data
        dset_mat_real.read(mat_real.data(), H5::PredType::NATIVE_DOUBLE);
        file.openDataSet(h5_mat_imag).read(mat_imag.data(), H5::PredType::NATIVE_DOUBLE);

        for (size_t i = 0; i < ndim; ++i) {
            std::ostringstream oss;
            oss << h5_grp << "/" << h5_eig_pre 
                << std::setw(9) << std::setfill('0') << i;
            auto eig_grp = file.openGroup(oss.str());

            // read eigenvalue from hdf5 attribute
            eig_grp.openAttribute(h5_eigval_real).read(H5::PredType::NATIVE_DOUBLE, &eigval_real[i]);
            eig_grp.openAttribute(h5_eigval_imag).read(H5::PredType::NATIVE_DOUBLE, &eigval_imag[i]);

            // read eigenvector from hdf5 dataset
            std::vector<double> tmp_real(ndim), tmp_imag(ndim);
            eig_grp.openDataSet(h5_eig_real).read(tmp_real.data(), H5::PredType::NATIVE_DOUBLE);
            eig_grp.openDataSet(h5_eig_imag).read(tmp_imag.data(), H5::PredType::NATIVE_DOUBLE);

            // copy eigenvector data into array
            for (size_t j = 0; j < ndim; ++j) {
                vec_real[i * ndim + j] = tmp_real[j];
                vec_imag[i * ndim + j] = tmp_imag[j];
            }
        }
    } catch (const H5::Exception& e) {
        std::cerr << "HDF5 error: " << e.getCDetailMsg() << "\n";
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return false;
    }

    return true;
}


void update_progress(int current, int total, std::chrono::steady_clock::time_point start){
    /*
       show progress bar based on counter 'current' and 'total' loop number
    */
    static int last_percent = -1;
    static bool is_tty = isatty(fileno(stdout));

    double progress = (double)(current+1) / total;
    int percent = static_cast<int>(progress * 100.0);

    if (percent == last_percent) return;
    last_percent = percent;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
    double eta_raw = (current > 0 && current < total - 1)
        ? elapsed * (1.0 / progress - 1.0) : 0.0;
    int eta = static_cast<int>(std::ceil(eta_raw));

    if (is_tty) {
        std::cout << "\r[" << std::setw(3) << percent << "%] "
            << (percent < 100 ? "ETA: ": "Elapsed: ")
            << (percent < 100 ? eta: elapsed) << "s          ";
        std::cout.flush();
    } else {
        std::cout << "[" << std::setw(3) << percent << "%] "
            << (percent < 100 ? "ETA: " : "Elapsed: ")
            << (percent < 100 ? eta: elapsed) << "s" <<std::endl;
    }
}


int main(int argc, char* argv[]){
    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // import calc paramter from input
    YAML::Node config;
    try {
        config = YAML::LoadFile("config.yaml");
    } catch (const std::exception& e) {
        std::cerr << "Failed to load config.yaml: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    const std::string input_h5 = config["input_hdf5"].as<std::string>();
    const std::string output_h5 = config["output_hdf5"].as<std::string>();

    int nq_meas, target_eig_index;

    try {
        YAML::Node qpe_params = config["qpe_params"];
        nq_meas = qpe_params["nq_measure"].as<int>();
        target_eig_index = qpe_params["target_eigen_index"].as<int>();
        if (nq_meas <= 0 || target_eig_index < 0) {
            throw std::invalid_argument("nq_measure must be > 0 and target_eigen_index >= 0");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading integer parameters from config.yaml: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::chrono::system_clock::time_point t0_rh5, t1_rh5, t0_wh5_ini, t1_wh5_ini, t0_wh5, t1_wh5;

    // >>>>>>>>>>>>>>>>>>> Read external matrix HDF5 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    std::vector<double> mat_real, mat_imag, eigval_real, eigval_imag, vec_real, vec_imag;
    size_t ndim = 0;

    t0_rh5 = std::chrono::system_clock::now();

    if (!loadMatrixAndEigensFromHDF5( input_h5,
                mat_real, mat_imag,
                eigval_real, eigval_imag,
                vec_real, vec_imag,
                ndim)) {
        std::cerr << "Failed to load data from HDF5.\n";
        return EXIT_FAILURE;
    }

    t1_rh5 = std::chrono::system_clock::now();
    double elapsed_rh5 = std::chrono::duration_cast<std::chrono::microseconds>(t1_rh5-t0_rh5).count();

    int nq_vecs = static_cast<int>( std::log2(ndim) );
    // <<<<<<<<<<<<<<<<<<< End read external matrix HDF5 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    std::vector<cuDoubleComplex> target_mat( ndim * ndim );
    for (int i=0; i<ndim*ndim; i++){
        target_mat[i] = {mat_real[i], mat_imag[i]};
    }
    cuDoubleComplex* d_target_mat;
    size_t mat_size = target_mat.size() * sizeof(cuDoubleComplex);

    // int nv_meas = (1 << nq_meas);
    const int nQubits = nq_meas + nq_vecs;

    const std::string h5_r_init = "x-real-initial";
    const std::string h5_i_init = "y-imaginary-initial";
    const std::string h5_real = "x-real";
    const std::string h5_imag = "y-imaginary";

    const int nSvSize = (1 << nQubits);
    const int adjoint = 0;
    const cuDoubleComplex mat_hadmard[] = {{1/std::sqrt(2), 0}, { 1/std::sqrt(2), 0},
                                           {1/std::sqrt(2), 0}, {-1/std::sqrt(2), 0}};
    /*
    const cuDoubleComplex mat_swap[]    = {{1, 0}, {0, 0}, {0, 0}, {0, 0},
                                           {0, 0}, {0, 0}, {1, 0}, {0, 0},
                                           {0, 0}, {1, 0}, {0, 0}, {0, 0},
                                           {0, 0}, {0, 0}, {0, 0}, {1, 0}};
    */
    // const cuDoubleComplex mat_not[] = {{0, 0}, { 1, 0},
    //                                    {1, 0}, { 0, 0}};
    std::vector<cuDoubleComplex> h_sv( nSvSize );
    // >>>>>>>>>>>>>>>>>>>>>> Prepare variables before loop >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // for not gate
    // int nTgt_not = 1;
    // int nCtl_not = 0;
    // int targets_not[] = {0};
    // int controls_not[] = {};
    /*
    // for SWAP gate
    int nTgt_swap = 2;
    int nCtl_swap = 0;
    int targets_swap[] = {0, 1};
    int controls_swap[] = {};
    */
    // for Hadmard gate
    int nTgt_hadmard = 1;
    int nCtl_hadmard = 0;
    int targets_hadmard[] = {0};
    int controls_hadmard[] = {};
    // for Rotation gate
    int nTgt_rot = 1;
    int nCtl_rot = 1;
    int targets_rot[] = {0};
    int controls_rot[] = {1};
    // for target matrix
    int nTgt_mat = nq_vecs;
    int nCtl_mat = 1;
    int targets_mat[nq_vecs];
    int controls_mat[] = {1};
    for (int i=0; i<nq_vecs; i++){
        targets_mat[i] = i;
    }
    // <<<<<<<<<<<<<<<<<<<<<< Prepare variables before loop ><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    cuDoubleComplex mat_rot[] = {{1, 0}, { 0, 0},
                                 {0, 0}, { 1, 0}};

    float norm2 = 0.0;
    float norm = std::sqrt(norm2);
    std::vector<double> svx( nSvSize );
    std::vector<double> svy( nSvSize );
    hsize_t dims[] = {static_cast<hsize_t>(nSvSize)};
    H5::DataSpace dspace( sizeof(dims) / sizeof(hsize_t), dims );

    // >>>>>>>>>>>>>>>>>>>>> Statevec initialization >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for (int i=0; i<nSvSize; i++){
        h_sv[i] = {0, 0};
    }
    /*
    printf("Eigenvectors : \n");
    for (int i=0; i<ndim; i++){
        for (int j=0; j<ndim; j++){
            printf("  % .4f %+.4fi", vec_real[i + j*ndim], vec_imag[i + j*ndim]);
        }
        printf("\n");
    }
    */
    printf("initial statevector\n");
    for (int i=0; i<ndim; i++){
        h_sv[i] = { vec_real[ target_eig_index * ndim + i ], vec_imag[ target_eig_index * ndim + i ]};
        printf("index %d : % .4f %+.4f\n", i, vec_real[ target_eig_index * ndim + i ], vec_imag[ target_eig_index * ndim + i ]);
    }
    // <<<<<<<<<<<<<<<<<<<<<< End statevec initialization <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
   
    cuDoubleComplex *d_sv;

    HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_sv, nSvSize * sizeof(cuDoubleComplex)) );

    HANDLE_CUDA_ERROR( cudaMemcpy(d_sv, h_sv.data(), nSvSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_target_mat, mat_size) );

    HANDLE_CUDA_ERROR( cudaMemcpy(d_target_mat, target_mat.data(), mat_size, cudaMemcpyHostToDevice) );

    // custatevec handle initialization
    custatevecHandle_t handle;
    HANDLE_ERROR( custatevecCreate(&handle) );

    void* extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    // >>>>>>>>>>>>>>>>>>>>> Statevec initialization by gates >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    HANDLE_CUDA_ERROR( cudaMemcpy(h_sv.data(), d_sv, nSvSize * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost) );
    // <<<<<<<<<<<<<<<<<<<<<< End statevec initialization by gates <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // >>>>>>>>>>>>>>>>>>>>>> Write iniital state vector in HDF5 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    t0_wh5_ini = std::chrono::system_clock::now();

    printf("Exporting HDF5 ...");
    for (int i=0; i<nSvSize; i++) {
        svx[i] = h_sv[i].x;
        svy[i] = h_sv[i].y;
    }

    H5::H5File filei( output_h5, H5F_ACC_TRUNC );

    H5::DataSet dset_x_ini = filei.createDataSet( h5_r_init, H5::PredType::NATIVE_DOUBLE, dspace );
    H5::DataSet dset_y_ini = filei.createDataSet( h5_i_init, H5::PredType::NATIVE_DOUBLE, dspace );
    dset_x_ini.write( svx.data(), dset_x_ini.getDataType() );
    dset_y_ini.write( svy.data(), dset_y_ini.getDataType() );

    dset_x_ini.close();
    dset_y_ini.close();
    filei.close();

    t1_wh5_ini = std::chrono::system_clock::now();
    double elapsed_wh5_ini = std::chrono::duration_cast<std::chrono::microseconds>(t1_wh5_ini-t0_wh5_ini).count();
    printf("Done !\n");
    // <<<<<<<<<<<<<<<<<<<<<< End HDF5 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    // >>>>>>>>>>>>>>>>>>>>>> Begin inverse QFT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    printf("Apply QPE\n");
    // Timer preparation
    cudaEvent_t cu_start, cu_stop;
    cudaEventCreate(&cu_start);
    cudaEventCreate(&cu_stop);
    cudaEventRecord(cu_start);
    // >>>>>>>>>>>>>>>>>>>>>> Begin Hadmard gate >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    targets_hadmard[0] = nq_vecs;
    // check the size of external workspace
    HANDLE_ERROR( custatevecApplyMatrixGetWorkspaceSize(
        handle, CUDA_C_64F, nQubits, mat_hadmard, CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, nTgt_hadmard, nCtl_hadmard,
        CUSTATEVEC_COMPUTE_64F, &extraWorkspaceSizeInBytes) );

    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
        HANDLE_CUDA_ERROR( cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes) );

    for (int i=0; i<nq_meas; i++){
        targets_hadmard[0] = nq_vecs + i;
        printf("Hadmard target : %d\n", nq_vecs + i);
        // apply gate
        HANDLE_ERROR( custatevecApplyMatrix(
            handle, d_sv, CUDA_C_64F, nQubits, mat_hadmard, CUDA_C_64F,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets_hadmard, nTgt_hadmard, controls_hadmard,
            nullptr, nCtl_hadmard, CUSTATEVEC_COMPUTE_64F,
            extraWorkspace, extraWorkspaceSizeInBytes) );
    }
    // <<<<<<<<<<<<<<<<<<<<<< End Hadmard gate <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // >>>>>>>>>>>>>>>>>>>>>> Begin applying target matrix >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    printf("Applying Unitary for bits : ");
    for (int i=0; i<nq_vecs; i++) {
        printf("%d, ", targets_mat[i]);
    }
    printf("\n");

    controls_mat[0] = nQubits - 1;
    // check the size of external workspace
    HANDLE_ERROR( custatevecApplyMatrixGetWorkspaceSize(
        handle, CUDA_C_64F, nQubits, d_target_mat, CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, nTgt_mat, nCtl_mat,
        CUSTATEVEC_COMPUTE_64F, &extraWorkspaceSizeInBytes) );
    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
        HANDLE_CUDA_ERROR( cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes) );

    for (int i=0; i<nq_meas; i++){
        controls_mat[0] = nQubits - i - 1;
        printf("Control : %d\n", nQubits - i - 1);
        printf("Applying Unitary %.0f times\n", std::pow(2, i));

        int total = std::pow(2, i);
        auto start = std::chrono::steady_clock::now();

        for (int j=0; j<std::pow(2, i); j++) {
            update_progress(j, total, start);
            // apply gate
            HANDLE_ERROR( custatevecApplyMatrix(
                handle, d_sv, CUDA_C_64F, nQubits, d_target_mat, CUDA_C_64F,
                CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets_mat, nTgt_mat, controls_mat,
                nullptr, nCtl_mat, CUSTATEVEC_COMPUTE_64F,
                extraWorkspace, extraWorkspaceSizeInBytes) );
        }
        std::cout << std::endl;
    }

    // <<<<<<<<<<<<<<<<<<<<<< End applying target matrix <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // Starting inverse QFT

    // check the size of external workspace
    controls_rot[0] = nq_vecs;
    targets_rot[0] = nq_vecs;
    HANDLE_ERROR( custatevecApplyMatrixGetWorkspaceSize(
        handle, CUDA_C_64F, nQubits, mat_rot, CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, nTgt_rot, nCtl_rot,
        CUSTATEVEC_COMPUTE_64F, &extraWorkspaceSizeInBytes) );

    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
        HANDLE_CUDA_ERROR( cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes) );

    for (int i=0; i < nq_meas; i++) {
        printf("Loop i : %d\n", i);
        // >>>>>>>>>>>>>>>>>>>>>> Start rotation gate application >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        for (int j=0; j < i; j++){
            printf("Rotation gate    : %.*f\n", 0, std::pow(2, i-j+1));
            printf("Rotation control : %d\n", nq_vecs + j);
            printf("Rotation target  : %d\n\n", nq_vecs + i);
            mat_rot[3] = {std::cos(2.0*PI/std::pow(2, i-j+1)), -1.0 * std::sin(2.0*PI/std::pow(2, i-j+1))};
            controls_rot[0] = nq_vecs + j;
            targets_rot[0] = nq_vecs + i;

            // apply gate
            HANDLE_ERROR( custatevecApplyMatrix(
                handle, d_sv, CUDA_C_64F, nQubits, mat_rot, CUDA_C_64F,
                CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets_rot, nTgt_rot, controls_rot,
                nullptr, nCtl_rot, CUSTATEVEC_COMPUTE_64F,
                extraWorkspace, extraWorkspaceSizeInBytes) );
        }
        // <<<<<<<<<<<<<<<<<<<<<< End  rotation gate application <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // >>>>>>>>>>>>>>>>>>>>>> Start hadmard gate application >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        targets_hadmard[0] = nq_vecs + i;
        printf("Hadmard target : %d\n", nq_vecs + i);
        // check the size of external workspace
        HANDLE_ERROR( custatevecApplyMatrixGetWorkspaceSize(
            handle, CUDA_C_64F, nQubits, mat_hadmard, CUDA_C_64F,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, nTgt_hadmard, nCtl_hadmard,
            CUSTATEVEC_COMPUTE_64F, &extraWorkspaceSizeInBytes) );

        // allocate external workspace if necessary
        if (extraWorkspaceSizeInBytes > 0)
            HANDLE_CUDA_ERROR( cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes) );

        // apply gate
        HANDLE_ERROR( custatevecApplyMatrix(
            handle, d_sv, CUDA_C_64F, nQubits, mat_hadmard, CUDA_C_64F,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets_hadmard, nTgt_hadmard, controls_hadmard,
            nullptr, nCtl_hadmard, CUSTATEVEC_COMPUTE_64F,
            extraWorkspace, extraWorkspaceSizeInBytes) );
        // <<<<<<<<<<<<<<<<<<<<<< End   hadmard gate application <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    }

    // <<<<<<<<<<<<<<<<<<<<<< End   inverse QFT <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    // destroy handle
    HANDLE_ERROR( custatevecDestroy(handle) );

    HANDLE_CUDA_ERROR( cudaMemcpy(h_sv.data(), d_sv, nSvSize * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost) );

    // for (int i=0; i<nSvSize; i++) {
    //     printf("% f  %+f i\n", h_sv[i].x, h_sv[i].y);
    // }

    HANDLE_CUDA_ERROR( cudaFree(d_sv) );
    HANDLE_CUDA_ERROR( cudaFree(d_target_mat) );

    if(extraWorkspaceSizeInBytes)
        HANDLE_CUDA_ERROR( cudaFree(extraWorkspace) );

    cudaEventRecord(cu_stop);
    cudaEventSynchronize(cu_stop);

    std::cout << "Elapsed time for reading input HDF5 [us] : " 
        << std::fixed << std::setprecision(0) << elapsed_rh5 << std::endl;

    float elapsed_time = 0, print_time = 0;
    cudaEventElapsedTime(&elapsed_time, cu_start, cu_stop);
    if (elapsed_time < kMillisecondsPerSecond){
        print_time = elapsed_time;
        std::cout << "GPU time : " << std::fixed << std::setprecision(2) << print_time << " [ms]" << std::endl;
    } else if (elapsed_time < kMillisecondsPerHour){
        print_time = elapsed_time / kMillisecondsPerSecond;
        std::cout << "GPU time : " << std::fixed << std::setprecision(2) << print_time << " [s]" << std::endl;
    } else {
        print_time = elapsed_time / kMillisecondsPerHour;
        std::cout << "GPU time : " << std::fixed << std::setprecision(2) << print_time << " [h]" << std::endl;

    }

    std::cout << "Elapsed time for writing initial SV [us] : " 
        << std::fixed << std::setprecision(0) << elapsed_wh5_ini << std::endl;

    cudaEventDestroy(cu_start);
    cudaEventDestroy(cu_stop);

    t0_wh5 = std::chrono::system_clock::now();

    // >>>>>>>>>>>>>>>>>>>>>> Write HDF5 output >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    printf("Exporting HDF5 ...\n");
    for (int i=0; i<nSvSize; i++) {
        svx[i] = h_sv[i].x;
        svy[i] = h_sv[i].y;
    }

    H5::H5File filew( output_h5, H5F_ACC_RDWR );

    H5::DataSet dset_x = filew.createDataSet( h5_real, H5::PredType::NATIVE_DOUBLE, dspace );
    H5::DataSet dset_y = filew.createDataSet( h5_imag, H5::PredType::NATIVE_DOUBLE, dspace );
    dset_x.write( svx.data(), dset_x.getDataType() );
    dset_y.write( svy.data(), dset_y.getDataType() );

    H5::DataSpace attr_dspace(H5S_SCALAR);
    H5::Attribute attr = filew.createAttribute("Elapsed time [ms]", H5::PredType::NATIVE_FLOAT, attr_dspace);
    attr.write(H5::PredType::NATIVE_FLOAT, &elapsed_time);

    t1_wh5 = std::chrono::system_clock::now();
    double elapsed_wh5 = std::chrono::duration_cast<std::chrono::microseconds>(t1_wh5-t0_wh5).count();

    std::cout << "Elapsed time for writing output HDF5 [us]: " 
        << std::fixed << std::setprecision(0) << elapsed_wh5 << std::endl;

    printf("Done !\n");

    // <<<<<<<<<<<<<<<<<<<<<< End HDF5 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    return EXIT_SUCCESS;
}



