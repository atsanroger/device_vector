#pragma once
#include <mpi.h>
#include <vector>
#include <cstdio>
#include "DeviceVector.cuh"

namespace GPU {

    // Helper for MPI Types
    template <typename T> struct MPITypeTraits;
    template <> struct MPITypeTraits<int>       { static MPI_Datatype type() { return MPI_INT; } };
    template <> struct MPITypeTraits<float>     { static MPI_Datatype type() { return MPI_FLOAT; } };
    template <> struct MPITypeTraits<double>    { static MPI_Datatype type() { return MPI_DOUBLE; } };
    template <> struct MPITypeTraits<long long> { static MPI_Datatype type() { return MPI_LONG_LONG; } };

    class DeviceComm {
    private:
        static MPI_Comm m_comm;
        static int m_rank;
        static int m_size;
        
        static int m_neighbors[6]; 

        DeviceComm() {} // Static only

    public:
        static void init(MPI_Comm comm, int rank, int* neighbors);

        template <typename T>
        static void exchange_halo(IDeviceVector<T>& send_vec, IDeviceVector<T>& recv_vec, int dir) {
            
            if (!DeviceEnv::instance().is_initialized()) return;
            
            int dest = MPI_PROC_NULL;
            int src  = MPI_PROC_NULL;
            
        }

        template <typename T>
        static void sendrecv(IDeviceVector<T>& send_buf, int dest_idx, 
                             IDeviceVector<T>& recv_buf, int src_idx) {
            
            int dest_rank = m_neighbors[dest_idx]; 
            int src_rank  = m_neighbors[src_idx];  

            cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());

            MPI_Sendrecv(send_buf.device_ptr(), send_buf.size(), MPITypeTraits<T>::type(), dest_rank, 0,
                         recv_buf.device_ptr(), recv_buf.size(), MPITypeTraits<T>::type(), src_rank, 0,
                         m_comm, MPI_STATUS_IGNORE);
        }
        
        template <typename T>
        static void allreduce(IDeviceVector<T>& vec, MPI_Op op) {
             cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
             MPI_Allreduce(MPI_IN_PLACE, vec.device_ptr(), vec.size(), 
                           MPITypeTraits<T>::type(), op, m_comm);
        }
    };
}