// #include "DeviceComm.cuh"

// namespace GPU {

//     MPI_Comm DeviceComm::m_comm = MPI_COMM_WORLD;
//     int DeviceComm::m_rank = 0;
//     int DeviceComm::m_size = 1;
//     int DeviceComm::m_neighbors[6] = {MPI_PROC_NULL}; // E, W, N, S, U, L

//     void DeviceComm::init(MPI_Comm comm, int rank, int* neighbors) {
//         m_comm = comm;
//         m_rank = rank;
//         MPI_Comm_size(comm, &m_size);

//         for(int i=0; i<6; i++) {
//             m_neighbors[i] = neighbors[i];
//         }
        
//         // printf("[GPUComm] Rank %d initialized. Neighbor E=%d\n", m_rank, m_neighbors[0]);
//     }

// }