// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <mpi.h>

namespace Tpetra::Details::igatherv {

class Req {

    public:
        int tag;
        MPI_Comm comm;
        int rank;
        int size;
        MPI_Datatype sendtype;
        int sendtypeSize;
        MPI_Datatype recvtype;
        int recvtypeSize;
        int root;
    
        const void *sendbuf;
        int sendcount;
    
        void *recvbuf;
        const int *recvcounts;
        const int *displs;

        bool completed;
    };
    
int post(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, const int recvcounts[], const int displs[],
            MPI_Datatype recvtype, int root, int tag, MPI_Comm comm,
            Req *req);

int wait(Req &req);

int get_status(const Req &req, int *flag, MPI_Status *status);

} // namespace Tpetra::Details::igatherv