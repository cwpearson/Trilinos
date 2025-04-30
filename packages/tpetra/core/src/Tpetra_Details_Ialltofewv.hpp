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

namespace Tpetra::Details::ialltofewv {

    struct Req {
        const void *sendbuf;
        const int *sendcounts; 
        const int *sdispls;
        MPI_Datatype sendtype;
        void *recvbuf;
        const int *recvcounts;              
        const int *rdispls;                    
        const int *roots;
        int nroots;
        MPI_Datatype recvtype;
        int tag;
        MPI_Comm comm;

        bool completed;
    };
  
    int post(const void *sendbuf,
                const int *sendcounts, // how much to each root (length nroots)
                const int *sdispls,    // where data for each root starts (length nroots)
                MPI_Datatype sendtype,
                void *recvbuf, // address of recv buffer (significant only at root)
                const int *recvcounts, // the number of elements recvd from each process
                                 // (signficant only at roots)
                const int *rdispls,    // where in `recvbuf` to place incoming data from
                                 // process i (signficant only at roots)
                const int *roots,      // list of root ranks (must be same on all procs)
                int nroots,      // size of list of root ranks
                MPI_Datatype recvtype, 
                int tag,
                MPI_Comm comm,
                Req *req);


    int wait(Req &req);

    int get_status(const Req &req, int *flag, MPI_Status *status);

} // namespace Tpetra::Details::ialltofewv
