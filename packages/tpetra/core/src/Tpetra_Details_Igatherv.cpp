// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <iostream>
#include <limits>

#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

void log(int rank, const std::string_view msg, const char *file, int line) {
  std::stringstream ss;
  ss << "[" << rank << "] " << file << ":" << line << " " << msg << "\n";
  std::cerr << ss.str();
}

#define LOG(rank, msg) log(rank, msg, __FILE__, __LINE__);

#include <mpi.h>

#include "Tpetra_Details_Igatherv.hpp"

// #define IGATHERV_DEBUG

namespace Tpetra::Details::igatherv {

      
        // how many ranks between phase 1 roots
        static int ph1_root_offset(int size) {
          return static_cast<int>(std::sqrt(size));
        }
        
        static int ph1_group_count(int size) {
          return (size + ph1_root_offset(size) - 1) / ph1_root_offset(size);
        }
        
        static int ph1_root(int rank, int size) {
          // size of phase 1 group
          int R = ph1_root_offset(size);
        
          // root of the phase 1 group
          return (rank / R) * R;
        }
        
        // return actual size of the phase-1 group that rank is in
        static int ph1_size(int rank, int size) {
        
          // size of phase 1 group
          const int R = ph1_root_offset(size);
        
          // root of the phase 1 group
          const int ph1Root = ph1_root(rank, size);
        
          // size of the this phase 1 group
          return std::min(ph1Root + R, size) - ph1Root;
        }
        
        static bool ph1_is_root(int rank, int size) {
          // size of phase 1 group
          int R = ph1_root_offset(size);
        
          // root of my phase 1 group this phase 1 group
          const int ph1Root = (rank / R) * R;
        
          // Am I a root of my phase 1 group?
          return (rank == ph1Root);
        }
        
        int post(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, const int recvcounts[], const int displs[],
                 MPI_Datatype recvtype, int root, int tag, MPI_Comm comm,
                 Req *req // output request
        ) {
          req->tag = tag;
          req->comm = comm;
          MPI_Comm_rank(comm, &(req->rank));
          MPI_Comm_size(comm, &(req->size));
          req->sendtype = sendtype;
          MPI_Type_size(sendtype, &(req->sendtypeSize));
          req->recvtype = recvtype;
          MPI_Type_size(recvtype, &(req->recvtypeSize));
          req->root = root;
        
          req->sendbuf = sendbuf;
          req->sendcount = sendcount;
          req->displs = displs;

          if (root == req->rank) {
            req->recvcounts = recvcounts;
            req->recvbuf = recvbuf;
          } else {
            req->recvcounts = nullptr;
            req->recvbuf = nullptr;
          }

          req->completed = false;
        
        #ifdef IGATHERV_DEBUG
          if (req->rank == root) {
            std::stringstream ss;
            ss << "recv counts @ " << req->recvcounts << ":";
            for (int i = 0; i < req->size; ++i) {
              ss << " " << req->recvcounts[i];
            }
            LOG(req->rank, ss.str());
          }
        #endif

          return MPI_SUCCESS;
        }
        
        int wait(Req &req) {
          // root of my phase 1 group this phase 1 group
          const int ph1Root = ph1_root(req.rank, req.size);
        
          // Am I a root of my phase 1 group?
          const bool isPh1Root = ph1_is_root(req.rank, req.size);
        
          // size of my phase 1 group
          const int ph1Size = ph1_size(req.rank, req.size);
        
          // number of ranks between roots
          const int R = ph1_root_offset(req.size);
        
        #ifdef IGATHERV_DEBUG
          {
            std::stringstream ss;
            ss << "isPh1Root=" << isPh1Root;
            LOG(req.rank, ss.str());
          }
          MPI_Barrier(req.comm);
        #endif
        
          // Step 0: global root sends recv counts to each phase 1 root
          std::vector<int> groupRecvCounts;
          std::vector<MPI_Request> ph0Reqs;
          // receive group recv counts
          if (isPh1Root) {
            groupRecvCounts.resize(ph1Size);
            MPI_Request ph0Req;
            #ifdef IGATHERV_DEBUG
              {
                std::stringstream ss;
                ss << "ph0 recv(?, " << groupRecvCounts.size() << ", int, " << req.root << ", " << req.tag << ")";
                LOG(req.rank, ss.str());
              }
            #endif
            MPI_Irecv(groupRecvCounts.data(), groupRecvCounts.size(), MPI_INT, req.root,
                      req.tag, req.comm, &ph0Req);
            ph0Reqs.push_back(ph0Req);
          }
        
          if (req.rank == req.root) {
            // send recv counts for each group to that group
            for (int dst = 0; dst < req.size; dst += R) {
              const int count = ph1_size(dst, req.size);
              const void *sendbuf = &(req.recvcounts[dst]);
            #ifdef IGATHERV_DEBUG
              {
                std::stringstream ss;
                ss << "ph0 send(" << sendbuf << ", " << count << ", int, " << dst << ", " << req.tag << ")";
                LOG(req.rank, ss.str());
              }
              {
                std::stringstream ss;
                ss << "ph0 send buf was:";
                for (int i = 0; i < count; ++i) {
                  ss << " " << req.recvcounts[dst+i];
                }
                LOG(req.rank, ss.str());
              }
            #endif
              MPI_Request ph0Req;
              MPI_Isend(sendbuf, count, MPI_INT, dst, req.tag, req.comm, &ph0Req);
              ph0Reqs.push_back(ph0Req);
            }
          }
        
          // wait for phase 0 communication
          if (isPh1Root || req.rank == req.root) {
            MPI_Waitall(ph0Reqs.size(), ph0Reqs.data(), MPI_STATUSES_IGNORE);
          }
        
        #ifdef IGATHERV_DEBUG
          if (isPh1Root) {
            std::stringstream ss;
            ss << "recv counts:";
            for (auto c : groupRecvCounts) {
              ss << " " << c;
            }
            LOG(req.rank, ss.str());
          }
        #endif
        
          // Step 1: Local gather to phase 1 roots
          // phase 1 root has already recieved how much data each rank in the group is
          // going to send
          std::vector<MPI_Request> ph1Rreqs;
          std::vector<char> ph1Recvbuf;
          if (isPh1Root) {
        
            // count up total data to be recieved
            size_t bytesToRecv = 0;
            for (size_t count : groupRecvCounts) {
              bytesToRecv += count * req.sendtypeSize;
            }
        
            // prepare buffer to receive phase 1 data
            ph1Recvbuf.resize(bytesToRecv);
            ph1Rreqs.reserve(ph1Size);
        
            // phase 1 root: recv from local group
            int off = 0;
            for (int i = 0; i < ph1Size; ++i) {
        
              const int src = req.rank + i;
        
              MPI_Request ph1Req;
              const int count = groupRecvCounts[i];
        #ifdef IGATHERV_DEBUG
              {
                std::stringstream ss;
                ss << "ph1 recv(?," << count << ",?," << src << ")";
                LOG(req.rank, ss.str());
              }
        #endif
              MPI_Irecv(&ph1Recvbuf[off], count, req.sendtype, src, req.tag, req.comm,
                        &ph1Req);
              ph1Rreqs.push_back(ph1Req);
              off += count * req.sendtypeSize;
            }
          }
        
          // Send to phase 1 root
        #ifdef IGATHERV_DEBUG
          {
            std::stringstream ss;
            ss << "ph1 send(x," << req.sendcount << ",x," << ph1Root << ")";
            LOG(req.rank, ss.str());
          }
        #endif
          MPI_Request ph1Sreq;
          MPI_Isend(req.sendbuf, req.sendcount, req.sendtype, ph1Root, req.tag,
                    req.comm, &ph1Sreq);
        
          if (isPh1Root) {
        
        #ifdef IGATHERV_DEBUG
            {
              std::stringstream ss;
              ss << "wait for " << ph1Rreqs.size() << " ph1 recvs";
              LOG(req.rank, ss.str());
            }
        #endif
            // wait to recv all data from ph1 group
            std::vector<MPI_Status> stats(ph1Size);
            MPI_Waitall(ph1Rreqs.size(), ph1Rreqs.data(), stats.data());
            // ph1RecvBuf now contains contiguous data from the group
          }
        
          // wait for phase 1 sends to complete
          // everyone participated in this one
        #ifdef IGATHERV_DEBUG
          LOG(req.rank, "wait for ph1 send");
        #endif
          MPI_Wait(&ph1Sreq, MPI_STATUS_IGNORE);
        
          // Step 2: Intermediate processes gather to root
          std::vector<MPI_Request> ph2Reqs;
          std::vector<std::vector<char>> ph2Data; // data from group i
          if (req.rank == req.root) {
            const int nR = ph1_group_count(req.size);
        
            ph2Reqs.reserve(nR + isPh1Root ? 1 : 0); // + 1 send if we're a phase 1 root
            ph2Data.reserve(nR);
        
            // prepare buffers to receive data from each group
            // we can't recv directly into recvbuffer because
            // user may have specified non-contiguous displacements
            // and this data comes in contiguous
            for (int gr = 0; gr < req.size; gr += R) {
              const int ph1GroupSize = ph1_size(gr, req.size);
        
              size_t groupTotalCount = 0;
              for (int r = gr; r < gr + ph1GroupSize; ++r) {
                groupTotalCount += req.recvcounts[r];
              }
        
              ph2Data.emplace_back(groupTotalCount * req.sendtypeSize);
            }
        
            // receive aggregate ph1 data for group i
            for (int i = 0; i < nR; ++i) {
              // operate in terms of provided datatype to help prevent count overflows
              const int count = ph2Data[i].size() / size_t(req.sendtypeSize);
        #ifdef IGATHERV_DEBUG
              {
                std::stringstream ss;
                ss << "ph2 recv(" << (void *)ph2Data[i].data() << ", " << count
                   << ", x," << i * R << ")";
                LOG(req.rank, ss.str());
              }
        #endif
              MPI_Request ph2Req;
              MPI_Irecv(ph2Data[i].data(), count, req.sendtype, i * R, req.tag,
                        req.comm, &ph2Req);
              ph2Reqs.push_back(ph2Req);
            }
          }
        
          if (isPh1Root) {
            // Phase 2: send packed buffer to root
        #ifdef IGATHERV_DEBUG
            {
              std::stringstream ss;
              ss << "ph2 send(x, " << ph1Recvbuf.size() << "B, x," << req.root << ")";
              LOG(req.rank, ss.str());
            }
        #endif
            MPI_Request ph2Req;
            // operate in terms of provided datatype to help prevent count overflows
            MPI_Isend(ph1Recvbuf.data(), ph1Recvbuf.size() / req.sendtypeSize,
                      req.sendtype, req.root, req.tag, req.comm, &ph2Req);
            ph2Reqs.push_back(ph2Req);
          }
        
          if (isPh1Root || req.rank == req.root) {
        // wait for phase 2 to complete
        #ifdef IGATHERV_DEBUG
            LOG(req.rank, "wait for ph2 sends/recvs");
        #endif
            MPI_Waitall(ph2Reqs.size(), ph2Reqs.data(), MPI_STATUSES_IGNORE);
          }
        
          // If the current process is the root, copy the received data to the final
          // recvbuf
          if (req.rank == req.root) {
        
        #ifdef IGATHERV_DEBUG
            {
              std::stringstream ss;
              ss << "fin recvbuf=" << req.recvbuf;
              LOG(req.rank, ss.str());
            }
        #endif
        
            size_t off = 0; // track offset into group
            for (int src = 0; src < req.size; ++src) {
              const int r = src / R; // ph1 group this sender was in
              if (src % R == 0)
                off = 0; // new group starts
        
              // copy counts elements into displacement in recvbuf
              void *ddst = static_cast<char *>(req.recvbuf) +
                           size_t(req.displs[src]) * size_t(req.recvtypeSize);
              const void *dsrc = ph2Data[r].data() + off;
              const size_t dcnt =
                  size_t(req.recvcounts[src]) * size_t(req.recvtypeSize);
        #ifdef IGATHERV_DEBUG
              {
                std::stringstream ss;
                ss << "fin memcpy(" << ddst << ", " << dsrc << ", " << dcnt << ")";
                LOG(req.rank, ss.str());
              }
        #endif
              std::memcpy(ddst, dsrc, dcnt);
              off += dcnt;
            }
          }
        
          req.completed = true;
          return MPI_SUCCESS;
        }

        int get_status(const Req &req, int *flag, MPI_Status */*status*/) {
          *flag = req.completed;
          return MPI_SUCCESS;
        }

} // namespace Tpetra::Details::igatherv