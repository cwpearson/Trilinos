// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Tpetra_Details_Ialltofewv.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <mpi.h>
#include <Kokkos_Core.hpp>

#ifndef NDEBUG
#include <iostream>
#include <sstream>
#endif

namespace {

  struct ProfilingRegion {
    ProfilingRegion() = delete;
    ProfilingRegion(const ProfilingRegion &other) = delete;
    ProfilingRegion(ProfilingRegion &&other) = delete;
    
    ProfilingRegion(const std::string &name) {
        Kokkos::Profiling::pushRegion(name);
    }
    ~ProfilingRegion() {
        Kokkos::Profiling::popRegion();
    }
};

  struct MemcpyArg {
    void *dst;
    void *src;
    size_t count;
};

#if 0
KOKKOS_INLINE_FUNCTION void serial_memcpy(void *dst, void *const src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        static_cast<char *>(dst)[i] = static_cast<char const *>(src)[i];
    }
}
#endif


template <typename T>
KOKKOS_INLINE_FUNCTION bool aligned(const void *dst, const void *src, size_t count) {
  if (0 != (uintptr_t(dst) % sizeof(T))) {
    return false;
  }
  if (0 != (uintptr_t(src) % sizeof(T))) {
    return false;
  }
  if (0 != (count % sizeof(T))) {
    return false;
  }

  return true;
}

template <typename T, typename Member>
KOKKOS_INLINE_FUNCTION void team_memcpy_as(const Member &member, void *dst, void *const src, size_t count) {
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, count),
        [&] (size_t i) {
            reinterpret_cast<T *>(dst)[i] = reinterpret_cast<T const *>(src)[i];
        }
    );
}

template <typename Member>
KOKKOS_INLINE_FUNCTION void team_memcpy(const Member &member, void *dst, void *const src, size_t count) {
    if (aligned<uint64_t>(dst, src, count)) {
        team_memcpy_as<uint64_t>(member, dst, src, count / sizeof(uint64_t));
    } else if (aligned<uint32_t>(dst, src, count)) {
        team_memcpy_as<uint32_t>(member, dst, src, count / sizeof(uint32_t));
    } else {
        team_memcpy_as<uint8_t>(member, dst, src, count);
    }
}

template <typename Member>
KOKKOS_INLINE_FUNCTION void team_memcpy(const Member &member, MemcpyArg &arg) {
    if (aligned<uint64_t>(arg.dst, arg.src, arg.count)) {
        team_memcpy_as<uint64_t>(member, arg.dst, arg.src, arg.count / sizeof(uint64_t));
    } else if (aligned<uint32_t>(arg.dst, arg.src, arg.count)) {
        team_memcpy_as<uint32_t>(member, arg.dst, arg.src, arg.count / sizeof(uint32_t));
    } else {
        team_memcpy_as<uint8_t>(member, arg.dst, arg.src, arg.count);
    }
}

}

namespace Tpetra::Details::ialltofewv {
  
template<typename RecvExecSpace>
int wait_impl(Req &req) {

  auto finalize = [&]() {
    req.completed=true;
    return MPI_SUCCESS;
  };

  // no one is sending anything
  if (0 == req.nroots) {
    return finalize();
  }

  ProfilingRegion pr("alltofewv::wait");

  const int rank = [&]() -> int {
    int _rank;
    MPI_Comm_rank(req.comm, &_rank);
    return _rank;
  }();

  const int size = [&]() -> int {
    int _size;
    MPI_Comm_size(req.comm, &_size);
    return _size;
  }();

  const size_t sendSize = [&]() -> size_t {
    int _size;
    MPI_Type_size(req.sendtype, &_size);
    return _size;
  }();

  const size_t recvSize = [&]() -> size_t {
    int _size;
    MPI_Type_size(req.recvtype, &_size);
    return _size;
  }();
  

    // Balance the number of incoming messages at each phase:
    // Aggregation = size / naggs * nroots
    // Root =        naggs
    // so
    // size / naggs * nroots = naggs
    // size * nroots = naggs^2
    // naggs = sqrt(size * nroots)
    const int naggs = std::sqrt(size_t(size) * size_t(req.nroots)) + /*rounding*/ 0.5;
  
    // how many srcs go to each aggregator
    const int srcsPerAgg = (size + naggs - 1) / naggs;
  
    // the aggregator I send to
    const int myAgg = rank / srcsPerAgg * srcsPerAgg;
  
    // is this rank a root? linear search - nroots expected to be small
    const bool isRoot = std::find(req.roots, req.roots + req.nroots, rank) !=  req.roots + req.nroots;

    #ifndef NDEBUG
    // {
    //   std::stringstream ss;
    //   ss << __FILE__ << ":" << __LINE__ 
    //      << " [" << rank << "]"
    //      << " req.devAccess=" << req.devAccess
    //      << " naggs=" << naggs
    //      << " srcsPerAgg=" << srcsPerAgg
    //      << " myAgg=" << myAgg
    //      << " req.nroots=" << req.nroots
    //      << " req.roots=[";
    //   for (int i = 0; i < req.nroots; ++i) {ss << " " << req.roots[i];}
    //   ss << "]"
    //      << " isRoot=" << isRoot
    //      << "\n";
    //   std::cerr << ss.str();
    // }
  #endif

    // ensure aggregators know how much data each rank is sending to the root
    // [si * nroots + r1] -> how much rank si wants to send to root ri
    std::vector<int> groupSendCounts(req.nroots * srcsPerAgg);
    std::vector<MPI_Request> reqs;
    if (rank == myAgg) {
      reqs.reserve(srcsPerAgg);
      // recv counts from each member of my group
      for (int si = 0; si < srcsPerAgg && si + rank < size; ++si) {
        MPI_Request rreq;

#ifndef NDEBUG
        if (size_t(si) * req.nroots + req.nroots > groupSendCounts.size()) {
          std::stringstream ss;
          ss << __FILE__ << ":" << __LINE__ 
             << " [" << rank << "]"
             << " OOB\n";
          std::cerr << ss.str();
        }
#endif
        MPI_Irecv(&groupSendCounts[si * req.nroots], req.nroots, MPI_INT, si + rank, 
                  req.tag, req.comm, &rreq);
        reqs.push_back(rreq);
      }
    }
    // send sendcounts to aggregator
    MPI_Send(req.sendcounts, req.nroots, MPI_INT, myAgg, req.tag, req.comm);
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    reqs.resize(0);

#ifndef NDEBUG
    // if (rank == myAgg) {
    //   std::stringstream ss;
    //   ss << __FILE__ << ":" << __LINE__ 
    //      << " [" << rank << "] groupSendCounts=";
    //   for (auto e : groupSendCounts) {
    //     ss << e << " ";
    //   }
    //   ss << "\n";
    //   std::cerr << ss.str();
    // }
#endif

    // at this point, in each aggregator, groupSendCounts holds the send counts
    // from each member of the aggregator. The first nroots entries are from the first rank,
    // the second nroots entries are from the second rank, etc

    // a temporary buffer to aggregate data. Data for a root is contiguous.
    // this buffer can always be on the host
    using AggBuf = Kokkos::View<char*, typename RecvExecSpace::memory_space>;
    AggBuf aggBuf("aggBuf");
    std::vector<size_t> rootCount(req.nroots, 0); // [ri] the count of data held for root ri
    if (rank == myAgg) {
      size_t aggBytes = 0;
      for (int si = 0; si < srcsPerAgg && si + rank < size; ++si) {
        for (int ri = 0; ri < req.nroots; ++ri) {
          int count = groupSendCounts[si * req.nroots + ri];
          rootCount[ri] += count;
          aggBytes += count * sendSize;
        }
      }

  #ifndef NDEBUG
      // {
      //   std::stringstream ss;
      //   ss << __FILE__ << ":" << __LINE__ 
      //      << " [" << rank << "] rootCount=";
      //   for (auto e : rootCount) {
      //     ss << e << " ";
      //   }
      //   ss << "\n";
      //   std::cerr << ss.str();
      // }
  #endif

#ifndef NDEBUG
      // {
      //   std::stringstream ss;
      //   ss << __FILE__ << ":" << __LINE__ 
      //      << " [" << rank << "] aggBuf.resize(" << aggBytes << ")\n";
      //   std::cerr << ss.str();
      // }
  #endif
      Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing), aggBuf, aggBytes);
    }
    // now, on the aggregator ranks,
    // * aggBuf is resized to accomodate all incoming data
    // * rootCount holds how much data i hold for each root
    


    // Send the actual data to the aggregator
    reqs.reserve(srcsPerAgg);
    if (rank == myAgg) {
      reqs.reserve(srcsPerAgg + req.nroots);
      // receive from all ranks in my group
      size_t displ = 0;
      // senders will send in root order, so we will recv in that order as well
      // this puts all data for a root contiguous in the aggregation buffer
      for (int ri = 0; ri < req.nroots; ++ri) {
        for (int si = 0; si < srcsPerAgg && si + rank < size; ++si) {
          // receive data for the ri'th root from the si'th sender
          const int count = groupSendCounts[si * req.nroots + ri];
          if (count) {
#ifndef NDEBUG
            // {
            //   std::stringstream ss;
            //   ss << __FILE__ << ":" << __LINE__ 
            //   << " [" << rank << "] ph1 recv(@" << displ << ", " << count << ", ..., " << si+rank << "\n";
            //   std::cerr << ss.str();
            // }
#endif
#ifndef NDEBUG
            if (displ + count * sendSize > aggBuf.size()) {
              std::stringstream ss;
              ss << __FILE__ << ":" << __LINE__ 
              << " [" << rank << "] OOB\n";
              std::cerr << ss.str();
            }
#endif
            MPI_Request rreq;
            // &aggBuf(displ) is causing a memory access violation
            MPI_Irecv(aggBuf.data() +displ, count, req.sendtype, si + rank, req.tag, req.comm, &rreq);
            reqs.push_back(rreq);
            displ += size_t(count) * sendSize;
          }
        }
      } 
    } else {
      reqs.reserve(req.nroots); // prepare for one send per root
    }
    
    // send data to aggregator
    for (int ri = 0; ri < req.nroots; ++ri) {
      const size_t displ = size_t(req.sdispls[ri]) * sendSize;
      const int count = req.sendcounts[ri];
      if (count) {
#ifndef NDEBUG
        // {
        //   std::stringstream ss;
        //   ss << __FILE__ << ":" << __LINE__ 
        //   << " [" << rank << "] ph1 send(@" << displ << ", " << count << ", ..., " << myAgg << "\n";
        //   std::cerr << ss.str();
        // }
#endif
        MPI_Request sreq;
        MPI_Isend(&reinterpret_cast<const char *>(req.sendbuf)[displ], req.sendcounts[ri],
        req.sendtype, myAgg, AGG_TAG, req.comm, &sreq);
        reqs.push_back(sreq);
      }
    }
  
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    reqs.resize(0);


    // if I am a root, recieve data from each aggregator
    // The aggregator will send contiguous data, which we may need to spread out according to rdispls
    Kokkos::View<uint8_t *, typename RecvExecSpace::memory_space>rootBuf("rootBuf");
    if (isRoot) {
      reqs.reserve(naggs); // receive from each aggregator

      const size_t totalRecvd = recvSize * [&]() -> size_t {
        size_t acc = 0;
        for (int i = 0; i < size; ++i) {
          acc += req.recvcounts[i];
        }
        return acc;
      }();
      Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing), rootBuf, totalRecvd);

      // Receive data from each aggregator.
      // Aggregators send data in order of the ranks they're aggregating,
      // which is also the order the root needs in its recv buffer.
      size_t displ = 0;
      for (int aggSrc = 0; aggSrc < size; aggSrc += srcsPerAgg) {
  
        // tally up the total data to recv from the sending aggregator
        int count = 0;
        for (int origSrc = aggSrc;
             origSrc < aggSrc + srcsPerAgg && origSrc < size; ++origSrc) {
          count += req.recvcounts[origSrc];
        }
  
        if (count) {
#ifndef NDEBUG
          // {
          //   std::stringstream ss;
          //   ss << __FILE__ << ":" << __LINE__ 
          //   << " [" << rank << "] ph2 recv(@" << displ << ", " << count << ", ..., " << aggSrc << "\n";
          //   std::cerr << ss.str();
          // }
#endif
          MPI_Request rreq;
          // &rootBuf(displ) causing memory access violations
          MPI_Irecv(rootBuf.data() + displ, count, req.recvtype, aggSrc, req.tag, req.comm, &rreq);
          reqs.push_back(rreq);
          displ += size_t(count) * recvSize;
        }
      }
    }

    // if I am an aggregator, forward data to the roots
    // To each root, send my data in order of the ranks that sent to me
    // which is the order the recvers expect
    if (rank == myAgg) {
      size_t displ = 0;
      for (int ri = 0; ri < req.nroots; ++ri) {
        const size_t count = rootCount[ri];
        if (count) {
#ifndef NDEBUG
          // {
          //   std::stringstream ss;
          //   ss << __FILE__ << ":" << __LINE__ 
          //   << " [" << rank << "] ph2 send(@" << displ << ", " << count << ", ..., " << req.roots[ri] << "\n";
          //   std::cerr << ss.str();
          // }
#endif

#ifndef NDEBUG
          if (size_t(displ) + size_t(count) * sendSize > aggBuf.extent(0)) {
            std::stringstream ss;
            ss << __FILE__ << ":" << __LINE__ 
            << " [" << rank << "] OOB\n";
            std::cerr << ss.str();
          }
#endif


          // &aggBuf[displ] is causing a memory access violation
          MPI_Send(aggBuf.data() + displ, count, req.sendtype, req.roots[ri], req.tag, req.comm);
          displ += count * sendSize;
        }
      }
    }
  
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // at root, copy data from contiguous buffer into recv buffer
    if (isRoot) {

      // set up src and dst for each block
      Kokkos::View<MemcpyArg*, typename RecvExecSpace::memory_space> args(Kokkos::view_alloc("args", Kokkos::WithoutInitializing), size);
      auto args_h = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, args);

      size_t srcOff = 0;
      for (int sRank = 0; sRank < size; ++sRank) {
        const size_t dstOff = req.rdispls[sRank] * recvSize;
        
        void *dst = &reinterpret_cast<char *>(req.recvbuf)[dstOff];
        void *const src = rootBuf.data() + srcOff; // &rootBuf(srcOff)
        const size_t count = req.recvcounts[sRank] * recvSize;
        args_h(sRank) = MemcpyArg{dst, src, count};

#ifndef NDEBUG
        if (srcOff + count > rootBuf.extent(0)) {
          std::stringstream ss;
          ss << __FILE__ << ":" << __LINE__ << " OOB!\n";
          std::cerr << ss.str();
        }
#endif
        srcOff += count;
      }

      // Actually copy the data
      Kokkos::deep_copy(args, args_h);
      using Policy = Kokkos::TeamPolicy<RecvExecSpace>;
      Policy policy(size, Kokkos::AUTO);
      Kokkos::parallel_for("fixup rdispl", policy, 
        KOKKOS_LAMBDA(typename Policy::member_type member){
          team_memcpy(member, args(member.league_rank()));
        }
      );
      Kokkos::fence("after fixup rdispl");

    }

    return finalize();
}


int wait(Req &req) {
  if (req.devAccess) {
    return wait_impl<Kokkos::DefaultExecutionSpace>(req);
  } else {
    return wait_impl<Kokkos::DefaultHostExecutionSpace>(req);
  }
}

int get_status(const Req &req, int *flag, MPI_Status */*status*/) {
  *flag = req.completed;
  return MPI_SUCCESS;
}


} // namespace Tpetra::Details::ialltofewv