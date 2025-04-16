// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Tpetra_Details_DistributorActor.hpp"

#include <fstream>

namespace Tpetra::Details {

  DistributorActor::DistributorActor()
    : mpiTag_(DEFAULT_MPI_TAG) {}

      // FIXME: should be able to default this guy
#if 0
  DistributorActor::DistributorActor(const DistributorActor& otherActor)
    : mpiTag_(otherActor.mpiTag_),
    requestsRecv_(otherActor.requestsRecv_),
    requestsSend_(otherActor.requestsSend_),
    requestsIgatherv_(otherActor.requestsIgatherv_) {
      if (!requestsIgatherv_.empty()) {
        std::cerr << __FILE__ << ":" << __LINE__ << " DistributorActor copy ctor with active igathers!\n";
      }
    }
#endif

  void DistributorActor::doWaits(const DistributorPlan& plan) {
    if (requestsRecv_.size() > 0) {
      ProfilingRegion wr("Tpetra::Distributor: doWaitsRecv[via doWaits]");

      Teuchos::waitAll(*plan.getComm(), requestsRecv_());

      // Restore the invariant that requests_.size() is the number of
      // outstanding nonblocking communication requests.
      requestsRecv_.resize(0);
    }

    if (requestsSend_.size() > 0) {
      ProfilingRegion ws("Tpetra::Distributor: doWaitsSend[via doWaits]");

      Teuchos::waitAll(*plan.getComm(), requestsSend_());

      // Restore the invariant that requests_.size() is the number of
      // outstanding nonblocking communication requests.
      requestsSend_.resize(0);
    }

    {
      ProfilingRegion ws("Tpetra::Distributor: doWaitsIgatherv[via doWaits]");
      doWaitsIgatherv(plan);
    }
  }

  void DistributorActor::doWaitsRecv(const DistributorPlan& plan) {
    if (requestsRecv_.size() > 0) {
      ProfilingRegion wr("Tpetra::Distributor::doWaitsRecv");

      Teuchos::waitAll(*plan.getComm(), requestsRecv_());

      // Restore the invariant that requests_.size() is the number of
      // outstanding nonblocking communication requests.
      requestsRecv_.resize(0);
    }

    {
      ProfilingRegion ws("Tpetra::Distributor: doWaitsIgatherv[via doWaitsRecv]");
      doWaitsIgatherv(plan);
    }
  }

  void DistributorActor::doWaitsSend(const DistributorPlan& plan) {
    if (requestsSend_.size() > 0) {
      ProfilingRegion ws("Tpetra::Distributor::doWaitsSend");

      Teuchos::waitAll(*plan.getComm(), requestsSend_());

      // Restore the invariant that requests_.size() is the number of
      // outstanding nonblocking communication requests.
      requestsSend_.resize(0);
    }
  }

  void DistributorActor::doWaitsIgatherv(const DistributorPlan& plan) {
    #ifdef HAVE_TPETRA_MPI
    if (!requestsIgatherv_.empty()) {

      // FIXME: debug
      // {
      //   std::stringstream ss;
      //   ss << __FILE__ << ":" << __LINE__ << " " << plan.getComm()->getRank() << " waitall[Igatherv]\n";
      //   std::cerr << ss.str();
      // }

      ProfilingRegion ws("Tpetra::Distributor: doWaitsIgatherv");

#ifdef TPETRA_USE_INTERNAL_IGATHERV
  for (auto &req : requestsIgatherv_) {
    Details::igatherv::wait(req);
  }
#else
  MPI_Waitall(requestsIgatherv_.size(), requestsIgatherv_.data(), MPI_STATUSES_IGNORE);
#endif
requestsIgatherv_.clear();
recvcountsIgatherv_.clear();
recvdisplsIgatherv_.clear();
    }
  #endif

  }

  bool DistributorActor::isReady() const {
    bool result = true;
    for (auto& request : requestsRecv_) {
      result &= request->isReady();
    }
    for (auto& request : requestsSend_) {
      result &= request->isReady();
    }

    // isReady just calls MPI_Test and returns true if the op
    // succeeded
    // don't use test because these are for a collective, and not
    // all ranks may call test, so progress may not be possible
#ifdef HAVE_TPETRA_MPI
    for (auto req : requestsIgatherv_) {
      int flag;
#ifdef TPETRA_USE_INTERNAL_IGATHERV
      Details::igatherv::get_status(req, &flag, MPI_STATUS_IGNORE);
#else
      MPI_Request_get_status(req, &flag, MPI_STATUS_IGNORE);
#endif
      result &= flag;
    }
#endif

    return result;
  }
}
