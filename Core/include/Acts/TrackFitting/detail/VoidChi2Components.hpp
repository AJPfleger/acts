// This file is part of the Acts project.
//
// Copyright (C) 2021-2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/TypeTraits.hpp"

namespace Acts {

inline void voidChi2Calibrator(
    const GeometryContext& /*gctx*/,
    MultiTrajectory::TrackStateProxy /*trackState*/) {
  throw std::runtime_error{"VoidChi2Calibrator should not ever execute"};
}

inline bool voidChi2OutlierFinder(
    MultiTrajectory::ConstTrackStateProxy /*trackState*/) {
  return false;
}

}  // namespace Acts