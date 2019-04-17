// This file is part of the Acts project.
//
// Copyright (C) 2019 Acts project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define BOOST_TEST_MODULE GenericCuboidVolumeBounds Test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

#include "Acts/Surfaces/PlanarBounds.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Tests/CommonHelpers/FloatComparisons.hpp"
#include "Acts/Utilities/Definitions.hpp"
#include "Acts/Utilities/GeometryContext.hpp"
#include "Acts/Utilities/PlyHelper.hpp"
#include "Acts/Volumes/GenericCuboidVolumeBounds.hpp"

namespace Acts {
namespace Test {

  GeometryContext gctx = GeometryContext();

  BOOST_AUTO_TEST_SUITE(Volumes)

  BOOST_AUTO_TEST_CASE(construction_test)
  {

    std::array<Vector3D, 8> vertices;
    vertices = {{{0, 0, 0},
                 {2, 0, 0},
                 {2, 1, 0},
                 {0, 1, 0},
                 {0, 0, 1},
                 {2, 0, 1},
                 {2, 1, 1},
                 {0, 1, 1}}};
    GenericCuboidVolumeBounds cubo(vertices);

    BOOST_CHECK(cubo.inside({0.5, 0.5, 0.5}));
    BOOST_CHECK(cubo.inside({1.5, 0.5, 0.5}));
    BOOST_CHECK(!cubo.inside({2.5, 0.5, 0.5}));
    BOOST_CHECK(!cubo.inside({0.5, 1.5, 0.5}));
    BOOST_CHECK(!cubo.inside({0.5, 0.5, 1.5}));
    BOOST_CHECK(!cubo.inside({-0.5, 0.5, 0.5}));

    BOOST_CHECK(!cubo.inside({2.2, 1, 1}, 0.1));
    BOOST_CHECK(cubo.inside({2.2, 1, 1}, 0.21));
    BOOST_CHECK(cubo.inside({2.2, 1, 1}, 0.3));
  }

  BOOST_AUTO_TEST_CASE(decomposeToSurfaces)
  {

    std::array<Vector3D, 8> vertices;
    vertices = {{{0, 0, 0},
                 {2, 0, 0},
                 {2, 1, 0},
                 {0, 1, 0},
                 {0, 0, 1},
                 {2, 0, 1},
                 {2, 1, 1},
                 {0, 1, 1}}};
    GenericCuboidVolumeBounds cubo(vertices);

    auto is_in = [](const auto& tvtx, const auto& vertices_) {
      for (const auto& vtx : vertices_) {
        if (checkCloseAbs(vtx, tvtx, 1e-9)) {
          return true;
        }
      }
      return false;
    };

    auto surfaces = cubo.decomposeToSurfaces(nullptr);
    for (const auto& srf : surfaces) {
      auto pbounds = dynamic_cast<const PlanarBounds*>(&srf->bounds());
      for (const auto& vtx : pbounds->vertices()) {
        Vector3D glob;
        srf->localToGlobal(gctx, vtx, {}, glob);
        // check if glob is in actual vertex list
        BOOST_CHECK(is_in(glob, vertices));
      }
    }

    vertices = {{{0, 0, 0},
                 {2, 0, 0.4},
                 {2, 1, 0.4},
                 {0, 1, 0},
                 {0, 0, 1},
                 {1.8, 0, 1},
                 {1.8, 1, 1},
                 {0, 1, 1}}};
    cubo = GenericCuboidVolumeBounds(vertices);

    surfaces = cubo.decomposeToSurfaces(nullptr);
    for (const auto& srf : surfaces) {
      auto pbounds = dynamic_cast<const PlanarBounds*>(&srf->bounds());
      for (const auto& vtx : pbounds->vertices()) {
        Vector3D glob;
        srf->localToGlobal(gctx, vtx, {}, glob);
        // check if glob is in actual vertex list
        BOOST_CHECK(is_in(glob, vertices));
      }
    }

    Transform3D trf;
    trf = Translation3D(Vector3D(0, 8, -5))
        * AngleAxis3D(M_PI / 3., Vector3D(1, -3, 9).normalized());

    surfaces = cubo.decomposeToSurfaces(&trf);
    for (const auto& srf : surfaces) {
      auto pbounds = dynamic_cast<const PlanarBounds*>(&srf->bounds());
      for (const auto& vtx : pbounds->vertices()) {
        Vector3D glob;
        srf->localToGlobal(gctx, vtx, {}, glob);
        // check if glob is in actual vertex list
        BOOST_CHECK(is_in(trf.inverse() * glob, vertices));
      }
    }
  }

  BOOST_AUTO_TEST_CASE(ply_test)
  {
    std::array<Vector3D, 8> vertices;
    vertices = {{{0, 0, 0},
                 {2, 0, 0},
                 {2, 1, 0},
                 {0, 1, 0},
                 {0, 0, 1},
                 {2, 0, 1},
                 {2, 1, 1},
                 {0, 1, 1}}};
    GenericCuboidVolumeBounds cubo(vertices);
    PlyHelper<double>         ply;
    cubo.draw(ply);

    std::ofstream os("cuboid.ply");
    os << ply << std::flush;
    os.close();
  }

  BOOST_AUTO_TEST_SUITE_END()
}
}
