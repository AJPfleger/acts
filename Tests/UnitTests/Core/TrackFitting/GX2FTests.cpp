// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <boost/test/unit_test.hpp>

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/CuboidVolumeBuilder.hpp"
#include "Acts/Geometry/Layer.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingGeometryBuilder.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Material/HomogeneousSurfaceMaterial.hpp"
#include "Acts/Material/HomogeneousVolumeMaterial.hpp"
#include "Acts/Material/Material.hpp"
#include "Acts/Material/MaterialSlab.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/StraightLineStepper.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"
#include "Acts/Tests/CommonHelpers/DetectorElementStub.hpp"
#include "Acts/Tests/CommonHelpers/FloatComparisons.hpp"
#include "Acts/Tests/CommonHelpers/PredefinedMaterials.hpp"

#include "Acts/Visualization/ObjVisualization3D.hpp"
#include "Acts/Visualization/GeometryView3D.hpp"

#include "Acts/Tests/CommonHelpers/CylindricalTrackingGeometry.hpp"

#include <vector>

using namespace Acts::UnitLiterals;

namespace Acts {
namespace Test {

struct StepVolumeCollector {
  ///
  /// @brief Data container for result analysis
  ///
  struct this_result {
    // Position of the propagator after each step
    std::vector<Vector3> position;
    // Volume of the propagator after each step
    std::vector<TrackingVolume const*> volume;
  };

  using result_type = this_result;

  template <typename propagator_state_t, typename stepper_t,
            typename navigator_t>
  void operator()(propagator_state_t& state, const stepper_t& stepper,
                  const navigator_t& navigator, result_type& result) const {
    result.position.push_back(stepper.position(state.stepping));
    result.volume.push_back(navigator.currentVolume(state.navigation));
  }
};

BOOST_AUTO_TEST_CASE(GX2FTest) {

  ///---------------------------------------------------------------------------
  std::cout << "\n*** Start the GX2F unit test ***\n" << std::endl;

  ObjVisualization3D obj;

  bool triangulate = true;
  ViewConfig viewSensitive = ViewConfig({0, 180, 240});
  viewSensitive.triangulate = triangulate;
  ViewConfig viewPassive = ViewConfig({240, 280, 0});
  viewPassive.triangulate = triangulate;
  ViewConfig viewVolume = ViewConfig({220, 220, 0});
  viewVolume.triangulate = triangulate;
  ViewConfig viewContainer = ViewConfig({220, 220, 0});
  viewContainer.triangulate = triangulate;
  ViewConfig viewGrid = ViewConfig({220, 0, 0});
  viewGrid.nSegments = 8;
  viewGrid.offset = 3.;
  viewGrid.triangulate = triangulate;

  std::string tag = "gx2f_toydet";
  ///---------------------------------------------------------------------------

  // Construct builder
  CuboidVolumeBuilder cvb;

  // Create a test context
  GeometryContext tgContext = GeometryContext();

  // Create configurations for surfaces
  std::vector<CuboidVolumeBuilder::SurfaceConfig> surfaceConfig;
  for (unsigned int i = 0; i < 5; i++) {
    // Position of the surfaces
    CuboidVolumeBuilder::SurfaceConfig cfg;
    cfg.position = {i * UnitConstants::m, 0, 0.};

    // Rotation of the surfaces
    double rotationAngle = M_PI * 0.5;
    Vector3 xPos(cos(rotationAngle), 0., sin(rotationAngle));
    Vector3 yPos(0., 1., 0.);
    Vector3 zPos(-sin(rotationAngle), 0., cos(rotationAngle));
    cfg.rotation.col(0) = xPos;
    cfg.rotation.col(1) = yPos;
    cfg.rotation.col(2) = zPos;
///Shape of the surface
    // Boundaries of the surfaces
    cfg.rBounds =
        std::make_shared<const RectangleBounds>(RectangleBounds(0.5_m, 0.5_m));

    // Material of the surfaces
    MaterialSlab matProp(makeBeryllium(), 0.5_mm);
    cfg.surMat = std::make_shared<HomogeneousSurfaceMaterial>(matProp);

    // Thickness of the detector element
    cfg.thickness = 1_um;

    cfg.detElementConstructor =
        [](const Transform3& trans,
           const std::shared_ptr<const RectangleBounds>& bounds,
           double thickness) {
          return new DetectorElementStub(trans, bounds, thickness);
        };
    surfaceConfig.push_back(cfg);
  }

  // Test that there are actually 4 surface configurations
//  BOOST_CHECK_EQUAL(surfaceConfig.size(), 4u);

  // Test that 4 surfaces can be built
//  for (const auto& cfg : surfaceConfig) {
//    std::shared_ptr<const Surface> pSur = cvb.buildSurface(tgContext, cfg);
//    BOOST_CHECK_NE(pSur, nullptr);
//    CHECK_CLOSE_ABS(pSur->center(tgContext), cfg.position, 1e-9);
//    BOOST_CHECK_NE(pSur->surfaceMaterial(), nullptr);
//    BOOST_CHECK_NE(pSur->associatedDetectorElement(), nullptr);
//  }

  ////////////////////////////////////////////////////////////////////
  // Build layer configurations
  std::vector<CuboidVolumeBuilder::LayerConfig> layerConfig;
  for (auto& sCfg : surfaceConfig) {
    CuboidVolumeBuilder::LayerConfig cfg;
    cfg.surfaceCfg = {sCfg};
    layerConfig.push_back(cfg);
  }

  ///What's happening here?
  for (auto& cfg : layerConfig) {
    cfg.surfaces = {};
  }
///Inner Volume
  // Build volume configuration
  CuboidVolumeBuilder::VolumeConfig volumeConfig;
  volumeConfig.position = {2.5_m, 0., 0.};
  volumeConfig.length = {5_m, 1_m, 1_m};
  volumeConfig.layerCfg = layerConfig;
  volumeConfig.name = "Test volume";
  volumeConfig.volumeMaterial =
      std::make_shared<HomogeneousVolumeMaterial>(makeBeryllium());

  ///volume soll kein material haben!
  std::shared_ptr<TrackingVolume> trVol;
  ///kein volumen im volumen

  volumeConfig.layers.clear();
  for (auto& lay : volumeConfig.layerCfg) {
    lay.active = true;
  }
  trVol = cvb.buildVolume(tgContext, volumeConfig);

  ////////////////////////////////////////////////////////////////////
  // Build TrackingGeometry configuration

///Outer volume
  CuboidVolumeBuilder::Config config;
  config.position = {2.5_m, 0., 0.};
  config.length = {5_m, 1_m, 1_m};
  config.volumeCfg = {volumeConfig};

  cvb.setConfig(config);
  TrackingGeometryBuilder::Config tgbCfg;
  tgbCfg.trackingVolumeBuilders.push_back(
      [=](const auto& context, const auto& inner, const auto&) {
        return cvb.trackingVolume(context, inner, nullptr);
      });
  TrackingGeometryBuilder tgb(tgbCfg);

  std::unique_ptr<const TrackingGeometry> detector =
      tgb.trackingGeometry(tgContext);

  ///---------------------------------------------------------------------------
  const Acts::TrackingVolume& tgVolume = *(detector->highestTrackingVolume());

  GeometryView3D::drawTrackingVolume(obj, tgVolume, tgContext, viewContainer,
                                     viewVolume, viewPassive, viewSensitive,
                                     viewGrid, true, tag);
  ///---------------------------------------------------------------------------
}

}  // namespace Test
}  // namespace Acts
