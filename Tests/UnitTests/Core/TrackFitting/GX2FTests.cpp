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
#include "Acts/Tests/CommonHelpers/MeasurementsCreator.hpp"
#include "Acts/Tests/CommonHelpers/PredefinedMaterials.hpp"

#include "Acts/Visualization/ObjVisualization3D.hpp"
#include "Acts/Visualization/GeometryView3D.hpp"

#include "Acts/Tests/CommonHelpers/CylindricalTrackingGeometry.hpp"
#include "Acts/Tests/CommonHelpers/CubicTrackingGeometry.hpp"

#include <vector>

using namespace Acts::UnitLiterals;

namespace Acts {
namespace Test {

//struct StepVolumeCollector {
//  ///
//  /// @brief Data container for result analysis
//  ///
//  struct this_result {
//    // Position of the propagator after each step
//    std::vector<Vector3> position;
//    // Volume of the propagator after each step
//    std::vector<TrackingVolume const*> volume;
//  };
//
//  using result_type = this_result;
//
//  template <typename propagator_state_t, typename stepper_t,
//            typename navigator_t>
//  void operator()(propagator_state_t& state, const stepper_t& stepper,
//                  const navigator_t& navigator, result_type& result) const {
//    result.position.push_back(stepper.position(state.stepping));
//    result.volume.push_back(navigator.currentVolume(state.navigation));
//  }
//};

/// AJP for propagator
//using StraightPropagator =
//    Acts::Propagator<Acts::StraightLineStepper, Acts::Navigator>;
//
//// Construct initial track parameters.
Acts::CurvilinearTrackParameters makeParameters() {
  // create covariance matrix from reasonable standard deviations
  Acts::BoundVector stddev;
  stddev[Acts::eBoundLoc0] = 100_um;
  stddev[Acts::eBoundLoc1] = 100_um;
  stddev[Acts::eBoundTime] = 25_ns;
  stddev[Acts::eBoundPhi] = 2_degree;
  stddev[Acts::eBoundTheta] = 2_degree;
  stddev[Acts::eBoundQOverP] = 1 / 100_GeV;
  Acts::BoundSymMatrix cov = stddev.cwiseProduct(stddev).asDiagonal();
  // define a track in the transverse plane along x
  Acts::Vector4 mPos4(-3_m, 0., 0., 42_ns);
  return Acts::CurvilinearTrackParameters(mPos4, 0_degree, 90_degree, 1_GeV,
                                          1_e, cov);
}


///vvvvvvvvvvvvvvvvvvvv WIP vvvvvvvvvvvvvvvvvvvv
// Construct a straight-line propagator.
auto makeStraightPropagator(std::shared_ptr<const Acts::TrackingGeometry> geo) {
  Acts::Navigator::Config cfg{std::move(geo)};
  cfg.resolvePassive = false;
  cfg.resolveMaterial = true;
  cfg.resolveSensitive = true;
  Acts::Navigator navigator(cfg);
  Acts::StraightLineStepper stepper;
  return Acts::Propagator<Acts::StraightLineStepper, Acts::Navigator>(
      stepper, std::move(navigator));
}


static std::vector<Acts::SourceLink> prepareSourceLinks(
    const std::vector<TestSourceLink>& sourceLinks) {
  std::vector<Acts::SourceLink> result;
  std::transform(sourceLinks.begin(), sourceLinks.end(),
                 std::back_inserter(result),
                 [](const auto& sl) { return Acts::SourceLink{sl}; });
  return result;
}
///^^^^^^^^^^^^^^^^^^^^ WIP ^^^^^^^^^^^^^^^^^^^^

std::shared_ptr<const TrackingGeometry> makeToyDetector(const GeometryContext &tgContext) {
  // Construct builder
  CuboidVolumeBuilder cvb;

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
  return detector;
}

struct Detector {
  // geometry
  std::shared_ptr<const TrackingGeometry> geometry;
};

BOOST_AUTO_TEST_SUITE(GX2FTest)

BOOST_AUTO_TEST_CASE(WIP) {
  std::cout << "\n*** Start the GX2F unit test ***\n" << std::endl;
  std::cout << "\n*** Create Detector ***\n" << std::endl;

  // Create a test context
  GeometryContext tgContext = GeometryContext();

  Detector detector;
  detector.geometry = makeToyDetector(tgContext);

  {
    std::cout << "\n*** Create .obj of Detector ***\n" << std::endl;
    // Only need for obj
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

    const Acts::TrackingVolume& tgVolume =
        *(detector.geometry->highestTrackingVolume());

    GeometryView3D::drawTrackingVolume(obj, tgVolume, tgContext, viewContainer,
                                       viewVolume, viewPassive, viewSensitive,
                                       viewGrid, true, tag);
  }

  std::cout << "\n*** Go to propagator ***\n" << std::endl;

  ///vvvvvvvvvvvvvvvvvvvv WIP vvvvvvvvvvvvvvvvvvvv
  /// AJP: not sure where we need fpe
//  FpeMonitor fpe;
  auto start = makeParameters();
//  auto kfOptions = makeDefaultKalmanFitterOptions();
//
//  const auto kfZeroPropagator =
  //    makeConstantFieldPropagator<ConstantFieldStepper>(tester.geometry, 0_T);
//  const auto kfZero = KalmanFitter(kfZeroPropagator);
//  // regular smoothing
//  kfOptions.reversedFiltering = false;
//  bool expected_reversed = false;
//  bool expected_smoothed = true;
//  tester.test_ZeroFieldWithSurfaceForward(kfZero, kfOptions, start, rng,
//                                          expected_reversed, expected_smoothed,
//                                          true);
//
//
//
////  template <typename fitter_t, typename fitter_options_t, typename parameters_t>
//  void test_ZeroFieldWithSurfaceForward(const fitter_t& fitter,
//                                        fitter_options_t options,
//                                        const parameters_t& start, Rng& rng,
//                                        const bool expected_reversed,
//                                        const bool expected_smoothed,
//                                        const bool doDiag) const {

// Context objects
Acts::GeometryContext geoCtx;
Acts::MagneticFieldContext magCtx;
//Acts::CalibrationContext calCtx;
std::default_random_engine rng(42);

MeasurementResolution resPixel = {MeasurementType::eLoc01, {25_um, 50_um}};
MeasurementResolution resStrip0 = {MeasurementType::eLoc0, {100_um}};
MeasurementResolution resStrip1 = {MeasurementType::eLoc1, {150_um}};
MeasurementResolutionMap resolutions = {
    {Acts::GeometryIdentifier().setVolume(0), resPixel}
};

// simulation propagator
Acts::Propagator<Acts::StraightLineStepper, Acts::Navigator> simPropagator =
      makeStraightPropagator(detector.geometry);
auto measurements = createMeasurements(simPropagator, geoCtx, magCtx, start,
                                       resolutions, rng);
auto sourceLinks = prepareSourceLinks(measurements.sourceLinks);
std::cout << "sourceLinks.size() = " << sourceLinks.size() << std::endl;
constexpr static size_t nMeasurements = 5u; /// AJP TODO: make detector size variable
BOOST_REQUIRE_EQUAL(sourceLinks.size(), nMeasurements);

//    // initial fitter options configured for backward filtereing mode
//    // backward filtering requires a reference surface
//    options.referenceSurface = &start.referenceSurface();
//    // this is the default option. set anyways for consistency
//    options.propagatorPlainOptions.direction = Acts::Direction::Forward;
//
//    Acts::TrackContainer tracks{Acts::VectorTrackContainer{},
//                                Acts::VectorMultiTrajectory{}};
//    tracks.addColumn<bool>("reversed");
//    tracks.addColumn<bool>("smoothed");
//
//    auto res = fitter.fit(sourceLinks.begin(), sourceLinks.end(), start,
//                          options, tracks);
//    BOOST_REQUIRE(res.ok());

//    const auto& track = res.value();
//    BOOST_CHECK_NE(track.tipIndex(), Acts::MultiTrajectoryTraits::kInvalid);



  ///^^^^^^^^^^^^^^^^^^^^ WIP ^^^^^^^^^^^^^^^^^^^^

}

BOOST_AUTO_TEST_SUITE_END()
}  // namespace Test
}  // namespace Acts
