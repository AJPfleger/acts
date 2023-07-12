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
//#include "Acts/Material/Material.hpp"
#include "Acts/Material/MaterialSlab.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/StraightLineStepper.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"
//#include "Acts/Tests/CommonHelpers/CubicTrackingGeometry.hpp"
//#include "Acts/Tests/CommonHelpers/CylindricalTrackingGeometry.hpp"
#include "Acts/Tests/CommonHelpers/DetectorElementStub.hpp"
//#include "Acts/Tests/CommonHelpers/FloatComparisons.hpp"
#include "Acts/Tests/CommonHelpers/MeasurementsCreator.hpp"
#include "Acts/Tests/CommonHelpers/PredefinedMaterials.hpp"
#include "Acts/Visualization/EventDataView3D.hpp"
#include "Acts/Visualization/GeometryView3D.hpp"
#include "Acts/Visualization/ObjVisualization3D.hpp"

#include <vector>

#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/TrackFitting/KalmanFitter.hpp"
#include "Acts/TrackFitting/GainMatrixSmoother.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/Propagator/EigenStepper.hpp"

#include "Acts/TrackFitting/GX2FFitter.hpp"

#include "FitterTestsCommon.hpp"

using namespace Acts::UnitLiterals;

namespace Acts {
namespace Test {

// struct StepVolumeCollector {
//   ///
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
// using StraightPropagator =
//     Acts::Propagator<Acts::StraightLineStepper, Acts::Navigator>;
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

/// vvvvvvvvvvvvvvvvvvvv WIP vvvvvvvvvvvvvvvvvvvv
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

std::shared_ptr<const TrackingGeometry> makeToyDetector(
    const GeometryContext& tgContext, const size_t nSurfaces = 5) {
  if (nSurfaces < 1) {
    throw std::invalid_argument("At least 1 surfaces needs to be created.");
  }
  // Construct builder
  CuboidVolumeBuilder cvb;

  // Create configurations for surfaces
  std::vector<CuboidVolumeBuilder::SurfaceConfig> surfaceConfig;
  for (unsigned int i = 0; i < nSurfaces; i++) {
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
    /// Shape of the surface
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

  /// What's happening here?
  for (auto& cfg : layerConfig) {
    cfg.surfaces = {};
  }
  /// Inner Volume
  // Build volume configuration
  CuboidVolumeBuilder::VolumeConfig volumeConfig;
  volumeConfig.position = {2.5_m, 0., 0.};
  volumeConfig.length = {5_m, 1_m, 1_m};
  volumeConfig.layerCfg = layerConfig;
  volumeConfig.name = "Test volume";
  volumeConfig.volumeMaterial =
      std::make_shared<HomogeneousVolumeMaterial>(makeBeryllium());

  /// volume soll kein material haben!
  std::shared_ptr<TrackingVolume> trVol;
  /// kein volumen im volumen

  volumeConfig.layers.clear();
  for (auto& lay : volumeConfig.layerCfg) {
    lay.active = true;
  }
  trVol = cvb.buildVolume(tgContext, volumeConfig);

  ////////////////////////////////////////////////////////////////////
  // Build TrackingGeometry configuration
  /// Outer volume
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

static void drawMeasurements(IVisualization3D& helper,
                             const Measurements& measurements,
                             std::shared_ptr<const TrackingGeometry> geometry,
                             const Acts::GeometryContext geoCtx,
                             double locErrorScale = 1.,
                             const ViewConfig& viewConfig = s_viewMeasurement) {
  std::cout << "\n*** Draw measurements ***\n" << std::endl;

  for (auto& singleMeasurement : measurements.sourceLinks) {
    auto cov = singleMeasurement.covariance;
    auto lposition = singleMeasurement.parameters;

    auto surf = geometry->findSurface(singleMeasurement.m_geometryId);
    auto transf = surf->transform(geoCtx);

    EventDataView3D::drawMeasurement(helper, lposition, cov, transf,
                                     locErrorScale, viewConfig);
  }
}

// using KalmanUpdater = Acts::GainMatrixUpdater;
const FitterTester tester;

// auto makeDefaultGx2FitterOptions() {
//   Experimental::GX2FFitterExtensions<VectorMultiTrajectory> extensions;
//   extensions.calibrator
//       .connect<&testSourceLinkCalibrator<VectorMultiTrajectory>>();
//   extensions.updater.connect<&KalmanUpdater::operator()<VectorMultiTrajectory>>(
//       &kfUpdater);
//
//   return Experimental::Gx2FitterOptions(tester.geoCtx, tester.magCtx,
//   tester.calCtx,
//                              extensions, PropagatorPlainOptions());
// }

BOOST_AUTO_TEST_SUITE(GX2FTest)

BOOST_AUTO_TEST_CASE(WIP) {
  std::cout << "\n*** Start the GX2F unit test ***\n" << std::endl;
  std::cout << "\n*** Create Detector ***\n" << std::endl;

  // Create a test context
  GeometryContext tgContext = GeometryContext();

  Detector detector;
  const size_t nSurfaces = 5;
  detector.geometry = makeToyDetector(tgContext, nSurfaces);

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

  /// vvvvvvvvvvvvvvvvvvvv WIP vvvvvvvvvvvvvvvvvvvv
  ///  AJP: not sure where we need fpe
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
  //                                          expected_reversed,
  //                                          expected_smoothed, true);
  //
  //
  //
  ////  template <typename fitter_t, typename fitter_options_t, typename
  /// parameters_t>
  //  void test_ZeroFieldWithSurfaceForward(const fitter_t& fitter,
  //                                        fitter_options_t options,
  //                                        const parameters_t& start, Rng& rng,
  //                                        const bool expected_reversed,
  //                                        const bool expected_smoothed,
  //                                        const bool doDiag) const {

  // Context objects
  Acts::GeometryContext geoCtx;
  Acts::MagneticFieldContext magCtx;
  // Acts::CalibrationContext calCtx;
  std::default_random_engine rng(42);

  MeasurementResolution resPixel = {MeasurementType::eLoc01, {25_um, 50_um}};
  // MeasurementResolution resStrip0 = {MeasurementType::eLoc0, {100_um}};
  // MeasurementResolution resStrip1 = {MeasurementType::eLoc1, {150_um}};
  MeasurementResolutionMap resolutions = {
      {Acts::GeometryIdentifier().setVolume(0), resPixel}};

  // simulation propagator
  using SimPropagator =
      Acts::Propagator<Acts::StraightLineStepper, Acts::Navigator>;
  SimPropagator simPropagator = makeStraightPropagator(detector.geometry);
  auto measurements = createMeasurements(simPropagator, geoCtx, magCtx, start,
                                         resolutions, rng);
  auto sourceLinks = prepareSourceLinks(measurements.sourceLinks);
  std::cout << "sourceLinks.size() = " << sourceLinks.size() << std::endl;

  BOOST_REQUIRE_EQUAL(sourceLinks.size(), nSurfaces);

  {
    std::cout << "\n*** Create .obj of measurements ***\n" << std::endl;
    ObjVisualization3D obj;

    double localErrorScale = 10000000.;
    ViewConfig mcolor({255, 145, 48});
    mcolor.offset = 2;
    //  mcolor.visible = true;

    drawMeasurements(obj, measurements, detector.geometry, geoCtx,
                     localErrorScale, mcolor);

    obj.write("meas");
  }

  std::cout << "\n*** Start fitting ***\n" << std::endl;
  std::cout << "\n*** Start fitting -> Kalman ***\n" << std::endl;
  /// KalmanFitter
  {
    const Surface* rSurface = &start.referenceSurface();

    Navigator::Config cfg{detector.geometry};
    cfg.resolvePassive = false;
    cfg.resolveMaterial = true;
    cfg.resolveSensitive = true;
    Navigator rNavigator(cfg);
    // Configure propagation with deactivated B-field
    auto bField = std::make_shared<ConstantBField>(Vector3(0., 0., 0.));
    using RecoStepper = EigenStepper<>;
    RecoStepper rStepper(bField);
    using RecoPropagator = Propagator<RecoStepper, Navigator>;
    RecoPropagator rPropagator(rStepper, rNavigator);

    using KalmanFitter = KalmanFitter<RecoPropagator, VectorMultiTrajectory>;

    KalmanFitter kFitter(rPropagator);

    Acts::GainMatrixSmoother kfSmoother;

    KalmanFitterExtensions<VectorMultiTrajectory> extensions;
    extensions.calibrator
        .connect<&Test::testSourceLinkCalibrator<VectorMultiTrajectory>>();

    extensions.smoother
        .connect<&Acts::GainMatrixSmoother::operator()<VectorMultiTrajectory>>(
            &kfSmoother);

    MagneticFieldContext mfContext = MagneticFieldContext();
    CalibrationContext calContext = CalibrationContext();

    KalmanFitterOptions kfOptions(tgContext, mfContext, calContext, extensions,
                                  PropagatorPlainOptions(), rSurface);

    Acts::TrackContainer tracks{Acts::VectorTrackContainer{},
                                Acts::VectorMultiTrajectory{}};

    // Fit the track
    auto fitRes = kFitter.fit(sourceLinks.begin(), sourceLinks.end(), start,
                              kfOptions, tracks);

    auto& track = *fitRes;

    {
      ObjVisualization3D obj;

      // Draw the track
      std::cout << "Draw the fitted track" << std::endl;
      double momentumScale = 10;
      double localErrorScale = 1000.;
      double directionErrorScale = 100000;

      ViewConfig scolor({214, 214, 214});
      ViewConfig mcolor({255, 145, 48});
      mcolor.offset = -0.01;
      ViewConfig ppcolor({51, 204, 51});
      ppcolor.offset = -0.02;
      ViewConfig fpcolor({255, 255, 0});
      fpcolor.offset = -0.03;
      ViewConfig spcolor({0, 125, 255});
      spcolor.offset = -0.04;

      EventDataView3D::drawMultiTrajectory(
          obj, tracks.trackStateContainer(), track.tipIndex(), tgContext,
          momentumScale, localErrorScale, directionErrorScale, scolor, mcolor,
          ppcolor, fpcolor, spcolor);

      obj.write("Fitted_Track_KF");
    }
  }
  std::cout << "\n*** Start fitting -> GX2F ***\n" << std::endl;
  /// GX2FFitter
  {
    const Surface* rSurface = &start.referenceSurface();

    Navigator::Config cfg{detector.geometry};
    cfg.resolvePassive = false;
    cfg.resolveMaterial = true;
    cfg.resolveSensitive = true;
    Navigator rNavigator(cfg);
    // Configure propagation with deactivated B-field
    auto bField = std::make_shared<ConstantBField>(Vector3(0., 0., 0.));
    using RecoStepper = EigenStepper<>;
    RecoStepper rStepper(bField);
    using RecoPropagator = Propagator<RecoStepper, Navigator>;
    RecoPropagator rPropagator(rStepper, rNavigator);

    using GX2FFitter =
        Experimental::GX2FFitter<RecoPropagator, VectorMultiTrajectory>;

    GX2FFitter xFitter(rPropagator);

    Experimental::GX2FFitterExtensions<VectorMultiTrajectory> extensions;
    extensions.calibrator
        .connect<&Test::testSourceLinkCalibrator<VectorMultiTrajectory>>();

    MagneticFieldContext mfContext = MagneticFieldContext();
    CalibrationContext calContext = CalibrationContext();

    Experimental::Gx2FitterOptions gx2fOptionsTest(
        tgContext, mfContext, calContext, extensions, PropagatorPlainOptions(),
        rSurface, false, false, FreeToBoundCorrection(false), 3);

    Acts::TrackContainer tracksTest{Acts::VectorTrackContainer{},
                                    Acts::VectorMultiTrajectory{}};

    // Fit the track
    auto fitRes = xFitter.fit(sourceLinks.begin(), sourceLinks.end(), start,
                              gx2fOptionsTest, tracksTest);

    auto& trackTest = *fitRes;

    //    {
    //      ObjVisualization3D obj;
    //
    //      // Draw the track
    //      std::cout << "Draw the fitted track" << std::endl;
    //      double momentumScale = 10;
    //      double localErrorScale = 1000.;
    //      double directionErrorScale = 100000;
    //
    //      ViewConfig scolor({214, 214, 214});
    //      ViewConfig mcolor({255, 145, 48});
    //      mcolor.offset = -0.01;
    //      ViewConfig ppcolor({51, 204, 51});
    //      ppcolor.offset = -0.02;
    //      ViewConfig fpcolor({255, 255, 0});
    //      fpcolor.offset = -0.03;
    //      ViewConfig spcolor({0, 125, 255});
    //      spcolor.offset = -0.04;
    //
    //      EventDataView3D::drawMultiTrajectory(
    //          obj, tracks.trackStateContainer(), track.tipIndex(), tgContext,
    //          momentumScale, localErrorScale, directionErrorScale, scolor,
    //          mcolor, ppcolor, fpcolor, spcolor);
    //
    //      std::cout << "tracks.size() = " << tracks.size() << std::endl;
    ////      std::cout << "tracks.container() = " << tracks.container() <<
    //      //      std::endl;
    ////      std::cout << "tracks.covariance() = " << tracks.covariance() <<
    /// std::endl; /      std::cout << "tracks.parameters() = " <<
    /// tracks.parameters() << std::endl; /      std::cout <<
    /// "tracks.component() = " << tracks.component() << std::endl;
    //
    //      obj.write("Fitted_Track_GX2F");
    //    }

    /// add some tests. probably need to rewrite to fit for gx2f
    //    BOOST_AUTO_TEST_CASE(ZeroFieldNoSurfaceForward)
    //    {
    //      using ConstantFieldStepper = Acts::EigenStepper<>;
    ////      using ConstantFieldStepper = Acts::StraightLineStepper<>;
    //
    //
    //      using ConstantFieldPropagator =
    //          Acts::Propagator<ConstantFieldStepper, Acts::Navigator>;
    ////      using KalmanUpdater = Acts::GainMatrixUpdater;
    ////      using KalmanSmoother = Acts::GainMatrixSmoother;
    //      using GX2FFitter2 =
    ////          Acts::GX2FFitter<ConstantFieldPropagator,
    /// VectorMultiTrajectory>;
    //          Acts::Experimental::GX2FFitter<RecoPropagator,
    //          VectorMultiTrajectory>;
    ////      const FitterTester tester;
    //      const auto kfZeroPropagator =
    //          makeConstantFieldPropagator<ConstantFieldStepper>(tester.geometry,
    //          0_T);
    //      const auto kfZero = GX2FFitter2(kfZeroPropagator);
    ////      FpeMonitor fpe;
    ////      auto start = makeParameters();
    //      auto gx2fOptions2 = makeDefaultGX2FFitterOptions();
    //
    //      bool expected_reversed = false;
    //      bool expected_smoothed = false;
    //      tester.test_ZeroFieldNoSurfaceForward(kfZero, gx2fOptions2, start,
    //      rng,
    //                                            expected_reversed,
    //                                            expected_smoothed, false);
    //    }
  }
  ///^^^^^^^^^^^^^^^^^^^^ WIP ^^^^^^^^^^^^^^^^^^^^
}

// This test checks if the call to the fitter works and no errors occur in the
// framework, without fitting and updating any parameters
BOOST_AUTO_TEST_CASE(NoFit) {
  std::cout << "Start test case NoFit" << std::endl;

  // Context objects
  Acts::GeometryContext geoCtx;
  Acts::MagneticFieldContext magCtx;
  Acts::CalibrationContext calCtx;
  std::default_random_engine rng(42);

  Detector detector;
  const size_t nSurfaces = 5;
  detector.geometry = makeToyDetector(geoCtx, nSurfaces);

  auto start = makeParameters();

  MeasurementResolution resPixel = {MeasurementType::eLoc01, {25_um, 50_um}};
  MeasurementResolutionMap resolutions = {
      {Acts::GeometryIdentifier().setVolume(0), resPixel}};

  // propagator
  using SimPropagator =
      Acts::Propagator<Acts::StraightLineStepper, Acts::Navigator>;
  SimPropagator simPropagator = makeStraightPropagator(detector.geometry);
  auto measurements = createMeasurements(simPropagator, geoCtx, magCtx, start,
                                         resolutions, rng);
  auto sourceLinks = prepareSourceLinks(measurements.sourceLinks);

  using GX2FFitter =
      Experimental::GX2FFitter<SimPropagator, VectorMultiTrajectory>;
  GX2FFitter Fitter(simPropagator);

  const Surface* rSurface = &start.referenceSurface();

  Experimental::GX2FFitterExtensions<VectorMultiTrajectory> extensions;
  extensions.calibrator
      .connect<&Test::testSourceLinkCalibrator<VectorMultiTrajectory>>();

  Experimental::Gx2FitterOptions gx2fOptions(
      geoCtx, magCtx, calCtx, extensions, PropagatorPlainOptions(), rSurface,
      false, false, FreeToBoundCorrection(false), 0);

  Acts::TrackContainer tracks{Acts::VectorTrackContainer{},
                              Acts::VectorMultiTrajectory{}};

  // Fit the track
  auto res = Fitter.fit(sourceLinks.begin(), sourceLinks.end(), start,
                        gx2fOptions, tracks);

  BOOST_REQUIRE(res.ok());

  auto& track = *res;
  BOOST_CHECK_EQUAL(track.tipIndex(), Acts::MultiTrajectoryTraits::kInvalid);
  BOOST_CHECK(!track.hasReferenceSurface());
  BOOST_CHECK_EQUAL(track.nMeasurements(), 0u);
  BOOST_CHECK_EQUAL(track.nHoles(), 0u);

  std::cout << "Finished test case NoFit" << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
}  // namespace Test
}  // namespace Acts
