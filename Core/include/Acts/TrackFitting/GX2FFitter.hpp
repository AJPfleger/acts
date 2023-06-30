// This file is part of the Acts project.
//
// Copyright (C) 2016-2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// Workaround for building on clang+libstdc++
#include "Acts/Utilities/detail/ReferenceWrapperAnyCompat.hpp"

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/EventData/Measurement.hpp"
#include "Acts/EventData/MeasurementHelpers.hpp"
#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/MultiTrajectoryHelpers.hpp"
#include "Acts/EventData/SourceLink.hpp"
#include "Acts/EventData/TrackHelpers.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/Material/MaterialSlab.hpp"
#include "Acts/Propagator/AbortList.hpp"
#include "Acts/Propagator/ActionList.hpp"
#include "Acts/Propagator/ConstrainedStep.hpp"
#include "Acts/Propagator/DirectNavigator.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/StandardAborters.hpp"
#include "Acts/Propagator/detail/PointwiseMaterialInteraction.hpp"
#include "Acts/TrackFitting/KalmanFitterError.hpp"
#include "Acts/TrackFitting/detail/KalmanUpdateHelpers.hpp"
#include "Acts/TrackFitting/detail/VoidKalmanComponents.hpp"
#include "Acts/Utilities/CalibrationContext.hpp"
#include "Acts/Utilities/Delegate.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/Result.hpp"

#include <functional>
#include <map>
#include <memory>

#include "Acts/Propagator/StraightLineStepper.hpp"

namespace Acts {
namespace Experimental {

/// Extension struct which holds delegates to customize the KF behavior
template <typename traj_t>
struct GX2FFitterExtensions {
  using TrackStateProxy = typename MultiTrajectory<traj_t>::TrackStateProxy;
  using ConstTrackStateProxy =
      typename MultiTrajectory<traj_t>::ConstTrackStateProxy;
  using Parameters = typename TrackStateProxy::Parameters;

  using Calibrator = Delegate<void(const GeometryContext&, TrackStateProxy)>;

  using Updater = Delegate<Result<void>(const GeometryContext&, TrackStateProxy,
                                        Direction, const Logger&)>;

  using OutlierFinder = Delegate<bool(ConstTrackStateProxy)>;

  /// The Calibrator is a dedicated calibration algorithm that allows
  /// to calibrate measurements using track information, this could be
  /// e.g. sagging for wires, module deformations, etc.
  Calibrator calibrator;

  /// The updater incorporates measurement information into the track parameters
  Updater updater;

  /// Determines whether a measurement is supposed to be considered as an
  /// outlier
  OutlierFinder outlierFinder;

  /// Default constructor which connects the default void components
  GX2FFitterExtensions() {
    calibrator.template connect<&voidKalmanCalibrator<traj_t>>();
    updater.template connect<&voidKalmanUpdater<traj_t>>();
    outlierFinder.template connect<&voidOutlierFinder<traj_t>>();
  }
};

/// Combined options for the Kalman fitter.
///
/// @tparam traj_t The trajectory type
template <typename traj_t>
struct Gx2FitterOptions {
  /// PropagatorOptions with context.
  ///
  /// @param gctx The goemetry context for this fit
  /// @param mctx The magnetic context for this fit
  /// @param cctx The calibration context for this fit
  /// @param extensions_ The KF extensions
  /// @param pOptions The plain propagator options
  /// @param rSurface The reference surface for the fit to be expressed at
  /// @param mScattering Whether to include multiple scattering
  /// @param eLoss Whether to include energy loss
  /// @param freeToBoundCorrection_ Correction for non-linearity effect during transform from free to bound
  Gx2FitterOptions(const GeometryContext& gctx,
                    const MagneticFieldContext& mctx,
                    std::reference_wrapper<const CalibrationContext> cctx,
                    GX2FFitterExtensions<traj_t> extensions_,
                    const PropagatorPlainOptions& pOptions,
                    const Surface* rSurface = nullptr, bool mScattering = false,
                    bool eLoss = false,
                    const FreeToBoundCorrection& freeToBoundCorrection_ =
                        FreeToBoundCorrection(false),
                    const size_t nUpdateMax_ = 5)
      : geoContext(gctx),
        magFieldContext(mctx),
        calibrationContext(cctx),
        extensions(extensions_),
        propagatorPlainOptions(pOptions),
        referenceSurface(rSurface),
        multipleScattering(mScattering),
        energyLoss(eLoss),
        freeToBoundCorrection(freeToBoundCorrection_),
        nUpdateMax(nUpdateMax_){}

  /// Contexts are required and the options must not be default-constructible.
  Gx2FitterOptions() = delete;

  /// Context object for the geometry
  std::reference_wrapper<const GeometryContext> geoContext;
  /// Context object for the magnetic field
  std::reference_wrapper<const MagneticFieldContext> magFieldContext;
  /// context object for the calibration
  std::reference_wrapper<const CalibrationContext> calibrationContext;

  GX2FFitterExtensions<traj_t> extensions;

  /// The trivial propagator options
  PropagatorPlainOptions propagatorPlainOptions;

  /// The reference Surface
  const Surface* referenceSurface = nullptr;

  /// Whether to consider multiple scattering
  bool multipleScattering = false;

  /// Whether to consider energy loss
  bool energyLoss = false;

  /// Whether to include non-linear correction during global to local
  /// transformation
  FreeToBoundCorrection freeToBoundCorrection;

  /// Max number of iterations during the fit
  const size_t nUpdateMax = 5;
};

template <typename traj_t>
struct GX2FFitterResult {
  // Fitted states that the actor has handled.
  traj_t* fittedStates{nullptr};

  // This is the index of the 'tip' of the track stored in multitrajectory.
  // This correspond to the last measurement state in the multitrajectory.
  // Since this KF only stores one trajectory, it is unambiguous.
  // SIZE_MAX is the start of a trajectory.
  size_t lastMeasurementIndex = Acts::MultiTrajectoryTraits::kInvalid;

  // This is the index of the 'tip' of the states stored in multitrajectory.
  // This correspond to the last state in the multitrajectory.
  // Since this KF only stores one trajectory, it is unambiguous.
  // SIZE_MAX is the start of a trajectory.
  size_t lastTrackIndex = Acts::MultiTrajectoryTraits::kInvalid;

  // The optional Parameters at the provided surface
  std::optional<BoundTrackParameters> fittedParameters;

  // Counter for states with non-outlier measurements
  size_t measurementStates = 0;

  // Counter for measurements holes
  // A hole correspond to a surface with an associated detector element with no
  // associated measurement. Holes are only taken into account if they are
  // between the first and last measurements.
  size_t measurementHoles = 0;

  // Counter for handled states
  size_t processedStates = 0;

  // Indicator if track fitting has been done
  bool finished = false;

  // Measurement surfaces without hits
  std::vector<const Surface*> missedActiveSurfaces;

  // Measurement surfaces handled in both forward and
  // backward filtering
  std::vector<const Surface*> passedAgainSurfaces;

  Result<void> result{Result<void>::success()};

  // collectors
  std::vector<ActsScalar> collectorMeasurements;
  std::vector<ActsScalar> collectorCovariance;
  //  std::vector<ActsScalar> collectorResiduals;
  std::vector<BoundMatrix> collectorJacobians;

  // first derivative of chi2 wrt starting track parameters
  //  BoundVector collectorDerive1Chi2Sum = BoundVector::Zero();
  //  BoundMatrix collectorDerive2Chi2Sum = BoundMatrix::Zero();

  BoundMatrix jacobianFromStart = BoundMatrix::Identity();

  //  // chi2 fitter results
  //  ActsDynamicVector residuals;
  //  ActsDynamicMatrix covariance;
  //  ActsScalar chisquare = -1;
  //  std::vector<ActsScalar> chisquares;
};

/// Kalman fitter implementation.
///
/// @tparam propagator_t Type of the propagation class
///
/// The Kalman filter contains an Actor and a Sequencer sub-class.
/// The Sequencer has to be part of the Navigator of the Propagator
/// in order to initialize and provide the measurement surfaces.
///
/// The Actor is part of the Propagation call and does the Kalman update
/// and eventually the smoothing.  Updater and Calibrator are
/// given to the Actor for further use:
/// - The Updater is the implemented kalman updater formalism, it
///   runs via a visitor pattern through the measurements.
///
/// Measurements are not required to be ordered for the KalmanFilter,
/// measurement ordering needs to be figured out by the navigation of
/// the propagator.
///
/// The void components are provided mainly for unit testing.
//template <typename propagator_t, typename traj_t>
template <typename propagator_t, typename traj_t>
class GX2FFitter {
/// Instead of template "typename propagator_t"
//  using propagator_t = Acts::Propagator<Acts::StraightLineStepper, Acts::Navigator>;
  /// The navigator type
  using Gx2fNavigator = typename propagator_t::Navigator;

  /// The navigator has DirectNavigator type or not
  static constexpr bool isDirectNavigator =
      std::is_same<Gx2fNavigator, DirectNavigator>::value;

 public:
  GX2FFitter(propagator_t pPropagator,
             std::unique_ptr<const Logger> _logger =
                 getDefaultLogger("KalmanFitter", Logging::INFO))
      : m_propagator(std::move(pPropagator)),
        m_logger{std::move(_logger)},
        m_actorLogger{m_logger->cloneWithSuffix("Actor")} {}

 private:
  /// The propagator for the transport and material update
  propagator_t m_propagator;

  /// The logger instance
  std::unique_ptr<const Logger> m_logger;
  std::unique_ptr<const Logger> m_actorLogger;

  const Logger& logger() const { return *m_logger; }


  /// @brief Propagator Actor plugin for the GX2F
  ///
  /// @tparam parameters_t The type of parameters used for "local" parameters.
  /// @tparam calibrator_t The type of calibrator
  /// @tparam outlier_finder_t Type of the outlier finder class
  ///
  /// The GX2FnActor does not rely on the measurements to be
  /// sorted along the track. /// TODO is this true?
  template <typename parameters_t>
  class Actor {
   public:
    /// Broadcast the result_type
    using result_type = GX2FFitterResult<traj_t>;

    /// The target surface
    const Surface* targetSurface = nullptr;

    /// Allows retrieving measurements for a surface
    const std::map<GeometryIdentifier, SourceLink>* inputMeasurements = nullptr;

    /// Whether to consider multiple scattering.
    bool multipleScattering = false; /// TODO implement later

    /// Whether to consider energy loss.
    bool energyLoss = false; /// TODO implement later

    /// Whether to include non-linear correction during global to local
    /// transformation
    FreeToBoundCorrection freeToBoundCorrection;

    /// Input MultiTrajectory
    std::shared_ptr<MultiTrajectory<traj_t>> outputStates;

    /// The logger instance
    const Logger* actorLogger{nullptr};

    /// Logger helper
    const Logger& logger() const { return *actorLogger; }

    GX2FFitterExtensions<traj_t> extensions;

    /// The Surface beeing
    SurfaceReached targetReached;

    /// @brief GX2F actor operation
    ///
    /// @tparam propagator_state_t is the type of Propagator state
    /// @tparam stepper_t Type of the stepper
    /// @tparam navigator_t Type of the navigator
    ///
    /// @param state is the mutable propagator state object
    /// @param stepper The stepper in use
    /// @param navigator The navigator in use
    /// @param result is the mutable result state object
    template <typename propagator_state_t, typename stepper_t,
              typename navigator_t>
    void operator()(propagator_state_t& state, const stepper_t& stepper,
                    const navigator_t& navigator, result_type& result,
                    const Logger& /*logger*/) const {
      assert(result.fittedStates && "No MultiTrajectory set");

      std::cout << "Actor: enter operator()" << std::endl;
      if (result.finished) {
        return;
      }

      ACTS_VERBOSE("GX2FFitter step at pos: "
                   << stepper.position(state.stepping).transpose()
                   << " dir: " << stepper.direction(state.stepping).transpose()
                   << " momentum: " << stepper.momentum(state.stepping));

      // Add the measurement surface as external surface to navigator.
      // We will try to hit those surface by ignoring boundary checks.
      if constexpr (not isDirectNavigator) {
        if (result.processedStates == 0) {
          for (auto measurementIt = inputMeasurements->begin();
               measurementIt != inputMeasurements->end(); measurementIt++) {
            navigator.insertExternalSurface(state.navigation,
                                            measurementIt->first);
          }
        }
      }

      // Update:
      // - Waiting for a current surface
      auto surface = navigator.currentSurface(state.navigation);
      std::string direction = state.stepping.navDir.toString();
      if (surface != nullptr) {
        // Check if the surface is in the measurement map
        // -> Get the measurement / calibrate
        // -> Create the predicted state
        // -> Check outlier behavior, if non-outlier:
        // -> Perform the kalman update
        // -> Fill strack state information & update stepper information
        ACTS_VERBOSE("Perform " << direction << " filter step");
        auto res = filter(surface, state, stepper, navigator, result);
        if (!res.ok()) {
          ACTS_ERROR("Error in " << direction << " filter: " << res.error());
          result.result = res.error();
        }
      }

      // Finalization:
      // when all track states have been handled or the navigation is breaked,
      // reset navigation&stepping before run reversed filtering or
      // proceed to run smoothing
      if (result.measurementStates == inputMeasurements->size() or
          (result.measurementStates > 0 and
           navigator.navigationBreak(state.navigation))) {
        // Remove the missing surfaces that occur after the last measurement
        result.missedActiveSurfaces.resize(result.measurementHoles);
        // now get track state proxy for the smoothing logic
        auto trackStateProxy =
            result.fittedStates->getTrackState(result.lastMeasurementIndex);
        {
          // --> Search the starting state to run the smoothing
          // --> Call the smoothing
          // --> Set a stop condition when all track states have been
          // handled
          ACTS_VERBOSE("Finalize/run smoothing");
          auto res = finalize(state, stepper, result);
          if (!res.ok()) {
            ACTS_ERROR("Error in finalize: " << res.error());
            result.result = res.error();
          }
        }
      }
    }


    /// @brief Kalman actor operation : update
    ///
    /// @tparam propagator_state_t is the type of Propagator state
    /// @tparam stepper_t Type of the stepper
    /// @tparam navigator_t Type of the navigator
    ///
    /// @param surface The surface where the update happens
    /// @param state The mutable propagator state object
    /// @param stepper The stepper in use
    /// @param navigator The navigator in use
    /// @param result The mutable result state object
    template <typename propagator_state_t, typename stepper_t,
              typename navigator_t>
    Result<void> filter(const Surface* surface, propagator_state_t& state,
                        const stepper_t& stepper, const navigator_t& navigator,
                        result_type& result) const {
      std::cout << "Actor: enter filter()" << std::endl;
      // Try to find the surface in the measurement surfaces
      auto sourcelink_it = inputMeasurements->find(surface->geometryId());
      if (sourcelink_it != inputMeasurements->end()) {
        // Screen output message
        ACTS_VERBOSE("Measurement surface " << surface->geometryId()
                                            << " detected.");
        // Transport the covariance to the surface
        stepper.transportCovarianceToBound(state.stepping, *surface,
                                           freeToBoundCorrection);

        // do the kalman update (no need to perform covTransport here, hence no
        // point in performing globalToLocal correction)
        auto trackStateProxyRes = detail::kalmanHandleMeasurement(
            state, stepper, extensions, *surface, sourcelink_it->second,
            *result.fittedStates, result.lastTrackIndex, false, logger());

        if (!trackStateProxyRes.ok()) {
          return trackStateProxyRes.error();
        }

        const auto& trackStateProxy = *trackStateProxyRes;
        result.lastTrackIndex = trackStateProxy.index();

        // Update the stepper if it is not an outlier
        if (trackStateProxy.typeFlags().test(
                Acts::TrackStateFlag::MeasurementFlag)) {
          // Update the stepping state with filtered parameters
          ACTS_VERBOSE("Filtering step successful, updated parameters are : \n"
                       << trackStateProxy.filtered().transpose());
          // update stepping state using filtered parameters after kalman
          stepper.update(state.stepping,
                         MultiTrajectoryHelpers::freeFiltered(
                             state.options.geoContext, trackStateProxy),
                         trackStateProxy.filtered(),
                         trackStateProxy.filteredCovariance(), *surface);
          // We count the state with measurement
          ++result.measurementStates;
        }

        // We count the processed state
        ++result.processedStates;
        // Update the number of holes count only when encoutering a
        // measurement
        result.measurementHoles = result.missedActiveSurfaces.size();
        // Since we encountered a measurment update the lastMeasurementIndex to
        // the lastTrackIndex.
        result.lastMeasurementIndex = result.lastTrackIndex;

      } else if (surface->associatedDetectorElement() != nullptr ||
                 surface->surfaceMaterial() != nullptr) {
        // We only create track states here if there is already measurement
        // detected or if the surface has material (no holes before the first
        // measurement)
        if (result.measurementStates > 0 ||
            surface->surfaceMaterial() != nullptr) {
          auto trackStateProxyRes = detail::kalmanHandleNoMeasurement(
              state, stepper, *surface, *result.fittedStates,
              result.lastTrackIndex, true, logger(), freeToBoundCorrection);

          if (!trackStateProxyRes.ok()) {
            return trackStateProxyRes.error();
          }

          const auto& trackStateProxy = *trackStateProxyRes;
          result.lastTrackIndex = trackStateProxy.index();

          if (trackStateProxy.typeFlags().test(TrackStateFlag::HoleFlag)) {
            // Count the missed surface
            result.missedActiveSurfaces.push_back(surface);
          }

          ++result.processedStates;
        }
      }
      return Result<void>::success();
    }



    /// @brief Kalman actor operation : finalize
    ///
    /// @tparam propagator_state_t is the type of Propagator state
    /// @tparam stepper_t Type of the stepper
    ///
    /// @param state is the mutable propagator state object
    /// @param stepper The stepper in use
    /// @param result is the mutable result state object
    template <typename propagator_state_t, typename stepper_t>
    Result<void> finalize(propagator_state_t& state, const stepper_t& stepper,
                          result_type& result) const {
      std::cout << "Actor: enter finalize()" << std::endl;
      // Remember you smoothed the track states
      result.smoothed = true;

      // Get the indices of the first states (can be either a measurement or
      // material);
      size_t firstStateIndex = result.lastMeasurementIndex;
      // Count track states to be smoothed
      size_t nStates = 0;
      result.fittedStates->applyBackwards(
          result.lastMeasurementIndex, [&](auto st) {
            bool isMeasurement =
                st.typeFlags().test(TrackStateFlag::MeasurementFlag);
            bool isMaterial = st.typeFlags().test(TrackStateFlag::MaterialFlag);
            if (isMeasurement || isMaterial) {
              firstStateIndex = st.index();
            }
            nStates++;
          });
      // Return error if the track has no measurement states (but this should
      // not happen)
      if (nStates == 0) {
        ACTS_ERROR("Smoothing for a track without measurements.");
        return KalmanFitterError::SmoothFailed;
      }

      // Return in case no target surface
      if (targetSurface == nullptr) {
        return Result<void>::success();
      }

      // Obtain the smoothed parameters at first/last measurement state
      auto firstCreatedState =
          result.fittedStates->getTrackState(firstStateIndex);
      auto lastCreatedMeasurement =
          result.fittedStates->getTrackState(result.lastMeasurementIndex);

      // Lambda to get the intersection of the free params on the target surface
      auto target = [&](const FreeVector& freeVector) -> SurfaceIntersection {
        return targetSurface->intersect(
            state.geoContext, freeVector.segment<3>(eFreePos0),
            state.stepping.navDir * freeVector.segment<3>(eFreeDir0), true);
      };

      // The smoothed free params at the first/last measurement state.
      // (the first state can also be a material state)
      auto firstParams = MultiTrajectoryHelpers::freeSmoothed(
          state.options.geoContext, firstCreatedState);
      auto lastParams = MultiTrajectoryHelpers::freeSmoothed(
          state.options.geoContext, lastCreatedMeasurement);
      // Get the intersections of the smoothed free parameters with the target
      // surface
      const auto firstIntersection = target(firstParams);
      const auto lastIntersection = target(lastParams);

      // Update the stepping parameters - in order to progress to destination.
      // At the same time, reverse navigation direction for further
      // stepping if necessary.
      // @note The stepping parameters is updated to the smoothed parameters at
      // either the first measurement state or the last measurement state. It
      // assumes the target surface is not within the first and the last
      // smoothed measurement state. Also, whether the intersection is on
      // surface is not checked here.
      bool reverseDirection = false;
      bool closerTofirstCreatedState =
          (std::abs(firstIntersection.intersection.pathLength) <=
           std::abs(lastIntersection.intersection.pathLength));
      if (closerTofirstCreatedState) {
        stepper.resetState(state.stepping, firstCreatedState.smoothed(),
                           firstCreatedState.smoothedCovariance(),
                           firstCreatedState.referenceSurface());
        reverseDirection = (firstIntersection.intersection.pathLength < 0);
      } else {
        stepper.resetState(state.stepping, lastCreatedMeasurement.smoothed(),
                           lastCreatedMeasurement.smoothedCovariance(),
                           lastCreatedMeasurement.referenceSurface());
        reverseDirection = (lastIntersection.intersection.pathLength < 0);
      }
      const auto& surface = closerTofirstCreatedState
                                ? firstCreatedState.referenceSurface()
                                : lastCreatedMeasurement.referenceSurface();
      ACTS_VERBOSE(
          "Smoothing successful, updating stepping state to smoothed "
          "parameters at surface "
          << surface.geometryId() << ". Prepared to reach the target surface.");

      // Reverse the navigation direction if necessary
      if (reverseDirection) {
        ACTS_VERBOSE(
            "Reverse navigation direction after smoothing for reaching the "
            "target surface");
        state.stepping.navDir = state.stepping.navDir.invert();
      }
      // Reset the step size
      state.stepping.stepSize = ConstrainedStep(
          state.stepping.navDir * std::abs(state.options.maxStepSize));
      // Set accumulatd path to zero before targeting surface
      state.stepping.pathAccumulated = 0.;

      return Result<void>::success();
    }
  };


  /// Aborter can stay like this probably
  template <typename parameters_t>
  class Aborter {
   public:
    /// Broadcast the result_type
    using action_type = Actor<parameters_t>;

    template <typename propagator_state_t, typename stepper_t,
              typename navigator_t, typename result_t>
    bool operator()(propagator_state_t& /*state*/, const stepper_t& /*stepper*/,
                    const navigator_t& /*navigator*/, const result_t& result,
                    const Logger& /*logger*/) const {
      if (!result.result.ok() or result.finished) {
        return true;
      }
      return false;
    }
  };

 public:
  /// Fit implementation of the forward filter, calls the
  /// the filter and smoother/reversed filter
  ///
  /// @tparam source_link_iterator_t Iterator type used to pass source links
  /// @tparam start_parameters_t Type of the initial parameters
  /// @tparam parameters_t Type of parameters used for local parameters
  /// @tparam track_container_t Type of the track container backend
  /// @tparam holder_t Type defining track container backend ownership
  ///
  /// @param it Begin iterator for the fittable uncalibrated measurements
  /// @param end End iterator for the fittable uncalibrated measurements
  /// @param sParameters The initial track parameters
  /// @param kfOptions KalmanOptions steering the fit
  /// @param trackContainer Input track container storage to append into
  /// @note The input measurements are given in the form of @c SourceLink s.
  /// It's the calibrators job to turn them into calibrated measurements used in
  /// the fit.
  ///
  /// @return the output as an output track
  template <typename source_link_iterator_t, typename start_parameters_t,
            typename parameters_t = BoundTrackParameters,
            typename track_container_t, template <typename> class holder_t,
            bool _isdn = isDirectNavigator>
  auto fit(source_link_iterator_t it, source_link_iterator_t end,
           const start_parameters_t& sParameters,
           const Gx2FitterOptions<traj_t>& kfOptions,
           TrackContainer<track_container_t, traj_t, holder_t>& trackContainer)
      const -> std::enable_if_t<
          !_isdn, Result<typename TrackContainer<track_container_t, traj_t,
                                                 holder_t>::TrackProxy>> {
    /// Preprocess Measurements (Sourcelinks -> map)
    // To be able to find measurements later, we put them into a map
    // We need to copy input SourceLinks anyways, so the map can own them.
    ACTS_VERBOSE("Preparing " << std::distance(it, end)
                              << " input measurements");
    std::map<GeometryIdentifier, SourceLink> inputMeasurements;
    // for (const auto& sl : sourcelinks) {
    for (; it != end; ++it) {
      SourceLink sl = *it;
      auto geoId = sl.geometryId();
      inputMeasurements.emplace(geoId, std::move(sl));
    }

    /// Fully understand Aborter, Actor, Result later
    // Create the ActionList and AbortList
    using GX2FAborter = Aborter<parameters_t>;
    using GX2FActor = Actor<parameters_t>;

    using GX2FResult = typename GX2FActor::result_type;
    using Actors = ActionList<GX2FActor>;
    using Aborters = AbortList<GX2FAborter>;

    // Create relevant options for the propagation options
    PropagatorOptions<Actors, Aborters> kalmanOptions(
        kfOptions.geoContext, kfOptions.magFieldContext);

    // Set the trivial propagator options
    kalmanOptions.setPlainOptions(kfOptions.propagatorPlainOptions);

    // Catch the actor and set the measurements
    auto& kalmanActor = kalmanOptions.actionList.template get<GX2FActor>();
    kalmanActor.inputMeasurements = &inputMeasurements;
    kalmanActor.targetSurface = kfOptions.referenceSurface;
    kalmanActor.multipleScattering = kfOptions.multipleScattering;
    kalmanActor.energyLoss = kfOptions.energyLoss;
    kalmanActor.freeToBoundCorrection = kfOptions.freeToBoundCorrection;
    kalmanActor.extensions = kfOptions.extensions;
    kalmanActor.actorLogger = m_actorLogger.get();



    // Create relevant options for the propagation options
    PropagatorOptions<Actors, Aborters> GX2FOptions(kfOptions.geoContext, kfOptions.magFieldContext);
    // Set the trivial propagator options
//    GX2FOptions.setPlainOptions(kfOptions.propagatorPlainOptions);


    std::cout << "dbg11" << std::endl;
    typename propagator_t::template action_list_t_result_t<
        CurvilinearTrackParameters, Actors>
        inputResult;
    std::cout << "dbg17" << std::endl;
    auto& r = inputResult.template get<GX2FFitterResult<traj_t>>();
    std::cout << "dbg16" << std::endl;
    r.fittedStates = &trackContainer.trackStateContainer();

    /// Actual Fitting
    std::cout << "dbg15" << std::endl;
    /// Iterate the fit and improve result. Abort after n steps or after convergence
    for (size_t nUpdate = 0; nUpdate < kfOptions.nUpdateMax; nUpdate++) {
      std::cout << "nUpdate = " << nUpdate << "/" << kfOptions.nUpdateMax << std::endl;
      /// update params

      /// propagate with params and return jacobians, residuals (and chi2)
      // Propagator + Actor (einfacher Jacobain accumulation) [actor vor dem loop allokieren] makeMeasurements umschreiben

      /// calculate delta params
      /// iterate through jacobians+residuals

      /// check delta params and abort
      // if (sum(delta_params) < 1e-3) {
      //   break;
      // }
    }

    /// Calculate covariance with inverse of a


    std::cout << "dbg12" << std::endl;

//    // Run the fitter
//    auto result = m_propagator.template propagate(sParameters, GX2FOptions,
//                                                  std::move(inputResult));
//    std::cout << "dbg18" << std::endl;
//    if (!result.ok()) {
//      ACTS_ERROR("Propagation failed: " << result.error());
//      return result.error();
//    }
//
//    std::cout << "dbg14" << std::endl;
//
//    auto& propRes = *result;
//
//    /// Get the result of the fit
//    auto gx2fResult = std::move(propRes.template get<GX2FResult>());
//
//    /// It could happen that the fit ends in zero measurement states.
//    /// The result gets meaningless so such case is regarded as fit failure.
//    if (gx2fResult.result.ok() and not gx2fResult.measurementStates) {
//      gx2fResult.result = Result<void>(KalmanFitterError::NoMeasurementFound);
//    }
//
//    if (!gx2fResult.result.ok()) {
//      ACTS_ERROR("KalmanFilter failed: "
//                 << gx2fResult.result.error() << ", "
//                 << gx2fResult.result.error().message());
//      return gx2fResult.result.error();
//    }


    std::cout << "dbg13" << std::endl;
    auto track = trackContainer.getTrack(trackContainer.addTrack());
    std::cout << "dbg1" << std::endl;
//    track.tipIndex() = gx2fResult.lastMeasurementIndex;
//    std::cout << "dbg2" << std::endl;
//    if (gx2fResult.fittedParameters) {
//      std::cout << "dbg3" << std::endl;
//      const auto& params = gx2fResult.fittedParameters.value();
//      std::cout << "dbg4" << std::endl;
//      track.parameters() = params.parameters();
//      std::cout << "dbg5" << std::endl;
//      track.covariance() = params.covariance().value();
//      std::cout << "dbg6" << std::endl;
//      track.setReferenceSurface(params.referenceSurface().getSharedPtr());
//    }
//    std::cout << "dbg7" << std::endl;
//    track.nMeasurements() = gx2fResult.measurementStates;
//    std::cout << "dbg8" << std::endl;
//    track.nHoles() = gx2fResult.measurementHoles;
//    std::cout << "dbg9" << std::endl;
//    calculateTrackQuantities(track);
//    std::cout << "dbg10" << std::endl;
//    std::cout << "track.chi2() = " << track.chi2() << std::endl;
//
//



    // Return the converted Track
    return track;
  }
};
}  // namespace Experimental
}  // namespace Acts
