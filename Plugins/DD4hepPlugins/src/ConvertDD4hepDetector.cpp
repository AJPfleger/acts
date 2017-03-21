// This file is part of the ACTS project.
//
// Copyright (C) 2016 ACTS project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ACTS/Plugins/DD4hepPlugins/ConvertDD4hepDetector.hpp"
#include <list>
#include <stdexcept>
#include "ACTS/Plugins/DD4hepPlugins/DD4hepLayerBuilder.hpp"
#include "ACTS/Plugins/DD4hepPlugins/IActsExtension.hpp"
#include "ACTS/Tools/CylinderVolumeBuilder.hpp"
#include "ACTS/Tools/CylinderVolumeHelper.hpp"
#include "ACTS/Tools/LayerArrayCreator.hpp"
#include "ACTS/Tools/LayerCreator.hpp"
#include "ACTS/Tools/SurfaceArrayCreator.hpp"
#include "ACTS/Tools/TrackingGeometryBuilder.hpp"
#include "ACTS/Tools/TrackingVolumeArrayCreator.hpp"
#include "TGeoManager.h"

namespace Acts {
std::unique_ptr<TrackingGeometry>
convertDD4hepDetector(DD4hep::Geometry::DetElement worldDetElement,
                      Logging::Level               loggingLevel,
                      BinningType                  bTypePhi,
                      BinningType                  bTypeR,
                      BinningType                  bTypeZ)
{
  // create local logger for conversion
  auto DD4hepConverterlogger
      = Acts::getDefaultLogger("DD4hepConversion", loggingLevel);
  ACTS_LOCAL_LOGGER(DD4hepConverterlogger);

  ACTS_INFO("Translating DD4hep geometry into ACTS geometry");
  // the return geometry -- and the highest volume
  std::unique_ptr<Acts::TrackingGeometry> trackingGeometry = nullptr;
  // create cylindervolumehelper which can be used by all instances
  // hand over LayerArrayCreator
  auto layerArrayCreator = std::make_shared<Acts::LayerArrayCreator>(
      Acts::getDefaultLogger("LayArrayCreator", loggingLevel));
  // tracking volume array creator
  auto trackingVolumeArrayCreator
      = std::make_shared<Acts::TrackingVolumeArrayCreator>(
          Acts::getDefaultLogger("TrkVolArrayCreator", loggingLevel));
  // configure the cylinder volume helper
  Acts::CylinderVolumeHelper::Config cvhConfig;
  cvhConfig.layerArrayCreator          = layerArrayCreator;
  cvhConfig.trackingVolumeArrayCreator = trackingVolumeArrayCreator;
  auto cylinderVolumeHelper = std::make_shared<Acts::CylinderVolumeHelper>(
      cvhConfig, Acts::getDefaultLogger("CylVolHelper", loggingLevel));

  // get the sub detectors of the world detector e.g. beampipe, pixel detector,
  // strip detector
  std::vector<DD4hep::Geometry::DetElement>     subDetectors;
  const DD4hep::Geometry::DetElement::Children& worldDetectorChildren
      = worldDetElement.children();
  // go through the detector hierarchies
  for (auto& worldDetectorChild : worldDetectorChildren)
    subDetectors.push_back(worldDetectorChild.second);
  // sort by id to build detector from bottom to top
  sort(subDetectors.begin(),
       subDetectors.end(),
       [](const DD4hep::Geometry::DetElement& a,
          const DD4hep::Geometry::DetElement& b) { return (a.id() < b.id()); });

  // the volume builders of the subdetectors
  std::list<std::shared_ptr<ITrackingVolumeBuilder>> volumeBuilders;
  bool                                               beampipe = false;
  // loop over the sub detectors
  for (auto& subDetector : subDetectors) {
    Acts::IActsExtension* subDetExtension = nullptr;
    // at this stage not every DetElement needs to have an Extension attached
    try {
      subDetExtension = subDetector.extension<Acts::IActsExtension>();
    } catch (std::runtime_error& e) {
      if (subDetector.type() == "compound") {
        ACTS_VERBOSE("[D] Subdetector : '"
                     << subDetector.name()
                     << "' has no ActsExtension and has type compound - "
                        "handling as a compound volume (a hierachy of a "
                        "barrel-endcap structure) and resolving the "
                        "subvolumes...");
        // Now create the Layerbuilders and Volumebuilder
        // the layers
        /// the DD4hep::DetElements of the layers of the negative volume
        std::vector<DD4hep::Geometry::DetElement> negativeLayers;
        /// the DD4hep::DetElements of the layers of the central volume
        std::vector<DD4hep::Geometry::DetElement> centralLayers;
        /// the DD4hep::DetElements of the layers of the positive volume
        std::vector<DD4hep::Geometry::DetElement> positiveLayers;

        // go through sub volumes
        const DD4hep::Geometry::DetElement::Children& subDetectorChildren
            = subDetector.children();
        // get rMin, rMax and zBoundaries of the sub Volumes
        double              rMin = 10e12;
        double              rMax = 10e-12;
        std::vector<double> zBoundaries;
        // flags to catch if sub volumes have been set already
        bool nEndCap = false;
        bool pEndCap = false;
        bool barrel  = false;
        for (auto& subDetectorChild : subDetectorChildren) {
          DD4hep::Geometry::DetElement volumeDetElement
              = subDetectorChild.second;

          // get the dimensions of the volume
          TGeoShape* geoShape
              = volumeDetElement.placement().ptr()->GetVolume()->GetShape();
          TGeoConeSeg* tube = dynamic_cast<TGeoConeSeg*>(geoShape);
          if (!tube)
            throw std::logic_error(
                std::string("Volume of DetElement: ") + volumeDetElement.name()
                + std::string(" has wrong shape - needs to be TGeoConeSeg!"));
          // get the dimension of TGeo and convert lengths
          rMin         = std::min(rMin, tube->GetRmin1() * units::_cm);
          rMax         = std::max(rMax, tube->GetRmax1() * units::_cm);
          double halfZ = tube->GetDz() * units::_cm;
          double zPos  = volumeDetElement.placement()
                            .ptr()
                            ->GetMatrix()
                            ->GetTranslation()[2]
              * units::_cm;

          ACTS_VERBOSE(
              "[V] Volume : '"
              << volumeDetElement.name()
              << "'is a compound volume -> resolve now the sub volumes");

          IActsExtension* volumeExtension = nullptr;
          try {
            volumeExtension = volumeDetElement.extension<IActsExtension>();
          } catch (std::runtime_error& e) {
            throw std::logic_error(
                std::string("[V] Current DetElement: ")
                + volumeDetElement.name()
                + std::string(
                      " has no ActsExtension! At this stage it should be a "
                      "detector volume declared as Barrel or Endcap. Please"
                      "check your detector construction."));
          }
          if (volumeExtension->isEndcap()) {
            ACTS_VERBOSE(
                std::string("[V] Subvolume : '") + volumeDetElement.name()
                + std::string("' is a disc volume -> handling as an endcap"));
            // add the boundaries
            zBoundaries.push_back(zPos - halfZ);
            zBoundaries.push_back(zPos + halfZ);

            if (zPos < 0.) {
              if (nEndCap)
                throw std::logic_error(
                    "[V] Negative Endcap was already given for this "
                    "hierachy! Please create a new "
                    "DD4hep_SubDetectorAssembly for the next "
                    "hierarchy.");
              nEndCap = true;
              ACTS_VERBOSE("[V]       ->is negative endcap");
              collectLayers(volumeDetElement, negativeLayers);

            } else {
              if (pEndCap)
                throw std::logic_error(
                    "[V] Positive Endcap was already given for this "
                    "hierachy! Please create a new "
                    "DD4hep_SubDetectorAssembly for the next "
                    "hierarchy.");
              pEndCap = true;
              ACTS_VERBOSE("[V]       ->is positive endcap");
              collectLayers(volumeDetElement, positiveLayers);
            }
          } else if (volumeExtension->isBarrel()) {
            // add the zBoundaries
            zBoundaries.push_back(zPos - halfZ);
            zBoundaries.push_back(zPos + halfZ);
            if (barrel)
              throw std::logic_error("[V] Barrel was already given for this "
                                     "hierachy! Please create a new "
                                     "DD4hep_SubDetectorAssembly for the next "
                                     "hierarchy.");
            barrel = true;
            ACTS_VERBOSE("[V] Subvolume : "
                         << volumeDetElement.name()
                         << " is a cylinder volume -> handling as a barrel");
            collectLayers(volumeDetElement, centralLayers);

          } else {
            throw std::logic_error(
                std::string("[V] Current DetElement: ")
                + volumeDetElement.name()
                + std::string(
                      " has wrong ActsExtension! At this stage it should be a "
                      "detector volume declared as Barrel or Endcap. Please "
                      "check your detector construction."));
          }
        }
        if ((pEndCap && !nEndCap) || (!pEndCap && nEndCap))
          throw std::logic_error("Only one Endcap is given for the current "
                                 "hierarchy! Endcaps should always occur in "
                                 "pairs. Please check you detector "
                                 "construction.");
        // sort the zBoundaries
        std::sort(zBoundaries.begin(), zBoundaries.end());
        std::vector<double> finalZBoundaries;
        finalZBoundaries.push_back(zBoundaries.front());
        finalZBoundaries.push_back(zBoundaries.back());
        // check if endcaps are present
        if (pEndCap && nEndCap) {
          finalZBoundaries.push_back(0.5
                                     * (zBoundaries.at(1) + zBoundaries.at(2)));
          finalZBoundaries.push_back(0.5
                                     * (zBoundaries.at(3) + zBoundaries.at(4)));
        }
        // configure SurfaceArrayCreator
        auto surfaceArrayCreator = std::make_shared<Acts::SurfaceArrayCreator>(
            Acts::getDefaultLogger("SurfaceArrayCreator", loggingLevel));
        // configure LayerCreator
        Acts::LayerCreator::Config lcConfig;
        lcConfig.surfaceArrayCreator = surfaceArrayCreator;
        auto layerCreator            = std::make_shared<Acts::LayerCreator>(
            lcConfig, Acts::getDefaultLogger("LayerCreator", loggingLevel));
        // configure DD4hepLayerBuilder
        Acts::DD4hepLayerBuilder::Config lbConfig;
        lbConfig.configurationName = subDetector.name();
        lbConfig.layerCreator      = layerCreator;
        lbConfig.negativeLayers    = negativeLayers;
        lbConfig.centralLayers     = centralLayers;
        lbConfig.positiveLayers    = positiveLayers;
        lbConfig.bTypePhi          = bTypePhi;
        lbConfig.bTypeR            = bTypeR;
        lbConfig.bTypeZ            = bTypeZ;
        auto dd4hepLayerBuilder    = std::make_shared<Acts::DD4hepLayerBuilder>(
            lbConfig,
            Acts::getDefaultLogger("DD4hepLayerBuilder", loggingLevel));

        // get the possible material of the surounding volume
        DD4hep::Geometry::Material ddmaterial = subDetector.volume().material();
        auto volumeMaterial = std::make_shared<Material>(ddmaterial.radLength(),
                                                         ddmaterial.intLength(),
                                                         ddmaterial.A(),
                                                         ddmaterial.Z(),
                                                         ddmaterial.density());
        // Create the sub volume setup
        std::sort(finalZBoundaries.begin(), finalZBoundaries.end());
        Acts::SubVolumeConfig subVolumeConfig;
        subVolumeConfig.rMin        = rMin;
        subVolumeConfig.rMax        = rMax;
        subVolumeConfig.zBoundaries = finalZBoundaries;
        // the configuration object of the volume builder
        Acts::CylinderVolumeBuilder::Config cvbConfig;
        cvbConfig.trackingVolumeHelper = cylinderVolumeHelper;
        cvbConfig.volumeSignature      = 0;
        cvbConfig.volumeName           = subDetector.name();
        cvbConfig.volumeMaterial       = volumeMaterial;
        cvbConfig.layerBuilder         = dd4hepLayerBuilder;
        cvbConfig.subVolumeConfig      = subVolumeConfig;
        auto cylinderVolumeBuilder
            = std::make_shared<Acts::CylinderVolumeBuilder>(
                cvbConfig,
                Acts::getDefaultLogger("CylinderVolumeBuilder", loggingLevel));
        volumeBuilders.push_back(cylinderVolumeBuilder);

      } else {
        ACTS_INFO(
            "[D] Subdetector with name : '"
            << subDetector.name()
            << "' has no ActsExtension and is not of type 'compound'. If you "
               "want to have this DetElement be translated into the tracking "
               "geometry you need add the right ActsExtension (at this stage "
               "the subvolume needs to be declared as beampipe or barrel) or "
               "if it is a compound DetElement (containing a barrel-endcap "
               "hierarchy), the type needs to be set 'compound'.");
        continue;
      }
    }
    if (subDetExtension && subDetExtension->isBeampipe()) {
      if (beampipe)
        throw std::logic_error("Beampipe has already been set! There can only "
                               "exist one beam pipe. Please check your "
                               "detector construction.");
      beampipe = true;
      ACTS_VERBOSE("[D] Subdetector : "
                   << subDetector.name()
                   << " is the beampipe - building beam pipe.");
      // get the dimensions of the volume
      TGeoShape* geoShape
          = subDetector.placement().ptr()->GetVolume()->GetShape();
      TGeoConeSeg* tube = dynamic_cast<TGeoConeSeg*>(geoShape);
      if (!tube)
        throw std::logic_error(
            "Beampipe has wrong shape - needs to be TGeoConeSeg!");
      // get the dimension of TGeo and convert lengths
      double rMin  = tube->GetRmin1() * units::_cm;
      double rMax  = tube->GetRmax1() * units::_cm;
      double halfZ = tube->GetDz() * units::_cm;
      ACTS_VERBOSE("[V] Extracting cylindrical volume bounds ( rmin / rmax / "
                   "halfZ )=  ( "
                   << rMin
                   << " / "
                   << rMax
                   << " / "
                   << halfZ
                   << " )");

      // get the possible material of the surounding volume
      DD4hep::Geometry::Material ddmaterial = subDetector.volume().material();
      auto volumeMaterial = std::make_shared<Material>(ddmaterial.radLength(),
                                                       ddmaterial.intLength(),
                                                       ddmaterial.A(),
                                                       ddmaterial.Z(),
                                                       ddmaterial.density());
      // the configuration object of the volume builder
      Acts::CylinderVolumeBuilder::Config cvbConfig;
      cvbConfig.trackingVolumeHelper = cylinderVolumeHelper;
      cvbConfig.volumeSignature      = 0;
      cvbConfig.volumeName           = subDetector.name();
      cvbConfig.buildToRadiusZero    = true;
      cvbConfig.volumeMaterial       = volumeMaterial;
      cvbConfig.volumeDimension      = {rMin, rMax, -halfZ, halfZ};
      auto cylinderVolumeBuilder
          = std::make_shared<Acts::CylinderVolumeBuilder>(
              cvbConfig,
              Acts::getDefaultLogger("CylinderVolumeBuilder", loggingLevel));
      volumeBuilders.push_back(cylinderVolumeBuilder);

    } else if (subDetExtension && subDetExtension->isBarrel()) {
      ACTS_VERBOSE("[D] Subdetector: "
                   << subDetector.name()
                   << " is a Barrel volume - building barrel.");
      /// the DD4hep::DetElements of the layers of the central volume
      std::vector<DD4hep::Geometry::DetElement> centralLayers;
      collectLayers(subDetector, centralLayers);

      // configure SurfaceArrayCreator
      auto surfaceArrayCreator = std::make_shared<Acts::SurfaceArrayCreator>(
          Acts::getDefaultLogger("SurfaceArrayCreator", loggingLevel));
      // configure LayerCreator
      Acts::LayerCreator::Config lcConfig;
      lcConfig.surfaceArrayCreator = surfaceArrayCreator;
      auto layerCreator            = std::make_shared<Acts::LayerCreator>(
          lcConfig, Acts::getDefaultLogger("LayerCreator", loggingLevel));
      // configure DD4hepLayerBuilder
      Acts::DD4hepLayerBuilder::Config lbConfig;
      lbConfig.configurationName = subDetector.name();
      lbConfig.layerCreator      = layerCreator;
      lbConfig.centralLayers     = centralLayers;
      lbConfig.bTypePhi          = bTypePhi;
      lbConfig.bTypeZ            = bTypeZ;
      auto dd4hepLayerBuilder    = std::make_shared<Acts::DD4hepLayerBuilder>(
          lbConfig, Acts::getDefaultLogger("DD4hepLayerBuilder", loggingLevel));

      // get the dimensions of the volume
      TGeoShape* geoShape
          = subDetector.placement().ptr()->GetVolume()->GetShape();
      TGeoConeSeg* tube = dynamic_cast<TGeoConeSeg*>(geoShape);
      if (!tube)
        throw std::logic_error(
            std::string("Volume of DetElement: ") + subDetector.name()
            + std::string(" has wrong shape - needs to be TGeoConeSeg!"));
      // get the dimension of TGeo and convert lengths
      std::vector<double> zBoundaries;
      double              halfZ = tube->GetDz() * units::_cm;
      double              zPos
          = subDetector.placement().ptr()->GetMatrix()->GetTranslation()[2]
          * units::_cm;
      zBoundaries.push_back(zPos - halfZ);
      zBoundaries.push_back(zPos + halfZ);
      std::sort(zBoundaries.begin(), zBoundaries.end());
      // create the sub volume setup
      Acts::SubVolumeConfig subVolumeConfig;
      subVolumeConfig.rMin        = tube->GetRmin1() * units::_cm;
      subVolumeConfig.rMax        = tube->GetRmax1() * units::_cm;
      subVolumeConfig.zBoundaries = zBoundaries;
      // get the possible material
      DD4hep::Geometry::Material ddmaterial = subDetector.volume().material();
      auto volumeMaterial = std::make_shared<Material>(ddmaterial.radLength(),
                                                       ddmaterial.intLength(),
                                                       ddmaterial.A(),
                                                       ddmaterial.Z(),
                                                       ddmaterial.density());
      // the configuration object of the volume builder
      Acts::CylinderVolumeBuilder::Config cvbConfig;
      cvbConfig.trackingVolumeHelper = cylinderVolumeHelper;
      cvbConfig.volumeSignature      = 0;
      cvbConfig.volumeName           = subDetector.name();
      cvbConfig.volumeMaterial       = volumeMaterial;
      cvbConfig.layerBuilder         = dd4hepLayerBuilder;
      cvbConfig.subVolumeConfig      = subVolumeConfig;
      auto cylinderVolumeBuilder
          = std::make_shared<Acts::CylinderVolumeBuilder>(
              cvbConfig,
              Acts::getDefaultLogger("CylinderVolumeBuilder", loggingLevel));

    } else {
      if (subDetector.type() != "compound") {
        ACTS_INFO(
            "[D] Subdetector with name : '"
            << subDetector.name()
            << "' has wrong ActsExtension for translation and is not of type "
               "'compound'. If you want to have this DetElement be translated "
               "into the tracking geometry you need add the right "
               "ActsExtension (at this stage the subvolume needs to be "
               "declared as beampipe or barrel) or if it is a compound "
               "DetElement (containing a barrel-endcap hierarchy), the type "
               "needs to be set to 'compound'.");
        continue;
      }
      continue;
    }
  }
  // hand over the collected volume builders
  Acts::TrackingGeometryBuilder::Config tgbConfig;
  tgbConfig.trackingVolumeHelper   = cylinderVolumeHelper;
  tgbConfig.trackingVolumeBuilders = volumeBuilders;
  auto trackingGeometryBuilder
      = std::make_shared<Acts::TrackingGeometryBuilder>(tgbConfig);
  return (trackingGeometryBuilder->trackingGeometry());
}

void
collectLayers(DD4hep::Geometry::DetElement&              detElement,
              std::vector<DD4hep::Geometry::DetElement>& layers)
{
  const DD4hep::Geometry::DetElement::Children& children
      = detElement.children();
  if (!children.empty()) {
    for (auto& child : children) {
      DD4hep::Geometry::DetElement childDetElement = child.second;
      Acts::IActsExtension*        detExtension    = nullptr;
      try {
        detExtension = childDetElement.extension<Acts::IActsExtension>();
      } catch (std::runtime_error& e) {
        continue;
      }
      if (detExtension->isLayer()) layers.push_back(childDetElement);
      collectLayers(childDetElement, layers);
    }
  }
}
}  // End of namespace Acts
