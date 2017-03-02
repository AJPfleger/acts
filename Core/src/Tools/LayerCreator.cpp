// This file is part of the ACTS project.
//
// Copyright (C) 2016 ACTS project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// LayerCreator.cpp, ACTS project
///////////////////////////////////////////////////////////////////

#include "ACTS/Tools/LayerCreator.hpp"
#include "ACTS/Layers/CylinderLayer.hpp"
#include "ACTS/Layers/DiscLayer.hpp"
#include "ACTS/Surfaces/CylinderBounds.hpp"
#include "ACTS/Surfaces/PlanarBounds.hpp"
#include "ACTS/Surfaces/RadialBounds.hpp"
#include "ACTS/Utilities/Definitions.hpp"
#include "ACTS/Utilities/Units.hpp"

Acts::LayerCreator::LayerCreator(const Acts::LayerCreator::Config& lcConfig,
                                 std::unique_ptr<Logger>           logger)
  : m_cfg(), m_logger(std::move(logger))
{
  setConfiguration(lcConfig);
}

void
Acts::LayerCreator::setConfiguration(const Acts::LayerCreator::Config& lcConfig)
{
  // @todo check consistency
  // copy the configuration
  m_cfg = lcConfig;
}

void
Acts::LayerCreator::setLogger(std::unique_ptr<Logger> newLogger)
{
  m_logger = std::move(newLogger);
}

Acts::LayerPtr
Acts::LayerCreator::cylinderLayer(const std::vector<const Surface*>& surfaces,
                                  double                             envelopeR,
                                  double                             envelopeZ,
                                  size_t                             binsPhi,
                                  size_t                             binsZ,
                                  std::shared_ptr<Transform3D> transform) const
{
  // fist loop over the surfaces and estimate the dimensions
  double minR   = 10e10;
  double maxR   = -10e10;
  double minZ   = 10e10;
  double maxZ   = -10e10;
  double minPhi = 10;
  double maxPhi = -10;

  // 1st LOOP --- ESTIMATION ------ test code for automated detection
  // make a surface loop and get the extends
  for (auto& surface : surfaces)
    moduleExtend(*surface, minR, maxR, minPhi, maxPhi, minZ, maxZ);

  // remaining layer parameters
  double layerR         = 0.5 * (minR + maxR);
  double layerHalfZ     = maxZ;
  double layerThickness = (maxR - minR) + 2 * envelopeR;

  // harmonize the phi boundaries (1st step)
  // @todo - allow for sectorally filled arrrays
  double phiStep = (maxPhi - minPhi) / (binsPhi - 1);
  minPhi -= 0.5 * phiStep;
  maxPhi += 0.5 * phiStep;

  // adjust the layer radius
  ACTS_VERBOSE("Creating a cylindrical Layer:");
  ACTS_VERBOSE(" - with layer R    = " << layerR);
  ACTS_VERBOSE(" - from R min/max  = " << minR << " / " << maxR);
  ACTS_VERBOSE(" - with z min/max  = " << -layerHalfZ << " / " << layerHalfZ);
  ACTS_VERBOSE(" - with thickness  = " << (maxR - minR));
  ACTS_VERBOSE("   and tolerance   = " << envelopeR);
  ACTS_VERBOSE(" - and phi min/max = " << minPhi << " / " << maxPhi);
  ACTS_VERBOSE(" - # of modules    = " << surfaces.size() << " ordered in ( "
                                       << binsPhi
                                       << " x "
                                       << binsZ
                                       << ")");

  // create the surface array
  std::unique_ptr<SurfaceArray> sArray
      = m_cfg.surfaceArrayCreator->surfaceArrayOnCylinder(surfaces,
                                                          layerR,
                                                          minPhi,
                                                          maxPhi,
                                                          layerHalfZ,
                                                          binsPhi,
                                                          binsZ,
                                                          transform);

  // create the layer and push it back
  std::shared_ptr<const CylinderBounds> cBounds(
      new CylinderBounds(layerR, layerHalfZ + envelopeZ));

  // create the layer
  LayerPtr cLayer = CylinderLayer::create(
      transform, cBounds, std::move(sArray), layerThickness, nullptr, active);

  if (!cLayer) ACTS_ERROR("Creation of cylinder layer did not succeed!");
  associateSurfacesToLayer(*cLayer);
  // now return
  return cLayer;
}

Acts::LayerPtr
Acts::LayerCreator::cylinderLayer(const std::vector<const Surface*>& surfaces,
                                  double                             layerRmin,
                                  double                             layerRmax,
                                  double                             layerHalfZ,
                                  BinningType                        bTypePhi,
                                  BinningType                        bTypeZ,
                                  std::shared_ptr<Transform3D> transform) const
{

  // remaining layer parameters
  double layerR         = 0.5 * (layerRmin + layerRmax);
  double layerThickness = layerRmax - layerRmin;

  // adjust the layer radius
  ACTS_VERBOSE("Creating a cylindrical Layer:");
  ACTS_VERBOSE(" - with layer R    = " << layerR);
  ACTS_VERBOSE(" - from R min/max  = " << layerRmin << " / " << layerRmax);
  ACTS_VERBOSE(" - with z min/max  = " << -layerHalfZ << " / " << layerHalfZ);
  ACTS_VERBOSE(" - with thickness  = " << layerThickness);
  ACTS_VERBOSE(" - # of modules    = " << surfaces.size() << ")");

  // create the surface array
  std::unique_ptr<SurfaceArray> sArray
      = m_cfg.surfaceArrayCreator->surfaceArrayOnCylinder(
          surfaces, bTypePhi, bTypeZ, transform);

  // create the layer and push it back
  std::shared_ptr<const CylinderBounds> cBounds(
      new CylinderBounds(layerR, layerHalfZ));

  // create the layer
  LayerPtr cLayer = CylinderLayer::create(
      transform, cBounds, std::move(sArray), layerThickness, nullptr, active);

  if (!cLayer) ACTS_ERROR("Creation of cylinder layer did not succeed!");
  associateSurfacesToLayer(*cLayer);
  // now return
  return cLayer;
}

Acts::LayerPtr
Acts::LayerCreator::discLayer(const std::vector<const Surface*>& surfaces,
                              double                             envelopeMinR,
                              double                             envelopeMaxR,
                              double                             envelopeZ,
                              size_t                             binsR,
                              size_t                             binsPhi,
                              std::shared_ptr<Transform3D> transform) const
{
  // loop over the surfaces and estimate
  double minR   = 10e10;
  double maxR   = 0.;
  double minZ   = 10e10;
  double maxZ   = -10e10;
  double minPhi = 10;
  double maxPhi = -10;

  // make a surface loop and get the extends
  for (auto& surface : surfaces)
    moduleExtend(*surface, minR, maxR, minPhi, maxPhi, minZ, maxZ);

  // harmonize the phi boundaries @todo - allow for sectorally filled arrrays
  // later
  double phiStep = (maxPhi - minPhi) / (binsPhi - 1);
  minPhi -= 0.5 * phiStep;
  maxPhi += 0.5 * phiStep;
  // layer parametres
  double layerZ         = 0.5 * (minZ + maxZ);
  double layerThickness = (maxZ - minZ) + 2 * envelopeZ;

  // adjust the layer radius
  ACTS_VERBOSE("Creating a disk Layer:");
  ACTS_VERBOSE(" - at Z position   = " << layerZ);
  ACTS_VERBOSE(" - from R min/max  = " << minR << " / " << maxR);
  ACTS_VERBOSE(" - with thickness  = " << (maxZ - minZ));
  ACTS_VERBOSE("   and tolerance   = " << envelopeZ);
  ACTS_VERBOSE(" - and phi min/max = " << minPhi << " / " << maxPhi);
  ACTS_VERBOSE(" - # of modules    = " << surfaces.size() << " ordered in ( "
                                       << binsR
                                       << " x "
                                       << binsPhi
                                       << ")");

  // create the surface array
  std::unique_ptr<SurfaceArray> sArray
      = m_cfg.surfaceArrayCreator->surfaceArrayOnDisc(
          surfaces, minR, maxR, minPhi, maxPhi, binsR, binsPhi, transform);

  // create the share disc bounds
  auto dBounds = std::make_shared<RadialBounds>(minR - envelopeMinR,
                                                maxR + envelopeMaxR);

  // create the layer transforms if not given
  if (!transform) {
    transform = std::make_shared<Transform3D>(Transform3D::Identity());
    transform->translation() = Vector3D(0., 0., layerZ);
  }

  // create the layers
  LayerPtr dLayer = DiscLayer::create(
      transform, dBounds, std::move(sArray), layerThickness, nullptr, active);

  if (!dLayer) ACTS_ERROR("Creation of disc layer did not succeed!");
  associateSurfacesToLayer(*dLayer);
  // return the layer
  return dLayer;
}

Acts::LayerPtr
Acts::LayerCreator::discLayer(const std::vector<const Surface*>& surfaces,
                              double                             layerZmin,
                              double                             layerZmax,
                              double                             layerRmin,
                              double                             layerRmax,
                              BinningType                        bTypeR,
                              BinningType                        bTypePhi,
                              std::shared_ptr<Transform3D> transform) const
{
  // layer parametres
  double layerZ         = 0.5 * (layerZmin + layerZmax);
  double layerThickness = fabs(layerZmax - layerZmin);

  // adjust the layer radius
  ACTS_VERBOSE("Creating a disk Layer:");
  ACTS_VERBOSE(" - at Z position   = " << layerZ);
  ACTS_VERBOSE(" - from R min/max  = " << layerRmin << " / " << layerRmax);
  ACTS_VERBOSE(" - with thickness  = " << layerThickness);
  ACTS_VERBOSE(" - # of modules    = " << surfaces.size() << ")");

  // create the surface array
  std::unique_ptr<SurfaceArray> sArray
      = m_cfg.surfaceArrayCreator->surfaceArrayOnDisc(
          surfaces, bTypeR, bTypePhi, transform);

  // create the shared disc bounds
  auto dBounds = std::make_shared<RadialBounds>(layerRmin, layerRmax);

  // create the layer transforms if not given
  if (!transform) {
    transform = std::make_shared<Transform3D>(Transform3D::Identity());
    transform->translation() = Vector3D(0., 0., layerZ);
  }

  // create the layers
  LayerPtr dLayer = DiscLayer::create(
      transform, dBounds, std::move(sArray), layerThickness, nullptr, active);

  if (!dLayer) ACTS_ERROR("Creation of disc layer did not succeed!");
  associateSurfacesToLayer(*dLayer);
  // return the layer
  return dLayer;
}

Acts::LayerPtr
Acts::LayerCreator::planeLayer(
    const std::vector<const Surface*>& /**surfaces*/,
    double /**envelopeXY*/,
    double /**envelopeZ*/,
    size_t /**binsX*/,
    size_t /**binsY*/,
    std::shared_ptr<Transform3D> /**transform*/) const
{
  //@todo implement
  return nullptr;
}

void
Acts::LayerCreator::moduleExtend(const Surface& sf,
                                 double&        minR,
                                 double&        maxR,
                                 double&        minPhi,
                                 double&        maxPhi,
                                 double&        minZ,
                                 double&        maxZ) const
{
  // get the associated detector element
  const DetectorElementBase* element = sf.associatedDetectorElement();
  if (element) {
    // get the thickness
    double thickness = element->thickness();
    // check the shape
    const PlanarBounds* pBounds
        = dynamic_cast<const PlanarBounds*>(&(sf.bounds()));
    if (pBounds) {
      // phi is always from the center for planar surfaces
      takeSmallerBigger(minPhi, maxPhi, sf.center().phi());
      // get the vertices
      std::vector<Vector2D> vertices  = pBounds->vertices();
      size_t                nVertices = vertices.size();
      // loop over the two sides of the module
      for (int side = 0; side < 2; ++side) {
        // loop over the vertex combinations
        for (size_t iv = 0; iv < nVertices; ++iv) {
          size_t ivp = iv ? iv - 1 : nVertices - 1;
          // thickness
          double locz = side ? 0.5 * thickness : -0.5 * thickness;
          // p1 & p2 vectors
          Vector3D p2(sf.transform() * Vector3D(vertices.at(iv).x(),
                                                vertices.at(iv).y(),
                                                locz));
          Vector3D p1(sf.transform() * Vector3D(vertices.at(ivp).x(),
                                                vertices.at(ivp).y(),
                                                locz));
          // let's get
          takeSmallerBigger(minZ, maxZ, p2.z());
          takeBigger(maxR, p2.perp());
          takeSmaller(minR, radialDistance(p1, p2));
        }
      }
    } else
      ACTS_WARNING("Not implemented yet for Non-planar bounds");
  }
}

double
Acts::LayerCreator::radialDistance(const Vector3D& pos1,
                                   const Vector3D& pos2) const
{
  // following nominclature found in header file and doxygen documentation
  // line one is the straight track
  const Vector3D& ma = pos1;
  const Vector3D  ea = (pos2 - pos1).unit();
  // line two is the line surface
  Vector3D mb(0., 0., 0);
  Vector3D eb(0., 0., 1.);
  // now go ahead and solve for the closest approach
  Vector3D mab(mb - ma);
  double   eaTeb = ea.dot(eb);
  double   denom = 1 - eaTeb * eaTeb;
  if (fabs(denom) > 10e-7) {
    double lambda0 = (mab.dot(ea) - mab.dot(eb) * eaTeb) / denom;
    // evaluate validaty in terms of bounds
    if (lambda0 < 1. && lambda0 > 0.) return (ma + lambda0 * ea).perp();
    return lambda0 < 0. ? pos1.perp() : pos2.perp();
  }
  return 10e101;
}

void

Acts::LayerCreator::associateSurfacesToLayer(const Layer& layer) const
{
  auto surfaces = layer.surfaceArray()->arrayObjects();

  for (auto& surface : surfaces) {
    surface->associateLayer(layer);
  }
}
