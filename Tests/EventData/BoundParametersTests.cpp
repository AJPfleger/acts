// This file is part of the ACTS project.
//
// Copyright (C) 2016 ACTS project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define BOOST_TEST_MODULE BoundParameters Tests

#include <boost/test/included/unit_test.hpp>
// leave blank as
#include <boost/test/data/test_case.hpp>

#include "ACTS/EventData/NeutralParameters.hpp"
#include "ACTS/EventData/TrackParameters.hpp"
#include "ACTS/Surfaces/CylinderBounds.hpp"
#include "ACTS/Surfaces/CylinderSurface.hpp"
#include "ACTS/Surfaces/DiscSurface.hpp"
#include "ACTS/Surfaces/PerigeeSurface.hpp"
#include "ACTS/Surfaces/PlaneSurface.hpp"
#include "ACTS/Surfaces/RadialBounds.hpp"
#include "ACTS/Surfaces/RectangleBounds.hpp"
#include "ACTS/Surfaces/StrawSurface.hpp"
#include "ACTS/Utilities/Definitions.hpp"
#include "ACTS/Utilities/Units.hpp"
#include "ParametersTestHelper.hpp"

namespace bdata = boost::unit_test::data;
namespace tt    = boost::test_tools;

namespace Acts {

namespace Test {

  /// @brief Unit test for parameters at a plane
  ///
  BOOST_DATA_TEST_CASE(bound_to_plane_test,
                       bdata::random(-1000., 1000.)
                           ^ bdata::random(-1000., 1000.)
                           ^ bdata::random(-1000., 1000.)
                           ^ bdata::random(0., M_PI)
                           ^ bdata::random(0., M_PI)
                           ^ bdata::random(0., M_PI)
                           ^ bdata::xrange(100),
                       x,
                       y,
                       z,
                       a,
                       b,
                       c,
                       index)
  {
    Vector3D center{x, y, z};
    auto     transform = std::make_shared<Transform3D>();
    transform->setIdentity();
    RotationMatrix3D rot;
    rot = AngleAxis3D(a, Vector3D::UnitX()) * AngleAxis3D(b, Vector3D::UnitY())
        * AngleAxis3D(c, Vector3D::UnitZ());
    transform->prerotate(rot);
    transform->pretranslate(center);
    // create the surfacex
    auto         bounds = std::make_shared<RectangleBounds>(100., 100.);
    PlaneSurface pSurface(transform, bounds);

    // now create parameters on this surface
    // l_x, l_y, phi, theta, q/p (1/p)
    std::array<double, 5> pars_array = {{-0.1234, 9.8765, 0.45, 0.888, 0.001}};
    TrackParametersBase::ParVector_t pars;
    pars << pars_array[0], pars_array[1], pars_array[2], pars_array[3],
        pars_array[4];

    const double phi   = pars_array[2];
    const double theta = pars_array[3];
    double       p     = fabs(1. / pars_array[4]);
    Vector3D     direction(
        cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
    Vector3D mom = p * direction;
    // the global position
    Vector3D pos
        = center + pars_array[0] * rot.col(0) + pars_array[1] * rot.col(1);
    // constructor from parameter vector
    BoundParameters ataPlane_from_pars(nullptr, pars, pSurface);
    consistencyCheck(ataPlane_from_pars, pos, mom, 1., pars_array);
    // constructor from global parameters
    BoundParameters ataPlane_from_global(nullptr, pos, mom, 1., pSurface);
    consistencyCheck(ataPlane_from_global, pos, mom, 1., pars_array);
    // constructor for neutral parameters
    NeutralBoundParameters n_ataPlane_from_pars(nullptr, pars, pSurface);
    consistencyCheck(n_ataPlane_from_pars, pos, mom, 0., pars_array);
    // constructor for neutral global parameters
    NeutralBoundParameters n_ataPlane_from_global(nullptr, pars, pSurface);
    consistencyCheck(n_ataPlane_from_global, pos, mom, 0., pars_array);

    // check that indeed the surfaces are copied
    BOOST_CHECK(&(ataPlane_from_pars.referenceSurface())
                != &(ataPlane_from_global.referenceSurface()));

    // check that the reference frame is the rotation matrix
    BOOST_CHECK(ataPlane_from_pars.referenceFrame().isApprox(rot));

    /// modification test via setter functions
    double ux = 0.3;
    double uy = 0.4;

    ataPlane_from_pars.set<Acts::eLOC_X>(ux);
    ataPlane_from_pars.set<Acts::eLOC_Y>(uy);
    // we should have a new updated position
    Vector3D lPosition3D(ux, uy, 0.);
    Vector3D uposition = rot * lPosition3D + center;
    BOOST_CHECK_EQUAL(uposition, ataPlane_from_pars.position());

    double uphi   = 1.2;
    double utheta = 0.2;
    double uqop   = 0.025;

    ataPlane_from_pars.set<Acts::ePHI>(uphi);
    ataPlane_from_pars.set<Acts::eTHETA>(utheta);
    ataPlane_from_pars.set<Acts::eQOP>(uqop);
    // we should have a new updated momentum
    Vector3D umomentum = 40. * Vector3D(cos(uphi) * sin(utheta),
                                        sin(uphi) * sin(utheta),
                                        cos(utheta));

    BOOST_CHECK(umomentum.isApprox(ataPlane_from_pars.momentum()));
  }

  /// @brief Unit test for parameters at a disc
  ///
  BOOST_DATA_TEST_CASE(bound_to_disc_test,
                       bdata::random(-1000., 1000.)
                           ^ bdata::random(-1000., 1000.)
                           ^ bdata::random(-1000., 1000.)
                           ^ bdata::random(0., M_PI)
                           ^ bdata::random(0., M_PI)
                           ^ bdata::random(0., M_PI)
                           ^ bdata::xrange(100),
                       x,
                       y,
                       z,
                       a,
                       b,
                       c,
                       index)
  {
    Vector3D center{x, y, z};
    auto     transform = std::make_shared<Transform3D>();
    transform->setIdentity();
    RotationMatrix3D rot;
    rot = AngleAxis3D(a, Vector3D::UnitX()) * AngleAxis3D(b, Vector3D::UnitY())
        * AngleAxis3D(c, Vector3D::UnitZ());
    transform->prerotate(rot);
    transform->pretranslate(center);

    auto        bounds = std::make_shared<RadialBounds>(100., 1200.);
    DiscSurface dSurface(transform, bounds);

    // now create parameters on this surface
    // r, phi, phi, theta, q/p (1/p)
    std::array<double, 5> pars_array = {{125., 0.345, 0.45, 0.888, 0.001}};
    TrackParametersBase::ParVector_t pars;
    pars << pars_array[0], pars_array[1], pars_array[2], pars_array[3],
        pars_array[4];

    const double phi   = pars_array[2];
    const double theta = pars_array[3];
    double       p     = fabs(1. / pars_array[4]);
    Vector3D     direction(
        cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
    Vector3D mom = p * direction;
    Vector3D pos = (pars_array[0] * cos(pars_array[1])) * rot.col(0)
        + (pars_array[0] * sin(pars_array[1])) * rot.col(1) + center;
    // constructor from parameter vector
    BoundParameters ataDisc_from_pars(nullptr, pars, dSurface);
    consistencyCheck(ataDisc_from_pars, pos, mom, 1., pars_array);
    // constructor from global parameters
    BoundParameters ataDisc_from_global(nullptr, pos, mom, 1., dSurface);
    consistencyCheck(ataDisc_from_global, pos, mom, 1., pars_array);
    // constructor for neutral parameters
    NeutralBoundParameters n_ataDisc_from_pars(nullptr, pars, dSurface);
    consistencyCheck(n_ataDisc_from_pars, pos, mom, 0., pars_array);
    // constructor for neutral global parameters
    NeutralBoundParameters n_ataDisc_from_global(nullptr, pars, dSurface);
    consistencyCheck(n_ataDisc_from_global, pos, mom, 0., pars_array);

    // check that indeed the surfaces are copied
    BOOST_CHECK(&(ataDisc_from_pars.referenceSurface())
                != &(ataDisc_from_global.referenceSurface()));

    // check that the reference frame is the
    // rotation matrix of the surface
    BOOST_CHECK(ataDisc_from_pars.referenceFrame().isApprox(
        dSurface.transform().rotation()));
  }

  /// @brief Unit test for parameters at a cylinder
  ///
  BOOST_DATA_TEST_CASE(bound_to_cylinder_test,
                       bdata::random(-1000., 1000.)
                           ^ bdata::random(-1000., 1000.)
                           ^ bdata::random(-1000., 1000.)
                           ^ bdata::random(0., M_PI)
                           ^ bdata::random(0., M_PI)
                           ^ bdata::random(0., M_PI)
                           ^ bdata::xrange(100),
                       x,
                       y,
                       z,
                       a,
                       b,
                       c,
                       index)
  {

    Vector3D center{x, y, z};
    auto     transform = std::make_shared<Transform3D>();
    transform->setIdentity();
    RotationMatrix3D rot;
    rot = AngleAxis3D(a, Vector3D::UnitX()) * AngleAxis3D(b, Vector3D::UnitY())
        * AngleAxis3D(c, Vector3D::UnitZ());
    transform->prerotate(rot);
    transform->pretranslate(center);

    auto            bounds = std::make_shared<CylinderBounds>(100., 1200.);
    CylinderSurface cSurface(transform, bounds);

    // now create parameters on this surface
    // rPhi, a, phi, theta, q/p (1/p)
    std::array<double, 5> pars_array = {{125., 343., 0.45, 0.888, 0.001}};
    TrackParametersBase::ParVector_t pars;
    pars << pars_array[0], pars_array[1], pars_array[2], pars_array[3],
        pars_array[4];

    const double phi   = pars_array[2];
    const double theta = pars_array[3];
    double       p     = fabs(1. / pars_array[4]);
    Vector3D     direction(
        cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
    Vector3D mom = p * direction;

    // 3D position in local frame
    const double phi_l = pars_array[0] / bounds->r();
    Vector3D     pos   = (bounds->r() * cos(phi_l)) * rot.col(0)
        + (bounds->r() * sin(phi_l)) * rot.col(1) + (pars_array[1]) * rot.col(2)
        + center;

    // constructor from parameter vector
    BoundParameters ataCylinder_from_pars(nullptr, pars, cSurface);
    consistencyCheck(ataCylinder_from_pars, pos, mom, 1., pars_array);
    // constructor from global parameters
    BoundParameters ataCylinder_from_global(nullptr, pos, mom, 1., cSurface);
    consistencyCheck(ataCylinder_from_global, pos, mom, 1., pars_array);
    // constructor for neutral parameters
    NeutralBoundParameters n_ataCylinder_from_pars(nullptr, pars, cSurface);
    consistencyCheck(n_ataCylinder_from_pars, pos, mom, 0., pars_array);
    // constructor for neutral global parameters
    NeutralBoundParameters n_ataCylinder_from_global(nullptr, pars, cSurface);
    consistencyCheck(n_ataCylinder_from_global, pos, mom, 0., pars_array);

    // check that indeed the surfaces are copied
    BOOST_CHECK(&(ataCylinder_from_pars.referenceSurface())
                != &(ataCylinder_from_global.referenceSurface()));

    // the reference frame is
    // transverse plane to the cylinder
    //    Vector3D normal_at_intersect = cSurface.normal(pos);
    //    Vector3D transverse_y = rot.col(2);
    //    Vectr3D transverse_x = transverse_y.cross(normal_at_intersect);
    //    RotationMatrix3D refframe;
    //    refframe.col(0) = transverse_x;
    //    refframe.col(1) = transverse_y;
    //    refframe.col(2) = normal_at_intersect;
    //
    //    BOOST_CHECK(ataCylinder_from_pars.referenceFrame().isApprox(
    //     refframe));
  }

  /// @brief Unit test for parameters at the perigee
  ///
  BOOST_DATA_TEST_CASE(bound_to_perigee_test,
                       bdata::random(-10., 10.) ^ bdata::random(-10., 10.)
                           ^ bdata::random(-10., 10.)
                           ^ bdata::random(0., 0.05 * M_PI)
                           ^ bdata::random(0., 0.05 * M_PI)
                           ^ bdata::xrange(100),
                       x,
                       y,
                       z,
                       a,
                       b,
                       index)
  {
    Vector3D center{x, y, z};
    auto     transform = std::make_shared<Transform3D>();
    transform->setIdentity();
    RotationMatrix3D rot;
    rot = AngleAxis3D(a, Vector3D::UnitX()) * AngleAxis3D(b, Vector3D::UnitY());
    transform->prerotate(rot);
    transform->pretranslate(center);

    // the straw surface
    PerigeeSurface pSurface(transform);

    // now create parameters on this surface
    // d0, z0, phi, theta, q/p (1/p)
    std::array<double, 5> pars_array = {{-0.7321, 22.5, 0.45, 0.888, 0.001}};
    TrackParametersBase::ParVector_t pars;
    pars << pars_array[0], pars_array[1], pars_array[2], pars_array[3],
        pars_array[4];

    BoundParameters ataPerigee_from_pars(nullptr, pars, pSurface);
    auto            pos = ataPerigee_from_pars.position();
    auto            mom = ataPerigee_from_pars.momentum();
    consistencyCheck(ataPerigee_from_pars, pos, mom, 1., pars_array);
    // constructor from global parameters
    BoundParameters ataPerigee_from_global(nullptr, pos, mom, 1., pSurface);
    consistencyCheck(ataPerigee_from_global, pos, mom, 1., pars_array);
    // constructor for neutral parameters
    NeutralBoundParameters n_ataPerigee_from_pars(nullptr, pars, pSurface);
    consistencyCheck(n_ataPerigee_from_pars, pos, mom, 0., pars_array);
    // constructor for neutral global parameters
    NeutralBoundParameters n_ataPerigee_from_global(nullptr, pars, pSurface);
    consistencyCheck(n_ataPerigee_from_global, pos, mom, 0., pars_array);

    // check that indeed the surfaces are copied
    BOOST_CHECK(&(n_ataPerigee_from_pars.referenceSurface())
                != &(n_ataPerigee_from_global.referenceSurface()));
  }

  /// @brief Unit test for parameters at a line
  ///
  BOOST_DATA_TEST_CASE(bound_to_line_test,
                       bdata::random(-1000., 1000.)
                           ^ bdata::random(-1000., 1000.)
                           ^ bdata::random(-1000., 1000.)
                           ^ bdata::random(0., M_PI)
                           ^ bdata::random(0., M_PI)
                           ^ bdata::random(0., M_PI)
                           ^ bdata::xrange(100),
                       x,
                       y,
                       z,
                       a,
                       b,
                       c,
                       index)
  {

    Vector3D center{x, y, z};
    auto     transform = std::make_shared<Transform3D>();
    transform->setIdentity();
    RotationMatrix3D rot;
    rot = AngleAxis3D(a, Vector3D::UnitX()) * AngleAxis3D(b, Vector3D::UnitY())
        * AngleAxis3D(c, Vector3D::UnitZ());
    transform->prerotate(rot);
    transform->pretranslate(center);

    // the straw surface
    StrawSurface sSurface(
        transform, 2. * Acts::units::_mm, 1. * Acts::units::_m);

    // now create parameters on this surface
    // r, z, phi, theta, q/p (1/p)
    std::array<double, 5> pars_array = {{0.2321, 22.5, 0.45, 0.888, 0.001}};
    TrackParametersBase::ParVector_t pars;
    pars << pars_array[0], pars_array[1], pars_array[2], pars_array[3],
        pars_array[4];

    // constructor from parameter vector
    BoundParameters ataLine_from_pars(nullptr, pars, sSurface);
    auto            pos = ataLine_from_pars.position();
    auto            mom = ataLine_from_pars.momentum();
    consistencyCheck(ataLine_from_pars, pos, mom, 1., pars_array);
    // constructor from global parameters
    BoundParameters ataLine_from_global(nullptr, pos, mom, 1., sSurface);
    consistencyCheck(ataLine_from_global, pos, mom, 1., pars_array);
    // constructor for neutral parameters
    NeutralBoundParameters n_ataLine_from_pars(nullptr, pars, sSurface);
    consistencyCheck(n_ataLine_from_pars, pos, mom, 0., pars_array);
    // constructor for neutral global parameters
    NeutralBoundParameters n_ataLine_from_global(nullptr, pars, sSurface);
    consistencyCheck(n_ataLine_from_global, pos, mom, 0., pars_array);

    // check that indeed the surfaces are copied
    BOOST_CHECK(&(n_ataLine_from_pars.referenceSurface())
                != &(n_ataLine_from_global.referenceSurface()));
  }
}
}