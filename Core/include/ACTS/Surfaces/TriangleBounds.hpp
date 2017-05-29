// This file is part of the ACTS project.
//
// Copyright (C) 2016 ACTS project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

///////////////////////////////////////////////////////////////////
// TriangleBounds.h, ACTS project
///////////////////////////////////////////////////////////////////

#ifndef ACTS_SURFACESTRIANGLEBOUNDS_H
#define ACTS_SURFACESTRIANGLEBOUNDS_H

#include <utility>

#include "ACTS/Surfaces/PlanarBounds.hpp"
#include "ACTS/Surfaces/RectangleBounds.hpp"
#include "ACTS/Utilities/Definitions.hpp"
#include "ACTS/Utilities/ParameterDefinitions.hpp"

namespace Acts {

/// @class TriangleBounds
///
/// Bounds for a triangular, planar surface.
///
/// @image html TriangularBounds.gif

class TriangleBounds : public PlanarBounds
{
public:
  // @enum BoundValues for readability
  enum BoundValues {
    bv_x1     = 0,
    bv_y1     = 1,
    bv_x2     = 2,
    bv_y2     = 3,
    bv_x3     = 4,
    bv_y3     = 5,
    bv_length = 6
  };

  TriangleBounds() = delete;

  /// Constructor with coordinates of vertices
  ///
  /// @param vertices is the vector of vertices
  TriangleBounds(const std::vector<Vector2D>& vertices);

  virtual ~TriangleBounds();

  virtual TriangleBounds*
  clone() const final override;

  virtual BoundsType
  type() const final override;

  virtual std::vector<TDD_real_t>
  valueStore() const final override;

  /// This method checks if the provided local coordinates are inside the
  /// surface bounds
  ///
  /// @param lpos local position in 2D local carthesian frame
  /// @param bcheck is the boundary check directive
  /// @return boolean indicator for the success of this operation
  virtual bool
  inside(const Vector2D&      lpos,
         const BoundaryCheck& bcheck) const final override;

  /// Minimal distance to boundary ( > 0 if outside and <=0 if inside)
  ///
  /// @param lpos is the local position to check for the distance
  /// @return is a signed distance parameter
  virtual double
  distanceToBoundary(const Vector2D& lpos) const final override;

  /// This method returns the coordinates of vertices
  std::vector<Vector2D>
  vertices() const final override;

  // Bounding box representation
  virtual const RectangleBounds&
  boundingBox() const final override;

  /// Output Method for std::ostream
  ///
  /// @param sl is the ostream to be dumped into
  virtual std::ostream&
  dump(std::ostream& sl) const final override;

private:
  Vector2D        m_vertices[3];
  RectangleBounds m_boundingBox;  ///< internal bounding box cache
};

}  // end of namespace

#endif  // ACTS_SURFACESRECTANGLEBOUNDS_H
