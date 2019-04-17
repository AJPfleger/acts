// This file is part of the Acts project.
//
// Copyright (C) 2019 Acts project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>
#include <string>        // for string printing
#include <system_error>  // bring in std::error_code et al

namespace Acts {
// This is the custom error code enum
enum class VertexingError { NumericFailure = 1 };

namespace detail {
  // Define a custom error code category derived from std::error_category
  class VertexingErrorCategory : public std::error_category
  {
  public:
    // Return a short descriptive name for the category
    const char*
    name() const noexcept final
    {
      return "VertexingError";
    }
    // Return what each enum means in text
    std::string
    message(int c) const final
    {
      switch (static_cast<VertexingError>(c)) {
      case VertexingError::NumericFailure:
        return "Numeric failure in calculation.";
      default:
        return "unknown";
      }
    }
  };
}

// Declare a global function returning a static instance of the custom category
extern inline const detail::VertexingErrorCategory&
VertexingErrorCategory()
{
  static detail::VertexingErrorCategory c;
  return c;
}

inline std::error_code
make_error_code(Acts::VertexingError e)
{
  return {static_cast<int>(e), Acts::VertexingErrorCategory()};
}
}

namespace std {
// register with STL
template <>
struct is_error_code_enum<Acts::VertexingError> : std::true_type
{
};
}
