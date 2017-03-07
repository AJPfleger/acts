// This file is part of the ACTS project.
//
// Copyright (C) 2016 ACTS project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef ACTS_MODULES_HPP
#define ACTS_MODULES_HPP 1
// clang-format off

/// @defgroup Design Design and concept descriptions
/// @brief description of general concepts used in ACTS

/// @defgroup Logging Debug output options
/// @ingroup Design
/// @brief description of debug output options
///
/// In order to add debug messages to your program, you should use the provided
/// macros for the different severity levels:
/// - #ACTS_VERBOSE
/// - #ACTS_DEBUG
/// - #ACTS_INFO
/// - #ACTS_WARNING
/// - #ACTS_ERROR
/// - #ACTS_FATAL
///
/// All of these macros require that a function <tt>logger()</tt> returning a
/// Acts::Logger object is available in the scope in which the macros are used.
/// For your convenience, the macro #ACTS_LOCAL_LOGGER is provided which does
/// the job for you. The ACTS logging facility supports several severity levels
/// which
/// allow you to control the amount of information displayed at run-time. Logger
/// objects can easily be created using the Acts::getDefaultLogger function
/// which should be sufficient to get you started. In case you need more
/// customized debug output, you can make use of the output decorators defined
/// in Acts::Logging or even write your own implementation of
/// Acts::Logging::OutputDecorator.
///
/// @par Code example illustrating the usage
/// @code{.cpp}
/// #include <fstream>
/// #include <memory>
/// #include "ACTS/Utilities/Logger.hpp"
///
/// void myFunction() {
///    // open the logfile
///    std::ofstream logfile("log.txt");
///
///    // setup a logger instance for >= INFO messages, streaming into the log file
///    // make sure you do NOT call the variable 'logger'
///    std::unique_ptr<Acts::Logger> myLogger = Acts::getDefaultLogger("MyLogger", Acts::Logging::INFO, &logfile);
///
///    // make sure the ACTS debug macros can work with your logger
///    ACTS_LOCAL_LOGGER(myLogger);
///
///    ACTS_VERBOSE("This message will not appear in the logfile.");
///    ACTS_INFO("But this one will: Hello World!");
///
///    // do not forget to close the logfile
///    logfile.close();
/// }
/// @endcode
///
/// In case you are using ACTS in another framework which comes with its own
/// logging facility (e.g. Gaudi) you can pipe the logging output from ACTS
/// tools and algorithms to your framework's logging system by supplying different
/// implementations of:
/// - Acts::Logging::OutputFilterPolicy (for mapping logging levels)
/// - Acts::Logging::OutputPrintPolicy (for passing the ACTS output to your internal logging system)
///
/// Since ACTS makes extensive use of Acts::getDefaultLogger to provide
/// sufficient information for debugging, you would need to provide a modified
/// implementation of this function (using your output filter and printing
/// policies) to also pipe this output to your framework.
///
/// Changing the implementation of an already defined function is a non-trivial
/// task C++. We recommend the following approach using the possibility to inject
/// custom code by pre-loading shared libraries with <tt>LD_PRELOAD</tt>. You
/// need to provide an appropriate implementation for a function of the following
/// signature into a separate and compile it in a shared library
///
/// @code{.cpp}
/// namespace Acts {
///   std::unique_ptr<Logger> getDefaultLogger(const std::string&, const Logging::Level&, std::ostream*);
/// }
/// @endcode
///
/// Then you can run your executable, which uses ACTS tools and algorithms, in
/// the following way (tested under Unix)
///
/// @code{bash}
/// LD_PRELOAD=<YOUR_SHARED_LIBRARY> path/to/your/exectuable
/// @endcode
///
/// For an example have a look at CustomDefaultLogger.cpp which you can use as follows:
///
/// @code{bash}
/// cd <ACTS/INSTALL/DIRECTORY>
/// source bin/setup.sh
/// LD_PRELOAD=lib/libACTSCustomLogger.so bin/Examples/ACTSGenericDetector
/// @endcode

/// @defgroup Core Core classes
/// @brief ACTS core classes

/// @defgroup Detector Tracking geometry
/// @ingroup Core
/// @brief Description of the tracking geometry

/// @defgroup EventData Event data model
/// @ingroup Core
/// @brief Event data model

/// @defgroup Extrapolation Track extrapolation
/// @ingroup Core
/// @brief Algorithms for extrapolation of track parameters

/// @defgroup Fitter Track fitters
/// @ingroup Core
/// @brief Algorithms for track fitting

/// @defgroup Layers Layers
/// @ingroup Core
/// @brief Description of detector layers

/// @defgroup MagneticField Magnetic field
/// @ingroup Core
/// @brief Description of magnetic field properties

/// @defgroup Material Material
/// @ingroup Core
/// @brief Description of material properties

/// @defgroup Surfaces Geometric surfaces
/// @ingroup Core
/// @brief Description of geometric surfaces

/// @defgroup Tools Tools
/// @ingroup Core
/// @brief Geometry building tools

/// @defgroup Utilities Helper classes
/// @ingroup Core
/// @brief Helper utilities

/// @defgroup Volumes Volumes
/// @ingroup Core
/// @brief Description of geometric volumes

/// @defgroup Examples Examples
/// @brief ACTS Examples


/// @defgroup Plugins Plugins
/// @brief ACTS extensions

/// @defgroup DD4hepPlugins DD4hepPlugins
/// @ingroup Plugins
/// @brief Build ACTS tracking geometry from \a %DD4hep input.
///
/// The DD4hepPlugin allows building of the Acts TrackingGeometry from <a href="http://aidasoft.web.cern.ch/DD4hep">DD4hep</a> input.
/// \a %DD4hep uses <a href="https://root.cern.ch">ROOT</a> TGeo as an underlying geometry model.
///
/// <B>General</B>
///
/// The basic input for building the detector are a detector description in the
/// XML file format and a corresponding detector constructor written in C++. These
/// two components have to be changed accordingly. Detector constructors use these
/// XML files as an input and construct a detector in the \a %DD4hep geometry format.
/// The whole detector is segmented into different detector parts, e.g. barrels
/// and endcaps which describe different sub-detectors. Since these sub-detectors
/// are built differently, they need different detector constructors. In this way
/// one detector description in XML can use various detector constructors, needed
/// for the different kinds of detector parts.
///
/// In the detector description model of \a %DD4hep any detector is a tree of instances
/// of the so-called \a DetElement class. This \a DetElement class provides all needed
/// detector information, e.g. readout segmentation, geometrical information,
/// environmental conditions. This tree is parallel to the volume tree, which
/// provides the TGeoVolumes and their placements.
/// The relation between these two trees is one-directional, i.e. every volume can be
/// accessed via its corresponding \a DetElement, but not vice versa.
/// Not every supporting material will be declared as a detector element, hence, the
/// geometrical tree can have a deeper hierarchy structure. In order to access both,
/// detector specific and geometrical information, the conversion to the tracking
/// geometry navigates through the detector tree.
/// The \a DetElement can also be extended, to add specific features or to access
/// information. This extension mechanism is used during the translation process.
///
/// <B>ActsExtensions</B>
///
/// \a %DD4hep provides a special extension mechanism for the \a DetElement which allows to
/// add custom features. In Acts this functionality is used for the conversion from
/// \a %DD4hep into Acts.
/// The extensions are used to indicate certain volumes, e.g. if a \a DetElement is the
/// beam pipe or if a \a DetElement is a layer carrying the sensitive modules. In addition
/// the extensions are used in order to distinguish if a sub detector is a barrel or
/// an endcap (which is described as a disc volume in Acts) which are both described
/// with the underlying TGeo class \c TGeoConeSegment in \a %DD4hep. Furthermore the
/// extensions are used to hand over specific information needed for tracking, e.g.
/// paramters for material mapping.
/// Please find further information in Acts::ActsExtension.
///
/// <B>DD4hepDetElement</B>
///
/// In Acts the surfaces describing the sensitive modules of a detector are directly
/// linked to these of the initial geometry input. In the case of \a %DD4hep the
/// Acts::DD4hepDetElement was introduced which is the direct link of Acts to \a %DD4hep.
/// In the case for tracking relevant paramters in the \a %DD4hep geometry description
/// are changed (e.g. alignment) it will be automatically changed in Acts.
///
/// <B>Build</B>
///
/// The DD4hepPlugin is only build on demand. The DD4hepPlugin depends on the TGeoPlugin
/// therefore both Plugins need to be installed.
/// During the cmake configuration the flags \c BUILD_DD4HEP_PLUGIN and \c BUILD_TGEO_PLUGIN need to be set \a ON.
/// In addition \a ROOT and \a %DD4hep need to be added to the \c CMAKE_PREFIX_PATH. When using the DD4hepPlugin
/// both the TGeoPlugin and the DD4hepPlugin components need to be loaded in the user's CMake configuration.
///
/// <B>Prerequisites</B>
///
/// To guarantee a working translation from \a %DD4hep input to ACTS geometry the
/// following conditions need to be met:
///
/// * The detector needs to have a barrel-endcap structure:
///   Every hierarchy of subdetectors (e.g. PixelDetector,
///   StripDetector,..)
///   needs to be decomposed of either a single barrel or of a barrel and two
///   endcaps
///
/// * If a hierachy is not only a single barrel but is decomposed of a barrel
/// and
///   its corresponding endcaps they need to be grouped together in an assembly
///   using the \c DD4hep_SubdetectorAssembly constructor which is provided by
///   \a %DD4hep.
///   Example of usage in xml file (where \c Barrel0, \c nEndCap0 and \c
///   pEndCap0
///   are sub detectors defined in the file \c PixelTracker.xml):
///   @code
///   <include ref="PixelTracker.xml"/>
///   <detectors>
///     <detector id="1" name="PixelTracker" type="DD4hep_SubdetectorAssembly"
///      vis="BlueVisTrans">
///       <shape name="PixelEnvelope" type="Tube" rmin="Env0_rmin"
///        rmax="Env0_rmax" dz="Env0_dz" material="Air"/>
///         <composite name="Barrel0"/>
///         <composite name="nEndCap0"/>
///         <composite name="pEndCap0"/>
///     </detector>
///   </detectors>
///   @endcode
///
///   If a user wants to create his/her own constructor to group these
///   volumes together the type needs to be set to "compound".
///
///	* Since the translation walks trough the \a DetElement tree the following
/// objects
///   need to be declared as a \a %DD4hep \a DetElement:
/// 	- the subvolumes e.g. \b barrel, \b endcap, \b beampipe (they are usually
/// build
/// with
///		  different \a %DD4hep constructors and are therefore \a %DD4hep \a DetElement's
/// per
/// 	  default)
/// 	- \b layers when containing sensitive material and/or the layer should
/// carry
/// 	  material (which will be mapped on the layer if indicated)
///       @note the layer does not need to be a direct child of the volume (barrel or endcap), it can be nested in substructures
///
///	    - sensitive detector modules
///		  @note the sensitive detector modules need to be placed in a layer however it can be nested in substructures (can be a component of a modules) i.e. it does not need to be a direct child of the layer
///
/// * The numbering of the id's of the sub detectors needs to be ascending.
///	  This is needed because the Tracking geometry is built from bottom to top to ensure Navigation.
///
/// * The Acts::ActsExtension's need to be used during the detector construction
///    indicating if a \a DetElement
///		- is a barrel
///		- is an endcap
///		- is the beampipe
///		- is a layer
///
///
/// Furthermore parameters can be handed over for material mapping or the axes
///   orientation of modules.
///
///
/// Summing up the \a DetElement tree in \a %DD4hep should have the following
/// structure:
/// \image html DD4hepPlugin1.jpeg
///
/// <B>Usage</B>
///
/// To receive the Acts::TrackingGeometry the user should use the global function
/// Acts::convertDD4hepDetector(), where he/she needs to hand over the
/// world \a DetElement of \a %DD4hep.
/// For a valid translation the user needs to make sure, that all prerequisites
/// described above are met and that the right
/// Acts::ActsExtension's are added during the \a %DD4hep construction.

/// @defgroup Contributing Contribution guide

// clang-format on
#endif  // ACTS_MODULES_HPP
