// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <boost/test/unit_test.hpp>

#include "Acts/Utilities/Zip.hpp"
#include "ActsExamples/EventData/SimHit.hpp"
#include "ActsExamples/EventData/SimParticle.hpp"
#include "ActsExamples/Io/Root/RootSimHitReader.hpp"
#include "ActsExamples/Io/Root/RootSimHitWriter.hpp"
#include "ActsTests/CommonHelpers/FloatComparisons.hpp"
#include "ActsTests/CommonHelpers/WhiteBoardUtilities.hpp"

#include "ActsExamples/Io/Root/RootMuonSpacePointReader.hpp"
#include "ActsExamples/Digitization/MeasurementCreation.hpp"
#include "ActsExamples/Io/Csv/CsvMeasurementWriter.hpp"

#include <fstream>
#include <random>

using namespace Acts;
using namespace ActsExamples;

std::mt19937 gen(23);

auto makeTestSimhits(std::size_t nSimHits) {
  std::uniform_int_distribution<std::uint64_t> distIds(
      1, std::numeric_limits<std::uint64_t>::max());
  std::uniform_int_distribution<std::int32_t> distIndex(1, 20);

  SimHitContainer simhits;
  for (auto i = 0ul; i < nSimHits; ++i) {
    GeometryIdentifier geoid(distIds(gen));
    SimBarcode pid =
        SimBarcode()
            .withVertexPrimary(
                static_cast<SimBarcode::PrimaryVertexId>(distIds(gen)))
            .withVertexSecondary(
                static_cast<SimBarcode::SecondaryVertexId>(distIds(gen)))
            .withParticle(static_cast<SimBarcode::ParticleId>(distIds(gen)))
            .withGeneration(static_cast<SimBarcode::GenerationId>(distIds(gen)))
            .withSubParticle(
                static_cast<SimBarcode::SubParticleId>(distIds(gen)));

    Vector4 pos4 = Vector4::Random();
    Vector4 before4 = Vector4::Random();
    Vector4 after4 = Vector4::Random();

    auto index = distIndex(gen);

    simhits.insert(SimHit(geoid, pid, pos4, before4, after4, index));
  }

  return simhits;
}

namespace ActsTests {

BOOST_AUTO_TEST_SUITE(RootSuite)

BOOST_AUTO_TEST_CASE(Converter) {
MuonSpacePointContainer mspContainer;

RootMuonSpacePointReader::Config readerConfig;
readerConfig.filePath = "./MS_SpacePoints.root";

RootMuonSpacePointReader reader(readerConfig, Logging::INFO);
auto readTool = GenericReadWriteTool<>()
        .add(readerConfig.outputSpacePoints, mspContainer);
const auto [all_msp] = readTool.read(reader);
reader.finalize();

MeasurementContainer measurements;

measurements.reserve(all_msp.size());

IndexMultimap<Index> mapOriginal;
int i = 0;
for (auto& msp_vec : all_msp) {
  // std::cout << msp_vec << std::endl;
  for (auto& msp : msp_vec) {

    // Only take Mdt
    if (!msp.isStraw()) {
      continue;
    }
    auto geoId = msp.geometryId();
    auto pos = msp.localPosition();
    auto cov = msp.covariance();
    std::cout << "\nnew msp\n" << geoId << "\n local pos " << pos 
    << "\n driftradius " << msp.driftRadius() << std::endl;
            std::cout << "covariance" << std::endl;
    for (auto& c : cov) {
      std::cout << c << std::endl;
    }

    // Local position is in volume frame.
    // We need drift radius in surface frame.

    DigitizedParameters dParameters;

    dParameters.indices.push_back(Acts::eBoundLoc1);
    dParameters.values.push_back(msp.driftRadius());
    dParameters.variances.push_back(cov[1]);

    // this one should be z-distance
    dParameters.indices.push_back(Acts::eBoundLoc0);
    // parentTrf inverse l158
    dParameters.values.push_back(pos[0]);
    dParameters.variances.push_back(cov[0]);
    
    // dParameters.indices.push_back(Acts::eBoundTime);
    // dParameters.values.push_back(pos[2]);
    // dParameters.variances.push_back(cov[2]);

    auto measurement = createMeasurement(measurements, geoId, dParameters);
    mapOriginal.insert(std::pair<Index, Index>{i, i});
  }
  i++;
  break;
}


CsvMeasurementWriter::Config writerConfig;
writerConfig.inputMeasurements = "meas";
writerConfig.inputMeasurementSimHitsMap = "map";
writerConfig.outputDir = "";

CsvMeasurementWriter writer(writerConfig, Logging::VERBOSE);

auto writeTool = GenericReadWriteTool<>()
        .add(writerConfig.inputMeasurements, measurements)
        .add(writerConfig.inputMeasurementSimHitsMap, mapOriginal);

writeTool.write(writer);


    // Write the fake simhits
{
  auto simhits1 = makeTestSimhits(20);

  ///////////
  // Write //
  ///////////
  RootSimHitWriter::Config SHwriterConfig;
  SHwriterConfig.inputSimHits = "simhits";
  SHwriterConfig.filePath = "./fakesimhits.root";

  RootSimHitWriter SHwriter(SHwriterConfig, Logging::WARNING);

  auto SHreadWriteTool =
      GenericReadWriteTool<>().add(SHwriterConfig.inputSimHits, simhits1);

  // Write two different events
  SHreadWriteTool.write(SHwriter, 11);

  SHwriter.finalize();
}
// {
//   auto simhits1 = makeTestSimhits(20);
//
//   ///////////
//   // Write //
//   ///////////
//   CsvSimHitWriter::Config SHwriterConfig;
//   SHwriterConfig.inputSimHits = "simhits";
//   SHwriterConfig.filePath = "./fakesimhits.csv";
//
//   CsvSimHitWriter SHwriter(SHwriterConfig, Logging::WARNING);
//
//   auto SHreadWriteTool =
//       GenericReadWriteTool<>().add(SHwriterConfig.inputSimHits, simhits1);
//
//   // Write two different events
//   SHreadWriteTool.write(SHwriter, 11);
//
//   SHwriter.finalize();
// }
}

// BOOST_AUTO_TEST_CASE(RoundTripTest) {
//   ////////////////////////////
//   // Create some dummy data //
//   ////////////////////////////
//   auto simhits1 = makeTestSimhits(20);
//   auto simhits2 = makeTestSimhits(15);
//
//   ///////////
//   // Write //
//   ///////////
//   RootSimHitWriter::Config writerConfig;
//   writerConfig.inputSimHits = "hits";
//   writerConfig.filePath = "./testhits.root";
//
//   RootSimHitWriter writer(writerConfig, Logging::WARNING);
//
//   auto readWriteTool =
//       GenericReadWriteTool<>().add(writerConfig.inputSimHits, simhits1);
//
//   // Write two different events
//   readWriteTool.write(writer, 11);
//
//   std::get<0>(readWriteTool.tuple) = simhits2;
//   readWriteTool.write(writer, 22);
//
//   writer.finalize();
//
//   //////////
//   // Read //
//   //////////
//   RootSimHitReader::Config readerConfig;
//   readerConfig.outputSimHits = "hits";
//   readerConfig.filePath = "./testhits.root";
//
//   RootSimHitReader reader(readerConfig, Logging::WARNING);
//   // Read two different events
//   const auto [hitsRead2] = readWriteTool.read(reader, 22);
//   const auto [hitsRead1] = readWriteTool.read(reader, 11);
//   reader.finalize();
//
//   ///////////
//   // Check //
//   ///////////
//
//   auto check = [](const auto &testhits, const auto &refhits, auto tol) {
//     BOOST_CHECK_EQUAL(testhits.size(), refhits.size());
//
//     for (const auto &[ref, test] : zip(refhits, testhits)) {
//       CHECK_CLOSE_ABS(test.fourPosition(), ref.fourPosition(), tol);
//       CHECK_CLOSE_ABS(test.momentum4After(), ref.momentum4After(), tol);
//       CHECK_CLOSE_ABS(test.momentum4Before(), ref.momentum4Before(), tol);
//
//       BOOST_CHECK_EQUAL(ref.geometryId(), test.geometryId());
//       BOOST_CHECK_EQUAL(ref.particleId(), test.particleId());
//       BOOST_CHECK_EQUAL(ref.index(), test.index());
//     }
//   };
//
//   check(hitsRead1, simhits1, 1.e-6);
//   check(hitsRead2, simhits2, 1.e-6);
// }

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
