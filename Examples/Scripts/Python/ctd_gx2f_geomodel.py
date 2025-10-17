import os
import acts
import argparse
from acts import (
    logging,
    GeometryContext,
    CylindricalContainerBuilder,
    DetectorBuilder,
    GeometryIdGenerator,
)

from acts.examples import (
    AlgorithmContext,
    WhiteBoard,
    ObjTrackingGeometryWriter,
    CsvMeasurementReader,
)

from acts.examples.reconstruction import (
    addGx2fTracks,
)

from acts import geomodel as gm
from acts import examples

from pathlib import Path
from propagation import runPropagation



def main():
    from argparse import ArgumentParser

    u = acts.UnitConstants

    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="",  # "/eos/user/c/cimuonsw/GeometryFiles/MockUp.db",
        help="Input SQL file",
    )
    parser.add_argument(
        "--mockupDetector",
        type=str,
        choices=["Muon"],
        help="Predefined mockup detector which is built transiently",
        default="Muon",
    )
    parser.add_argument("--outDir", default="./", help="Output")
    parser.add_argument("--nEvents", default=100, type=int, help="Number of events")
    parser.add_argument(
        "--randomSeed", default=1602, type=int, help="Random seed for event generation"
    )
    parser.add_argument(
        "--geoSvgDump",
        default=False,
        action="store_true",
        help="Dump the tracking geometry in an obj format",
    )

    args = parser.parse_args()

    gContext = acts.GeometryContext()
    logLevel = logging.INFO

    # Create the tracking geometry builder for the muon system
    gmBuilderConfig = gm.GeoModelMuonMockupBuilder.Config()

    # Read the geometry model from the database
    gmTree = None
    ### Use an external geo model file
    if len(args.input):
        gmTree = gm.readFromDb(args.input)
        gmBuilderConfig.stationNames = ["BIL", "BML", "BOL"]

    elif args.mockupDetector == "Muon":
        mockUpCfg = gm.GeoMuonMockupExperiment.Config()
        mockUpCfg.dumpTree = True
        mockUpCfg.dbName = "ActsGeoMS.db"
        mockUpCfg.nSectors = 12
        mockUpCfg.nEtaStations = 8
        mockUpCfg.buildEndcaps = False
        mockUpBuilder = gm.GeoMuonMockupExperiment(mockUpCfg, "GeoMockUpMS", logLevel)
        gmBuilderConfig.stationNames = ["Inner", "Middle", "Outer"]

        gmTree = mockUpBuilder.constructMS()
    else:
        raise RuntimeError(f"{args.mockupDetector} not implemented yet")

    gmFactoryConfig = gm.GeoModelDetectorObjectFactory.Config()
    gmFactoryConfig.nameList = [
        "RpcGasGap",
        "MDTDriftGas",
        "TgcGasGap",
        "SmallWheelGasGap",
    ]
    gmFactoryConfig.convertSubVolumes = True
    gmFactoryConfig.convertBox = ["MDT", "RPC"]

    gmFactory = gm.GeoModelDetectorObjectFactory(gmFactoryConfig, logLevel)
    # The options
    gmFactoryOptions = gm.GeoModelDetectorObjectFactory.Options()
    gmFactoryOptions.queries = ["Muon"]

    # The Cache & construct call
    gmFactoryCache = gm.GeoModelDetectorObjectFactory.Cache()
    gmFactory.construct(gmFactoryCache, gContext, gmTree, gmFactoryOptions)

    gmBuilderConfig.volumeBoxFPVs = gmFactoryCache.boundingBoxes

    gmDetectorCfg = gm.GeoModelDetector.Config()
    gmDetectorCfg.geoModelTree = gmTree
    detector = gm.GeoModelDetector(gmDetectorCfg)

    field = acts.ConstantBField(acts.Vector3(0, 0, 0 * u.T))

    trackingGeometryBuilder = gm.GeoModelMuonMockupBuilder(
        gmBuilderConfig, "GeoModelMuonMockupBuilder", logLevel
    )

    trackingGeometry = detector.buildTrackingGeometry(gContext, trackingGeometryBuilder)

    # Sequencer
    s = acts.examples.Sequencer(
        events=1, numThreads=-1, logLevel=acts.logging.INFO
    )

    # Load external particles. We only need them for the initial guess.
    inputParticlePath = Path("./particles_simulation.root")
    acts.logging.getLogger("CTD").info(
        "Reading particles from %s", inputParticlePath.resolve()
    )
    assert inputParticlePath.exists()
    s.addReader(
        acts.examples.RootParticleReader(
            level=acts.logging.INFO,
            filePath=str(inputParticlePath.resolve()),
            outputParticles="particles_generated",
        )
    )

    # Read measurements from file
    s.addReader(
        CsvMeasurementReader(
            level=acts.logging.INFO,
            outputMeasurements="measurements",
            outputMeasurementSimHitsMap="measurement_simhits_map",
            outputMeasurementParticlesMap="meas_ptcl_map",
            # inputSimHits=simAlg.config.outputSimHits,
            outputParticleMeasurementsMap="particle_measurements_map",
            inputDir=str(""),
            geometry4track=trackingGeometry,
        )
    )

    # Read fake simhits
    s.addReader(
        acts.examples.RootSimHitReader(
            level=acts.logging.INFO,
            filePath="./fakesimhits.root",
            outputSimHits="simhits",
        )
    )
    # s.addReader(
    #     acts.examples.CsvSimHitReader(
    #         level=acts.logging.INFO,
    #         outputSimHits="simhits",
    #         filePath="./fakesimhits.csv",
    #     )
    # )

    # Create truth_particle_tracks for gx2f
    s.addAlgorithm(
      acts.examples.TruthTrackFinder(
        level=acts.logging.INFO,
        inputParticles="particles_generated",
        inputMeasurements="measurements",
        inputParticleMeasurementsMap="particle_measurements_map",
        inputSimHits="simhits",
        inputMeasurementSimHitsMap="measurement_simhits_map",
        outputProtoTracks="truth_particle_tracks",
      )
    )

    # Set up the fitter
    addGx2fTracks(
        s,
        trackingGeometry,
        field,
        nUpdateMax=17,
        relChi2changeCutOff=1e-7,
        multipleScattering=True,
    )

    # s.addAlgorithm(
    #     acts.examples.TrackSelectorAlgorithm(
    #         level=acts.logging.INFO,
    #         inputTracks="tracks",
    #         outputTracks="selected-tracks",
    #         selectorConfig=acts.TrackSelector.Config(
    #             minMeasurements=7,
    #         ),
    #     )
    # )
    # s.addWhiteboardAlias("tracks", "selected-tracks")
    #
    # s.addWriter(
    #     acts.examples.RootTrackStatesWriter(
    #         level=acts.logging.INFO,
    #         inputTracks="tracks",
    #         inputParticles="particles_selected",
    #         inputTrackParticleMatching="track_particle_matching",
    #         inputSimHits="simhits",
    #         inputMeasurementSimHitsMap="measurement_simhits_map",
    #         filePath=str(outputDir / "trackstates_gx2f.root"),
    #     )
    # )
    #
    # s.addWriter(
    #     acts.examples.RootTrackSummaryWriter(
    #         level=acts.logging.INFO,
    #         inputTracks="tracks",
    #         inputParticles="particles_selected",
    #         inputTrackParticleMatching="track_particle_matching",
    #         filePath=str(outputDir / "tracksummary_gx2f.root"),
    #         writeGx2fSpecific=True,
    #     )
    # )



    # algSequence = runGeant4(
    #     detector=detector,
    #     trackingGeometry=trackingGeometry,
    #     field=field,
    #     outputDir=args.outDir,
    #     volumeMappings=gmFactoryConfig.nameList,
    #     events=args.nEvents,
    #     seed=args.randomSeed,
    # )
    #
    # from acts.examples import MuonSpacePointDigitizer
    #
    # digiAlg = MuonSpacePointDigitizer(
    #     randomNumbers=acts.examples.RandomNumbers(
    #         acts.examples.RandomNumbers.Config(seed=2 * args.randomSeed)
    #     ),
    #     trackingGeometry=trackingGeometry,
    #     dumpVisualization=False,
    #     digitizeTime=True,
    #     outputSpacePoints="MuonSpacePoints",
    #     level=logLevel,
    # )
    # algSequence.addAlgorithm(digiAlg)
    #
    # from acts.examples import RootMuonSpacePointWriter
    #
    # algSequence.addWriter(
    #     RootMuonSpacePointWriter(
    #         level=logLevel,
    #         inputSpacePoints="MuonSpacePoints",
    #         filePath=f"{args.outDir}/MS_SpacePoints.root",
    #     )
    # )
    #
    # if args.geoSvgDump:
    #     wb = WhiteBoard(acts.logging.INFO)
    #     context = AlgorithmContext(0, 0, wb, 10)
    #     obj_dir = Path(args.outDir) / "obj"
    #     obj_dir.mkdir(exist_ok=True)
    #     writer = ObjTrackingGeometryWriter(
    #         level=acts.logging.INFO, outputDir=str(obj_dir)
    #     )
    #
    #     writer.write(context, trackingGeometry)
    #
    # algSequence.run()


if __name__ == "__main__":
    print("start")
    main()
    print("end")
