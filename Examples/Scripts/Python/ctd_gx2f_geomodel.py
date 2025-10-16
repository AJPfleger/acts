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
    inputParticlePath = Path("./particles")
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
        conf_const(
            CsvMeasurementReader,
            level=acts.logging.INFO,
            outputMeasurements="measurements",
            outputMeasurementSimHitsMap="simhitsmap",
            outputMeasurementParticlesMap="meas_ptcl_map",
            inputSimHits=simAlg.config.outputSimHits,
            inputDir=str(""),
        )
    )


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
