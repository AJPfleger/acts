#!/usr/bin/env python3
import pathlib, acts, acts.examples, acts.examples.itk
from acts.examples.simulation import (
    addParticleGun,
    MomentumConfig,
    EtaConfig,
    PhiConfig,
    ParticleConfig,
    addPythia8,
    addFatras,
    ParticleSelectorConfig,
    addDigitization,
)
from acts.examples.reconstruction import (
    addSeeding,
    SeedingAlgorithm,
    TruthSeedRanges,
    addCKFTracks,
    TrackSelectorConfig,
    addAmbiguityResolution,
    AmbiguityResolutionConfig,
    addVertexFitting,
    VertexFinder,
    addGx2fTracks,
)

from typing import Optional, Union, List
# def addTruthTrackFinder(
#         sequence: acts.examples.Sequencer,
#         logLevel: acts.logging.Level = None,
#         truthSeedRanges: Optional[TruthSeedRanges] = TruthSeedRanges(),
# ):
#
#     if truthSeedRanges is not None:
#         selectedParticles = "truth_seeds_selected"
#         addSeedingTruthSelection(
#             s,
#             inputParticles,
#             selectedParticles,
#             truthSeedRanges,
#             logLevel,
#         )
#     else:
#         selectedParticles = inputParticles
#
#     truthTrkFndAlg = acts.examples.TruthTrackFinder(
#         level=logLevel,
#         inputParticles=selectedParticles,
#         inputMeasurementParticlesMap="measurement_particles_map",
#         outputProtoTracks="truth_particle_tracks",
#     )
#     sequence.addAlgorithm(truthTrkFndAlg)

def runItkFc(ETA, PHI):

    ttbar_pu200 = False
    u = acts.UnitConstants
    geo_dir = pathlib.Path("../../acts-itk")
    actxtra_dir = pathlib.Path("../../actxtra")
    outputDir = pathlib.Path.cwd() / "itk_output"
    # acts.examples.dump_args_calls(locals())  # show acts.examples python binding calls

    detector, trackingGeometry, decorators = acts.examples.itk.buildITkGeometry(geo_dir)
    field = acts.examples.MagneticFieldMapXyz(str(geo_dir / "bfield/ATLAS-BField-xyz.root"))
    rnd = acts.examples.RandomNumbers(seed=42)

    s = acts.examples.Sequencer(events=200000, numThreads=-1, outputDir=str(outputDir),
                                logLevel=acts.logging.ERROR)

    if not ttbar_pu200:
        addParticleGun(
            s,
            MomentumConfig(100.0 * u.GeV, 100.0 * u.GeV, transverse=True),
            # EtaConfig(ETA, ETA, uniform=True),
            # PhiConfig(PHI * u.degree, PHI * u.degree),
            EtaConfig(-2, 2, uniform=True),
            PhiConfig(0 * u.degree, 360 * u.degree),
            ParticleConfig(2, acts.PdgParticle.eMuon, randomizeCharge=True),
            rnd=rnd,
        )
    else:
        addPythia8(
            s,
            hardProcess=["Top:qqbar2ttbar=on"],
            npileup=200,
            vtxGen=acts.examples.GaussianVertexGenerator(
                stddev=acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns),
                mean=acts.Vector4(0, 0, 0, 0),
            ),
            rnd=rnd,
            outputDirRoot=outputDir,
        )

    addFatras(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        preSelectParticles=ParticleSelectorConfig(
            rho=(0.0 * u.mm, 28.0 * u.mm),
            absZ=(0.0 * u.mm, 1.0 * u.m),
            # eta=(0.05, 0.05),
            # phi=(0, 360),
            pt=(150 * u.MeV, None),
            removeNeutral=True,
        )
        if ttbar_pu200
        else ParticleSelectorConfig(),
        outputDirRoot=outputDir,
    )

    addDigitization(
        s,
        trackingGeometry,
        field,
        # digiConfigFile=geo_dir
        #                / "itk-hgtd/itk-smearing-config.json",  # change this file to make it do digitization
        digiConfigFile=actxtra_dir
                       # / "scripts/configs/itk-smearing-config-pixelBarrel.json",  # change this file to make it do digitization
                       / "scripts/configs/itk-smearing-config-fullBarrel.json",  # change this file to make it do digitization
        outputDirRoot=outputDir,
        rnd=rnd,
    )

    # addSeeding(
    #     s,
    #     trackingGeometry,
    #     field,
    #     TruthSeedRanges(pt=(1.0 * u.GeV, None), eta=(-4.0, 4.0), nHits=(9, None))
    #     if ttbar_pu200
    #     else TruthSeedRanges(),
    #     seedingAlgorithm=SeedingAlgorithm.Gbts,
    #     *acts.examples.itk.itkSeedingAlgConfig(
    #         acts.examples.itk.InputSpacePointsType.PixelSpacePoints
    #     ),
    #     geoSelectionConfigFile=geo_dir / "itk-hgtd/geoSelection-ITk.json",
    #     layerMappingConfigFile=geo_dir / "itk-hgtd/ACTS_FTF_mapinput.csv",
    #     connector_inputConfigFile=geo_dir / "itk-hgtd/binTables_ITK_RUN4.txt",
    #     )
    # we don't use the truth seeding



    addSeeding(
        s,
        trackingGeometry,
        field,
        seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
        rnd=rnd,
        truthSeedRanges=TruthSeedRanges(
            pt=(1 * u.GeV, None),
            nHits=(5, None),
        ),
        # geoSelectionConfigFile=actxtra_dir / "scripts/geoSelection-ITk-pixelBarrel.json",
    )

    # addTruthTrackFinder(
    #     s,
    #     truthSeedRanges=TruthSeedRanges(
    #         pt=(1 * u.GeV, None),
    #         nHits=(7, None),
    #     ),
    # )

    # add the truth tracks here
    # dump out the whiteboard

    addGx2fTracks(
        s,
        trackingGeometry,
        field,
        nUpdateMax=27,
        relChi2changeCutOff=1e-7,
        logLevel=acts.logging.ERROR,
    )

    # # Output
    # s.addWriter(
    #     acts.examples.RootTrackStatesWriter(
    #         level=acts.logging.INFO,
    #         inputTracks="tracks",
    #         inputParticles="truth_seeds_selected",
    #         inputSimHits="simhits",
    #         inputMeasurementParticlesMap="measurement_particles_map",
    #         inputMeasurementSimHitsMap="measurement_simhits_map",
    #         filePath=str(outputDir / "itk_trackstates_fitter.root"),
    #     )
    # )

    s.addWriter(
        acts.examples.RootTrackStatesWriter(
            level=acts.logging.INFO,
            inputTracks="tracks",
            inputParticles="truth_seeds_selected",
            inputSimHits="simhits",
            inputMeasurementParticlesMap="measurement_particles_map",
            inputMeasurementSimHitsMap="measurement_simhits_map",
            filePath=str(outputDir / "itk_trackstates_fitter.root"),
        )
    )


    s.addWriter(
        acts.examples.RootTrackSummaryWriter(
            level=acts.logging.ERROR,
            inputTracks="tracks",
            inputParticles="truth_seeds_selected",
            inputMeasurementParticlesMap="measurement_particles_map",
            filePath=str(outputDir / "itk_tracksummary_fitter.root"),
            writeGx2fSpecific=True,
        )
    )


    import time
    startTime = time.time()
    s.run()

    executionTime = (time.time() - startTime)
    print('Execution time in minutes: ' + str(executionTime/60))



if "__main__" == __name__:
    # srcdir = Path(__file__).resolve().parent.parent.parent.parent

    runItkFc(0.05, 2)


    # # detector, trackingGeometry, _ = getOpenDataDetector()
    # detector, trackingGeometry, decorators = acts.examples.GenericDetector.create()
    #
    # field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))
    #
    # runTruthTrackingGx2f(
    #     trackingGeometry=trackingGeometry,
    #     # decorators=decorators,
    #     field=field,
    #     digiConfigFile=srcdir
    #                    / "Examples/Algorithms/Digitization/share/default-smearing-config-generic.json",
    #     # "thirdparty/OpenDataDetector/config/odd-digi-smearing-config.json",
    #     # outputCsv=True,
    #     # inputParticlePath=inputParticlePath,
    #     outputDir=Path.cwd(),
    # ).run()




#  284 11:15:40    TrackFitting   VERBOSE   Initial parameters:
#  0.00336065 0.00974549 -0.0110047    410.251 ->  0.940455 -0.324308  0.101824
# 5395 oldChi2sum = 23.5958
# 5396 chi2sum = 23.5958
# 5397 11:15:40    Gx2fFitter     VERBOSE   Abort with relChi2changeCutOff after 4/17 iterations.
# 5398 11:15:40    Gx2fFitter     DEBUG     Finished to iterate
# 5399 11:15:40    Gx2fFitter     VERBOSE   final params:
# 5400 loc0:     0.01433 +- 0.02         1.000
# 5401 loc1:     0.00143 +- 0.02         0.000  1.000
# 5402 phi:      -0.3406 +- 0.01745      0.000  0.000  1.000
# 5403 theta:      1.471 +- 0.01745      0.000  0.000  0.000  1.000
# 5404 q/p:     -0.01148 +- 0.0004975    0.000  0.000  0.000  0.000  1.000
# 5405 time:       410.3 +- 299.8        0.000  0.000  0.000  0.000  0.000  1.000
# 5406 on surface undefined of type Acts::PerigeeSurface
# 5407 11:15:40    Gx2fFitter     VERBOSE   final covariance:
# 5408  0.000217433  3.30924e-06 -1.70608e-06   1.9363e-08 -9.02334e-06            0
# 5409  3.30924e-06  5.62394e-05 -8.38682e-08  8.29149e-08 -8.56657e-07            0
# 5410 -1.70608e-06 -8.38682e-08  1.92181e-08 -4.04405e-10  1.31657e-07            0
# 5411   1.9363e-08  8.29149e-08 -4.04405e-10  1.73981e-10 -4.30196e-09            0
# 5412 -9.02334e-06 -8.56657e-07  1.31657e-07 -4.30196e-09  1.15696e-06            0
# 5413            0            0            0            0            0            1