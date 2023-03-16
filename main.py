#!/usr/bin/env python3

from estimator import *
from parsecli import *

import depthai as dai

from pathlib import Path

if __name__ == "__main__":
  scriptPath = Path(__file__).parent

  # Init
  print("Init...")
  cliArgs = parseCli()
  inputDirPath = Path(cliArgs.input)

  ## Check output path
  outputDirPath = Path(cliArgs.out)
  if not outputDirPath.exists():
    outputDirPath.mkdir(parents=True, exist_ok=True)

  ## Get file list
  inputStereos = parseInputStereo(inputDirPath)
  results = {}

  for inputStereo in inputStereos:
    calibration = dai.CalibrationHandler(
        inputDirPath.joinpath(inputStereo + "calibration.json"))
    depthEstimator = DepthEstimator(calibration, cliArgs.preview, cliArgs.mode,
                                    cliArgs.subpixel, cliArgs.extended)

    extension = ""
    if inputDirPath.joinpath(inputStereo + "rgb.h265").exists():
      extension = "h265"
    if inputDirPath.joinpath(inputStereo + "rgb.mjpeg").exists():
      extension = "mjpeg"

    results[inputStereo] = depthEstimator.estimateFromVideo(
        inputDirPath.joinpath(inputStereo + f"rgb.{extension}"),
        inputDirPath.joinpath(inputStereo + f"left.{extension}"),
        inputDirPath.joinpath(inputStereo + f"right.{extension}"),
        outputDirPath,
    )

  for key, result in results.items():
    if not result:
      print(f"!Warning: {key} is incompleted!")
  print(f"Output files in: \"{outputDirPath.absolute()}\"")
