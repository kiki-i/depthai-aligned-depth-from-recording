#!/usr/bin/env python3

from estimator import *
from parsecli import *

from pathlib import Path

if __name__ == "__main__":
  scriptPath = Path(__file__).parent

  # Init
  print("Init...")
  cliArgs = parseCli()
  inputDirPath = Path(cliArgs.input)
  if not inputDirPath.exists():
    raise FileNotFoundError(f"{inputDirPath.absolute()} don't esist!")

  ## Check output path
  outputDirPath = Path(cliArgs.out)
  outputDirPath.mkdir(parents=True, exist_ok=True)

  ## Get file list
  inputDirs: list[Path] = []
  for path in inputDirPath.iterdir():
    if path.is_dir():
      inputDirs.append(path)

  results: dict[Path, bool] = {}
  for inputDir in inputDirs:
    depthEstimator = DepthEstimator(cliArgs.preview, cliArgs.mode,
                                    cliArgs.subpixel, cliArgs.extended)

    results[inputDir] = depthEstimator.estimateFromVideo(
        inputDir,
        outputDirPath,
    )

  for key, result in results.items():
    if not result:
      print(f"!Warning: {key} is incompleted!")
  print(f"Output files in: \"{outputDirPath.absolute()}\"")
