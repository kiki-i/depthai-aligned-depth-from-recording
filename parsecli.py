from pathlib import Path

import argparse


def parseCli():
  description = "Record RGB and stereo video with DepthAI OAK-D"
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument(
      "-p",
      "--preview",
      action="store_const",
      const=True,
      default=False,
      help="Show preview")
  parser.add_argument(
      "-i",
      "--input",
      type=str,
      metavar="",
      default="input",
      help="Input files directory path, default=\"input\"")
  parser.add_argument(
      "-o",
      "--out",
      type=str,
      metavar="",
      default="output",
      help="Output directory path, default=\"output\"")
  parser.add_argument(
      "-m",
      "--mode",
      type=str,
      choices=["density", "accuracy"],
      metavar="",
      default="density",
      help="Depth mode: [\"density\", \"accuracy\"], default value is \"density\""
  )
  parser.add_argument(
      "-s",
      "--subpixel",
      action="store_const",
      const=True,
      default=False,
      help="Enable subpixel disparity")
  parser.add_argument(
      "-e",
      "--extended",
      action="store_const",
      const=True,
      default=False,
      help="Enable extended disparity")
  args = parser.parse_args()
  return args


def parseInputStereo(inputDirPath: Path) -> set[str]:
  if not inputDirPath.exists():
    raise FileNotFoundError(f"{inputDirPath.absolute()} don't esist!")

  inputStereos: set[str] = set()
  for file in inputDirPath.iterdir():
    filename = file.name
    if filename.endswith(".json"):
      tagHeadIndex = filename.index("[")
      tagTailIndex = len(filename) - filename[::-1].index("]")
      inputStereos.add(filename[tagHeadIndex:tagTailIndex])

  return inputStereos
