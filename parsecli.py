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
