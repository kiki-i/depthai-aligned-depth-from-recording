import cv2
import depthai as dai
import numpy as np

import datetime
import re

from pathlib import Path


class DepthEstimator():

  def __init__(self, preview: bool, mode: str, subpixel: bool, extended: bool):
    modeMap = {
        "density": dai.node.StereoDepth.PresetMode.HIGH_DENSITY,
        "accuracy": dai.node.StereoDepth.PresetMode.HIGH_ACCURACY
    }

    self.mode = modeMap[mode]
    self.preview = preview
    self.subpixel = subpixel
    self.extended = extended

  def initPipeline(self, calibration: dai.CalibrationHandler, rgbRes: tuple,
                   monoRes: tuple):
    self.calibration = calibration
    self.rgbRes = rgbRes
    self.monoRes = monoRes

    pipeline = dai.Pipeline()

    # Init nodes
    stereo = pipeline.create(dai.node.StereoDepth)

    leftIn = pipeline.create(dai.node.XLinkIn)
    rightIn = pipeline.create(dai.node.XLinkIn)
    depthOut = pipeline.create(dai.node.XLinkOut)

    leftIn.setStreamName("leftIn")
    rightIn.setStreamName("rightIn")
    depthOut.setStreamName("depthOut")

    # Link nodes
    leftIn.out.link(stereo.left)
    rightIn.out.link(stereo.right)
    stereo.depth.link(depthOut.input)

    # Config
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setInputResolution(monoRes[0], monoRes[1])
    stereo.setOutputSize(rgbRes[0], rgbRes[1])
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(self.subpixel)
    stereo.setExtendedDisparity(self.extended)

    # Disparity preview
    if self.preview:
      disparityOut = pipeline.create(dai.node.XLinkOut)
      stereo.disparity.link(disparityOut.input)
      disparityOut.setStreamName("disparityOut")
      maxDisparity = stereo.initialConfig.getMaxDisparity()
      self.maxDisparity = maxDisparity
    return pipeline

  def estimateFromVideo(self, inputDirPath: Path, outputDirPath: Path) -> bool:

    # Init path
    for path in inputDirPath.iterdir():
      if path.stem.endswith(f"rgb"):
        rgbPath = path
      if path.stem.endswith(f"left"):
        leftPath = path
      if path.stem.endswith(f"right"):
        rightPath = path

    timestampPath = inputDirPath.joinpath(f"timestamp.txt")
    calibration = dai.CalibrationHandler(
        inputDirPath.joinpath("calibration.json"))

    outputDirPath.mkdir(parents=True, exist_ok=True)

    # Get video
    leftCapture = cv2.VideoCapture(str(leftPath))
    rightCapture = cv2.VideoCapture(str(rightPath))

    # Get resolution
    resMap = {
        "12MP": (4056, 3040),
        "4K": (3840, 2160),
        "1080P": (1920, 1080),
        "800P": (1280, 800),
        "720P": (1280, 720),
        "400P": (640, 400)
    }
    rgbRes = resMap[re.search(r"\[(.+)?\]", str(rgbPath.stem)).group(1)]
    monoRes = resMap[re.search(r"\[(.+)?\]", str(leftPath.stem)).group(1)]

    # Init output subdir path
    tag = inputDirPath.name
    if self.subpixel:
      tag += "[Subpixel]"
    if self.extended:
      tag += "[Extended]"
    subDirPath = outputDirPath.joinpath(f"{tag}/")
    subDirPath.mkdir(exist_ok=True)

    self.pipeline = self.initPipeline(calibration, rgbRes, monoRes)
    with dai.Device(self.pipeline) as device, open(timestampPath,
                                                   "rt") as timestampFile:
      # Init queues
      leftInQ = device.getInputQueue("leftIn", 10, blocking=True)
      rightInQ = device.getInputQueue("rightIn", 10, blocking=True)
      depthOutQ = device.getOutputQueue("depthOut", 10, blocking=True)
      if self.preview:
        disparityOutQ = device.getOutputQueue(
            "disparityOut", 10, blocking=False)

      index: int = 0
      canceled: bool = True
      while True:
        try:
          print(f"\r{tag}: Frame {index}...", end="")

          # Read files
          leftExist, leftFrame = leftCapture.read()
          rightExist, rightFrame = rightCapture.read()
          timestamp = timestampFile.readline().rstrip().replace(":", ";")

          if leftExist and rightExist:
            # Create ImgFrame for StereoDepth node
            frameTimestamp = datetime.timedelta(index)

            leftFrame = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
            leftImage = dai.ImgFrame()
            leftImage.setData(
                leftFrame.reshape(self.monoRes[0] * self.monoRes[1]))
            leftImage.setTimestamp(frameTimestamp)
            leftImage.setInstanceNum(dai.CameraBoardSocket.LEFT)
            leftImage.setType(dai.ImgFrame.Type.RAW8)
            leftImage.setWidth(self.monoRes[0])
            leftImage.setHeight(self.monoRes[1])

            rightFrame = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)
            rightImage = dai.ImgFrame()
            rightImage.setData(
                rightFrame.reshape(self.monoRes[0] * self.monoRes[1]))
            rightImage.setTimestamp(frameTimestamp)
            rightImage.setInstanceNum(dai.CameraBoardSocket.RIGHT)
            rightImage.setType(dai.ImgFrame.Type.RAW8)
            rightImage.setWidth(self.monoRes[0])
            rightImage.setHeight(self.monoRes[1])

            leftInQ.send(leftImage)
            rightInQ.send(rightImage)

            # Get depth output
            depthFrame = depthOutQ.get().getFrame()
            depthPath = subDirPath.joinpath(f"{timestamp}.npy")
            np.save(str(depthPath), depthFrame)

            # Display disparity preview
            if self.preview:
              disparityOut = disparityOutQ.tryGet()
              if disparityOut is not None:
                disparityFrame = disparityOut.getFrame()
                disparityFrame = (disparityFrame * 255. /
                                  self.maxDisparity).astype(np.uint8)
                disparityFrame = cv2.applyColorMap(disparityFrame,
                                                   cv2.COLORMAP_BONE)
                disparityFrame = np.ascontiguousarray(disparityFrame)
                cv2.imshow("Preview", disparityFrame)
                if cv2.waitKey(1) == ord("q"):
                  print("\nCancel...")
                  break

            index += 1
          else:
            canceled = False
            print()
            break
        except KeyboardInterrupt:
          print("\nCancel...")
          break

    leftCapture.release()
    rightCapture.release()

    if canceled:
      return False
    else:
      return True
