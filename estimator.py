import cv2
import depthai as dai
import numpy as np

import datetime

from pathlib import Path


class DepthEstimator():

  def __init__(self, calibration: dai.CalibrationHandler, preview: bool,
               mode: str, subpixel: bool, extended: bool):
    modeMap = {
        "density": dai.node.StereoDepth.PresetMode.HIGH_DENSITY,
        "accuracy": dai.node.StereoDepth.PresetMode.HIGH_ACCURACY
    }

    self.calibration = calibration
    self.mode = modeMap[mode]
    self.preview = preview
    self.subpixel = subpixel
    self.extended = extended

  def initPipeline(self, rgbRes: tuple, monoRes: tuple):
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

  def estimateFromVideo(self, rgbPath: Path, leftPath: Path, rightPath: Path,
                        outputDirPath: Path) -> bool:
    # Check output path
    if not outputDirPath.exists():
      outputDirPath.mkdir(parents=True, exist_ok=True)

    # Get video
    rgbCapture = cv2.VideoCapture(str(rgbPath))
    leftCapture = cv2.VideoCapture(str(leftPath))
    rightCapture = cv2.VideoCapture(str(rightPath))
    rgbRes = (int(rgbCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(rgbCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    monoRes = (int(leftCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(leftCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Init output subdir path
    filename = rgbPath.name
    tagHeadIndex = filename.index("[")
    tagTailIndex = len(filename) - filename[::-1].index("]")
    tag = filename[tagHeadIndex:tagTailIndex]
    if self.subpixel:
      tag += "[Subpixel]"
    if self.extended:
      tag += "[Extended]"
    subDirPath = outputDirPath.joinpath(f"{tag}/")
    subDirPath.mkdir(exist_ok=True)

    self.pipeline = self.initPipeline(rgbRes, monoRes)
    with dai.Device(self.pipeline) as device:
      # Init queues
      leftInQ = device.getInputQueue("leftIn", 100, blocking=True)
      rightInQ = device.getInputQueue("rightIn", 100, blocking=True)
      depthOutQ = device.getOutputQueue("depthOut", 100, blocking=True)
      if self.preview:
        disparityOutQ = device.getOutputQueue(
            "disparityOut", 10, blocking=False)

      index: int = 0
      canceled: bool = False
      while True:
        try:
          print(f"\r{tag}: Frame {index}...", end="")
          # Send stereo frames
          leftExist, leftFrame = leftCapture.read()
          rightExist, rightFrame = rightCapture.read()
          if leftExist and rightExist:
            timestamp = datetime.timedelta(index)

            leftFrame = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
            leftImage = dai.ImgFrame()
            leftImage.setData(
                leftFrame.reshape(self.monoRes[0] * self.monoRes[1]))
            leftImage.setTimestamp(timestamp)
            leftImage.setInstanceNum(dai.CameraBoardSocket.LEFT)
            leftImage.setType(dai.ImgFrame.Type.RAW8)
            leftImage.setWidth(self.monoRes[0])
            leftImage.setHeight(self.monoRes[1])

            rightFrame = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)
            rightImage = dai.ImgFrame()
            rightImage.setData(
                rightFrame.reshape(self.monoRes[0] * self.monoRes[1]))
            rightImage.setTimestamp(timestamp)
            rightImage.setInstanceNum(dai.CameraBoardSocket.RIGHT)
            rightImage.setType(dai.ImgFrame.Type.RAW8)
            rightImage.setWidth(self.monoRes[0])
            rightImage.setHeight(self.monoRes[1])

            leftInQ.send(leftImage)
            rightInQ.send(rightImage)

            # Get depth output
            depthFrame = depthOutQ.get().getFrame()
            depthPath = subDirPath.joinpath(f"{str(index).zfill(10)}.npy")
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
                  break

            index += 1
          else:
            break
        except KeyboardInterrupt:
          canceled = True
          print("\nCancel...")
          break

    rgbCapture.release()
    leftCapture.release()
    rightCapture.release()

    if canceled:
      return False
    else:
      return True
