import tempfile
from pathlib import Path
import cog
import mmcv
import cv2
from inference_fast import Inferencer
from utils import util
from model.model import ResHalf


class Predictor(cog.Predictor):
    def setup(self):
        self.invhalfer = Inferencer(
            checkpoint_path="checkpoints/model_best.pth.tar", model=ResHalf(train=False)
        )

    @cog.input(
        "image",
        type=Path,
        help="input image. Output will be the dithered input image (halftone) and the restored image "
        "from the generated halftone.",
    )
    def predict(self, image):
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        color = mmcv.imread(str(image), flag="color") / 127.5 - 1.0
        refHalf = mmcv.imread(str(image), flag="grayscale") / 127.5 - 1.0
        h, c = self.invhalfer(util.img2tensor(color), util.img2tensor(refHalf))
        h = util.tensor2img(h / 2.0 + 0.5) * 255.0
        c = util.tensor2img(c / 2.0 + 0.5) * 255.0
        mmcv.imwrite(h, "halftone.png")
        mmcv.imwrite(c, "restored.png")
        # concatenate gray and color image using opencv
        halftone = cv2.imread("halftone.png")
        restored = cv2.imread("restored.png")
        result = cv2.vconcat([restored, halftone])
        cv2.imwrite(str(out_path), result)
        return out_path
