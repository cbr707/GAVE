from Model import model
from Loss import loss


class UniversalFactory:
    classes = []

    def __init__(self, classes=None):
        if classes is not None:
            self.classes = classes
        self.classes_names = {class_.__name__: class_ for class_ in self.classes}

    def create_class(self, class_name, *args, **kwargs):
        instance = self.classes_names[class_name](*args, **kwargs)
        return instance


class ModelFactory(UniversalFactory):
    classes = [
        model.UNet,
        model.GAVENet,
        model.GAVENetV2,
        model.GAVENetV3,
        model.SMPUNet,
        model.SMPUNetV2,
        model.SMPUNetV3,
        model.SMPGAVENet,
        model.SMPGAVENetV2,
        model.SMPGAVENetV3,
        model.SMPGAVENetV4,
        model.RRSMPUNet,
    ]


class LossesFactory(UniversalFactory):
    classes = [
        loss.BCE3Loss,
        loss.BCE4Loss,
        loss.BCE4LossNew,
        loss.RRLoss,
        loss.RRLossNew,
        loss.BCETverskyLossVT,
        loss.BCETverskyLossA,
        loss.BCETverskyLossV,
        loss.BCETverskyLossAV,
        loss.BCETversky2Loss,
        loss.BCETversky3Loss,
        loss.BCETversky4Loss,
        loss.BTCB3Loss

    ]
