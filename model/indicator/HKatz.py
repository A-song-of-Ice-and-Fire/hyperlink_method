from .BKatz import BaseKatzIndex
class HyperKatzIndex(BaseKatzIndex):
    def __init__(self):         
        super().__init__(binarization=False)     