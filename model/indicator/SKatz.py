from .BKatz import BaseKatzIndex
class SimpleKatzIndex(BaseKatzIndex):
    def __init__(self):         
        super().__init__(binarization=True)     