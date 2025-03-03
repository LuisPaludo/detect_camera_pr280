class SafeBox:
    NOT_CONTAINS = 0
    PARTIAL = 1
    CONTAINS = 2

    def __init__(self, x1, x2, y1, y2, x3 = None, y3 = None):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3

    def contains(self, obj_x1, obj_x2, obj_y1, obj_y2):
        if not self.x3 and not self.y3:
            if obj_y2 > self.y2 and self.x1 < obj_x1 and self.x2 > obj_x2:
                return SafeBox.CONTAINS
            else:
                return SafeBox.NOT_CONTAINS
        else:
            if self.x1 < obj_x1 and self.x2 > obj_x2 and obj_y2 > (self.y1/2 - self.y3/2) :
                return SafeBox.CONTAINS
            else:
                return SafeBox.NOT_CONTAINS
