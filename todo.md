ok, this will be in the cpp lib and be a general quad analysis class called QuadAnalysis that has several methods, topBottomParallel(quad, epsilon=0.0038) return true if top and bottom lines in quad are within epsilon of parallel, same for leftRightParallel(quad, epsilon=0.0038), topRightRatio(quad) computes and returns top line vs right line of quad ratio, same for topLeftRatio(quad)

there should be another method called orientation(quad), which will return one of SHORT_SIDE, LONG_SIDE, TOP_DOWN, OTHER (enum) which are determined by the following:

LONG_SIDE if topBottomParallel and not leftRightParallel, and topRightRatio >= 1.75
SHORT_SIDE if topBottomParallel and not leftRightParallel, and topRightRatio < 1.75
TOP_DOWN if topBottomParallel and leftRightParallel
OTHER otherwise.

This class should be called in the table detection ffi and the orientation should be returned along with the quad. if the orientation is LONG_SIDE, you will need to rotate the ball locations to match the long side of the table in landscape mode. 


NOTES WITH GAGE

1) The phone will be in landscape mode and 16:9 aspect to take a photo of the table from the short side


BUGS
1) 45deg angle is now called SHORT_SIDE


