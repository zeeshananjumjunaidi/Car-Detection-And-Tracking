# CONSTANTS FOR FEATURES SELECTION

CSPACE = 'YCrCb'            # default: 'RGB' other values could be
HIST_RANGE = (0, 256)       # default: (0, 256)
HIST_BIN = 16               # default: 32
SPATIAL_SIZE = 16           # default: 32

#HOG SELECTION CONSTANTS
HOG_CELL_PER_BLOCK = 2      # default: 2
HOG_CHANNEL = '0'           # default: 0, values: 0, 1, 2, 'ALL', works good with RGB channels but value but slow
HOG_PIX_PER_CELL = 8        # default: 8
HOG_ORIENT_BINS = 8         # default: 9, typically between 6 and 12

FRAMES_TO_PRESERVE = 5     # default: 1

#CONSTANT FOR OTHER FEATURES

LANE_DETECTION = True       # default: False