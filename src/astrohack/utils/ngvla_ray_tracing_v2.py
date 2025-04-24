from astrohack.utils.ray_tracing_general import *





class NgvlaRayTracer:

    def __init__(self):
        # Metadata
        self.experimental = True

        # Local QPS objects
        self.primary_qps = None
        self.secondary_qps = None

        # Data arrays
        self.x_axis = None
        self.y_axis = None
        self.grd_pm_points = None
        self.grd_pm_normals = None
        self.grd_idx = None
