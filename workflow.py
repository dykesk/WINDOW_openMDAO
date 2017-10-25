from src.api import WakeModel
from WakeModel.jensen import JensenWakeFraction, JensenWakeDeficit
from openmdao.api import IndepVarComp, Problem, Group, view_model, ParallelGroup
import numpy as np
from time import time
from src.api import FarmAeroPower
from Power.power_models import PowerPolynomial
from input_params import turbine_radius
from WakeModel.WakeMerge.RSS import WakeMergeRSS


class FarmGroup(Group):
    def __init__(self, power_model, fraction_model, deficit_model, merge_model):
        super(FarmGroup, self).__init__()
        self.power_model = power_model
        self.merge_model = merge_model
        self.fraction_model = fraction_model
        self.deficit_model = deficit_model

    def setup(self):
        self.add_subsystem('farmg', Group())
        self.add_subsystem('wakemodel', WakeModel(self.fraction_model, self.deficit_model, self.merge_model),
                           promotes_inputs=['original', 'n_turbines', 'r'])
        self.add_subsystem('power', self.power_model(), promotes_inputs=['n_turbines'])
        self.add_subsystem('farmpower', FarmAeroPower(), promotes_inputs=['n_turbines'])

        self.connect('wakemodel.U', 'power.U')
        self.connect('power.p', 'farmpower.ind_powers')


class WorkingGroup(Group):
    def __init__(self, power_model, fraction_model, deficit_model, merge_model):
        super(WorkingGroup, self).__init__()
        self.power_model = power_model
        self.fraction_model = fraction_model
        self.deficit_model = deficit_model
        self.merge_model = merge_model

    def setup(self):
        indep2 = self.add_subsystem('indep2', IndepVarComp())
        # indep2.add_output('layout', val=read_layout('horns_rev9.dat'))
        indep2.add_output('layout', val=np.array([[0, 0.0, 0.0], [1, 560.0, 0.0], [2, 1120.0, 0.0], [3, 1680.0, 0.0],
                                                  [4, 0.0, 1120.0], [5, 0.0, 1120.0], [6, 0.0, 1120.0],
                                                  [7, 0.0, 1120.0], [8, 0.0, 1120.0], [9, 0.0, 1120.0]]))
        # indep2.add_output('layout', val=np.array(
        #     [[0, 0.0, 0.0], [1, 560.0, 560.0], [2, 1120.0, 1120.0], [3, 1120.0, 0.0], [4, 0.0, 1120.0],
        #     [5, 6666.6, 6666.6], [6, 6666.6, 6666.6], [7, 6666.6, 6666.6], [8, 6666.6, 6666.6], [9, 6666.6, 6666.6]]))
        # indep2.add_output('layout', val=np.array([[0, 0.0, 0.0], [1, 560.0, 560.0], [2, 1120.0, 1120.0],
        # [3, 1120.0, 0.0], [4, 0.0, 1120.0], [5, float('nan'), float('nan')]]))
        indep2.add_output('r', val=turbine_radius)
        indep2.add_output('n_turbines', val=4)
        indep2.add_output('freestream', val=[8.5, 8.0])
        indep2.add_output('angle', val=[90.0, 270.0])  # Follows windrose convention N = 0 deg, E = 90 deg, S = 180 deg,
        # W = 270 deg

        parallel = self.add_subsystem('parallel', Group())

        for n in range(2):
            for m in range(2):
                parallel.add_subsystem('wake{}'.format(2*n+m), FarmGroup(PowerPolynomial, JensenWakeFraction,
                                                                     JensenWakeDeficit, WakeMergeRSS),
                                       promotes_inputs=['original', 'n_turbines', 'r'])
                self.connect('indep2.angle', 'parallel.wake{}.wakemodel.angle'.format(2*n+m), src_indices=n)
                self.connect('indep2.freestream', 'parallel.wake{}.wakemodel.freestream'.format(2*n+m), src_indices=m)

        self.connect('indep2.layout', 'parallel.original')
        self.connect('indep2.n_turbines', 'parallel.n_turbines')
        self.connect('indep2.r', 'parallel.r')


def read_layout(layout_file):
    layout_file = open(layout_file, 'r')
    layout = []
    i = 0
    for line in layout_file:
        columns = line.split()
        layout.append([i, float(columns[0]), float(columns[1])])
        i += 1

    return np.array(layout)


prob = Problem()
prob.model = WorkingGroup(PowerPolynomial, JensenWakeFraction, JensenWakeDeficit, WakeMergeRSS)
prob.setup()
start = time()
prob.run_model()
print time() - start, "seconds"
for n in range(4):
    print [ind for ind in prob['parallel.wake{}.wakemodel.U'.format(n)] if ind > 0]
    print prob['parallel.wake{}.farmpower.farm_power'.format(n)]

view_model(prob)
# data = prob.check_totals(of=['farmpower.farm_power'], wrt=['indep2.k'])
# print data
# data = prob.check_partials(suppress_output=True)
# print(data['farmpower']['farm_power', 'ind_powers'])
# prob.model.list_outputs()
