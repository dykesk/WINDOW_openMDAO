from openmdao.api import IndepVarComp, Problem, Group, view_model, ParallelGroup, PETScVector
from src.api import WakeModel
from WakeModel.jensen import JensenWakeFraction, JensenWakeDeficit
import numpy as np
from time import time, clock
from Power.power_models import PowerPolynomial
from input_params import turbine_radius
from WakeModel.WakeMerge.RSS import WakeMergeRSS
from src.api import AEPWorkflow

real_angle = 180.0
artificial_angle = 1.0
n_windspeedbins = 23


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
        indep2.add_output('layout', val=np.array([[0, 0.0, 0.0], [1, 0.0, 560.0], [2, 0.0, 1120.0], [3, 0.0, 1680.0]]))
        # indep2.add_output('layout', val=np.array(
        #     [[0, 0.0, 0.0], [1, 560.0, 560.0], [2, 1120.0, 1120.0], [3, 1120.0, 0.0], [4, 0.0, 1120.0],
        #     [5, 6666.6, 6666.6], [6, 6666.6, 6666.6], [7, 6666.6, 6666.6], [8, 6666.6, 6666.6], [9, 6666.6, 6666.6]]))
        # indep2.add_output('layout', val=np.array([[0, 0.0, 0.0], [1, 560.0, 560.0], [2, 1120.0, 1120.0],
        # [3, 1120.0, 0.0], [4, 0.0, 1120.0], [5, float('nan'), float('nan')]]))
        indep2.add_output('weibull_shapes', val=[1.0, 1.0])
        indep2.add_output('weibull_scales', val=[8.5, 8.5])
        indep2.add_output('dir_probabilities', val=[50.0, 50.0])
        indep2.add_output('wind_directions', val=[0.0, 180.0])  # Follows windrose convention N = 0 deg, E = 90 deg,
        #  S = 180 deg, W = 270 deg
        indep2.add_output('cut_in', val=4.0)
        indep2.add_output('cut_out', val=25.0)
        indep2.add_output('r', val=40.0)
        indep2.add_output('n_turbines', val=4)
        self.add_subsystem('AEP', AEPWorkflow(real_angle, artificial_angle, n_windspeedbins))

        self.connect('indep2.layout', 'AEP.original')
        self.connect('indep2.n_turbines', 'AEP.n_turbines')
        self.connect('indep2.r', 'AEP.r')
        self.connect('indep2.cut_in', 'AEP.cut_in')
        self.connect('indep2.cut_out', 'AEP.cut_out')
        self.connect('indep2.weibull_shapes', 'AEP.weibull_shapes')
        self.connect('indep2.weibull_scales', 'AEP.weibull_scales')
        self.connect('indep2.dir_probabilities', 'AEP.dir_probabilities')
        self.connect('indep2.wind_directions', 'AEP.wind_directions')


def read_layout(layout_file):
    layout_file = open(layout_file, 'r')
    layout = []
    i = 0
    for line in layout_file:
        columns = line.split()
        layout.append([i, float(columns[0]), float(columns[1])])
        i += 1

    return np.array(layout)

print clock(), "Before defining problem"
prob = Problem()
print clock(), "Before defining model"
prob.model = WorkingGroup(PowerPolynomial, JensenWakeFraction, JensenWakeDeficit, WakeMergeRSS)
print clock(), "Before setup"
# prob.setup(vector_class=PETScVector, check=True, mode='fwd')
prob.setup()
print clock(), "After setup"
prob.set_solver_print(level=0)
print clock(), "After solver print"
# view_model(prob)
start = time()
prob['indep2.cut_in'] = 4.1
print clock(), "Before 1st run"
prob.run_model()
print clock(), "After 1st run"
print time() - start, "seconds", clock()
print prob['AEP.AEP']

print "second run"
start = time()
prob['indep2.cut_in'] = 4.2
print clock(), "Before 2nd run"
prob.run_model()
print clock(), "After 2nd run"
print time() - start, "seconds", clock()
print prob['AEP.AEP']


print "third run"
start = time()
prob['indep2.cut_in'] = 4.3
print clock(), "Before 3rd run"
prob.run_model()
print clock(), "After 3rd run"
print time() - start, "seconds", clock()
print prob['AEP.AEP']

# for nn in range(8):
#     print prob['AEP.windrose.cases'][nn]
#     print [ind for ind in prob['AEP.parallel.farmpower{}.wakemodel.U'.format(nn)] if ind > 0]
#     print prob['AEP.parallel.farmpower{}.farmonepower.farm_power'.format(nn)]
# print prob['AEP.combine.combined_p']
# print
# print prob['AEP.energy.probabilities']
# print prob['AEP.energy.powers']
# print prob['AEP.energy.energies']

# data = prob.check_totals(of=['farmpower.farm_power'], wrt=['indep2.k'])
# print data
# data = prob.check_partials(suppress_output=True)
# print(data['farmpower']['farm_power', 'ind_powers'])
# prob.model.list_outputs()
