from openmdao.api import Group, ExplicitComponent, ParallelGroup
from src.AbsWakeModel.wake_linear_solver import WakeModel
from src.AbsPower.abstract_power import FarmAeroPower
from src.AbsAEP.WindroseProcess import WindrosePreprocessor
from Power.power_models import PowerPolynomial
from WakeModel.jensen import JensenWakeFraction, JensenWakeDeficit
from WakeModel.WakeMerge.RSS import WakeMergeRSS
from time import clock


class AEPWorkflow(Group):
    def __init__(self, real_angle, artificial_angle, n_windspeedbins):
        super(AEPWorkflow, self).__init__()
        self.real_angle = real_angle
        self.artificial_angle = artificial_angle
        self.n_windspeedbins = n_windspeedbins
        self.n_angles = 360.0 / self.artificial_angle
        self.n_windspeeds = n_windspeedbins + 1
        self.n_cases = self.n_angles * self.n_windspeeds

    def setup(self):
        self.add_subsystem('windrose', WindrosePreprocessor(self.real_angle, self.artificial_angle,
                                                            self.n_windspeedbins), promotes_inputs=['cut_in', 'cut_out',
                                                                                                    'weibull_shapes',
                                                                                                    'weibull_scales',
                                                                                                    'dir_probabilities',
                                                                                                    'wind_directions'])
        self.add_subsystem('parallel', Parallel(self.artificial_angle, self.n_windspeedbins),
                           promotes_inputs=['original', 'n_turbines', 'r'])
        self.add_subsystem('combine', CombinePowers(self.artificial_angle, self.n_windspeedbins))
        self.add_subsystem('energy', PowersToAEP(self.artificial_angle, self.n_windspeedbins),
                           promotes_outputs=['energies', 'AEP'])

        for n in range(int(self.n_cases)):
            self.connect('windrose.cases', 'parallel.farmpower{}.angle'.format(n), src_indices=[(n, 0)],
                         flat_src_indices=False)
            self.connect('windrose.cases', 'parallel.farmpower{}.freestream'.format(n), src_indices=[(n, 1)],
                         flat_src_indices=False)
            self.connect('parallel.farmpower{}.farm_power'.format(n), 'combine.p{}'.format(n))
        self.connect('combine.combined_p', 'energy.powers')
        self.connect('windrose.probabilities', 'energy.probabilities')


class CombinePowers(ExplicitComponent):
    def __init__(self, artificial_angle, n_windspeedbins):
        super(CombinePowers, self).__init__()
        self.n_angles = 360.0 / artificial_angle
        self.n_windspeeds = n_windspeedbins + 1
        self.n_cases = int(self.n_angles * self.n_windspeeds)

    def setup(self):
        for n in range(self.n_cases):
            self.add_input('p{}'.format(n), val=0.0)
        self.add_output('combined_p', shape=(self.n_cases, 1))

    def compute(self, inputs, outputs):
        outputs['combined_p'] = [inputs['p{}'.format(n)] for n in range(self.n_cases)]


class Parallel(Group):
    def __init__(self, artificial_angle, n_windspeedbins):
        super(Parallel, self).__init__()
        self.n_angles = 360.0 / artificial_angle
        self.n_windspeeds = n_windspeedbins + 1
        self.n_cases = self.n_angles * self.n_windspeeds
        # print self.n_cases

    def setup(self):
        for n in range(int(self.n_cases)):
            self.add_subsystem('farmpower{}'.format(n), OneAngleFarmPower(PowerPolynomial, JensenWakeFraction,
                                                                          JensenWakeDeficit, WakeMergeRSS),
                               promotes_inputs=['original', 'n_turbines', 'r'])


class OneAngleFarmPower(Group):
    def __init__(self, power_model, fraction_model, deficit_model, merge_model):
        super(OneAngleFarmPower, self).__init__()
        self.power_model = power_model
        self.merge_model = merge_model
        self.fraction_model = fraction_model
        self.deficit_model = deficit_model

    def setup(self):
        self.add_subsystem('wakemodel', WakeModel(self.fraction_model, self.deficit_model, self.merge_model),
                           promotes_inputs=['original', 'n_turbines', 'r', 'angle', 'freestream'])
        self.add_subsystem('power', self.power_model(), promotes_inputs=['n_turbines'])
        self.add_subsystem('farmonepower', FarmAeroPower(), promotes_inputs=['n_turbines'], promotes_outputs=['farm_power'])

        self.connect('wakemodel.U', 'power.U')
        self.connect('power.p', 'farmonepower.ind_powers')


class PowersToAEP(ExplicitComponent):
    def __init__(self, artificial_angle, n_windspeedbins):
        super(PowersToAEP, self).__init__()
        self.n_angles = 360.0 / artificial_angle
        self.n_windspeeds = n_windspeedbins + 1
        self.windrose_cases = int(self.n_angles * self.n_windspeeds)

    def setup(self):
        self.add_input('powers', shape=self.windrose_cases)
        self.add_input('probabilities', shape=self.windrose_cases)

        self.add_output('energies', shape=self.windrose_cases)
        self.add_output('AEP', val=0.0)

    def compute(self, inputs, outputs):
        powers = inputs['powers']
        probs = inputs['probabilities']
        energies = powers * probs * 8760.0
        outputs['energies'] = energies
        outputs['AEP'] = sum(energies)
        print clock(), "Last line compute AEP energies"
