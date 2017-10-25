from openmdao.api import Group, ExplicitComponent
from src.AbsWakeModel.wake_linear_solver import WakeModel
from src.AbsPower.abstract_power import FarmAeroPower
from src.AbsAEP.WindroseProcess import WindrosePreprocessor


class OneAngleFarmPower(Group):
    def __init__(self, power_model, fraction_model, deficit_model, merge_model):
        super(OneAngleFarmPower, self).__init__()
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


class PowersToAEP(ExplicitComponent):
    def __init__(self, windrose_cases):
        super(PowersToAEP, self).__init__()
        self.windrose_cases = windrose_cases

    def setup(self):
        self.add_input('powers', shape=self.windrose_cases)
        self.add_input('probabilities', shape=self.windrose_cases)

        self.add_output('energies', shape=self.windrose_cases)
        self.add_output('AEP', val=0.0)

